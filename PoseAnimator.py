import keras
import scipy
import math
import PoseEstimatonModel
import cv2
import gc
import matplotlib
import pylab as plt
import numpy as np
# from config_reader import confing_reader
from configobj import ConfigObj


def config_reader():
    config = ConfigObj('config')

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = list(map(float, param['scale_search']))
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model

import util

from scipy.ndimage.filters import gaussian_filter

def calculate_start_angle(x0,y0,x1,y1):
    return math.degrees(math.atan2(x0 - x1, y0 - y1))

def calculate_len(a,b):
    return np.linalg.norm((a[0]-b[0],a[1]-b[1]))

def get_subregion(img,rect):
    print(rect)
    return img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]


class body_part:
    def __init__(self, img, region_dict, childs=list(), copy=False):
        if copy:
            self.region_dict = region_dict.copy()
            self.pivot = self.region_dict['pivot']
            self.rect = self.region_dict['rect']
            self.angle = self.region_dict['angle']
            self.length = self.region_dict['length']
            self.img = img.copy()
            self.childs = childs
        else:
            self.region_dict = region_dict
            temp_img = img.copy()
            temp_img *= 0
            temp_img[region_dict['rect'][0][1]:region_dict['rect'][1][1],
            region_dict['rect'][0][0]:region_dict['rect'][1][0], :] = self.__get_subregion(img,
                                                                                           region_dict['rect']).copy()
            self.img = temp_img
            self.childs = childs
            self.pivot = region_dict['pivot']
            self.rect = region_dict['rect']
            self.angle = region_dict['angle']
            self.length = region_dict['length']

    def copy(self, childs):
        return body_part(self.img, self.region_dict, childs, True)

    def __get_subregion(self, img, rect):
        return img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    def __update_childs(self):
        end_pos = (self.pivot[0] + np.sin((180 + self.angle) * np.pi / 180.) * self.length,
                   self.pivot[1] + np.cos((180 + self.angle) * np.pi / 180.) * self.length)
        for child in self.childs:
            child.move(end_pos)

    def __rotateImage(self, delta_angle):
        rot_mat = cv2.getRotationMatrix2D(self.pivot, delta_angle, 1.0)
        self.img = cv2.warpAffine(self.img, rot_mat, self.img.shape[1::-1], borderMode=cv2.BORDER_TRANSPARENT)

    def __moveImage(self, delta_pos):
        rows, cols, _ = self.img.shape
        M = np.float32([[1, 0, delta_pos[0]], [0, 1, delta_pos[1]]])
        self.img = cv2.warpAffine(self.img, M, (cols, rows), borderMode=cv2.BORDER_TRANSPARENT)

    def draw(self, output):
        result = np.zeros(output.shape, np.uint8)
        alpha = self.img[:, :, 3] / 255.0
        result[:, :, 0] = (1. - alpha) * output[:, :, 0] + alpha * self.img[:, :, 0]
        result[:, :, 1] = (1. - alpha) * output[:, :, 1] + alpha * self.img[:, :, 1]
        result[:, :, 2] = (1. - alpha) * output[:, :, 2] + alpha * self.img[:, :, 2]
        result[:, :, 3] = np.clip((1. - alpha) * output[:, :, 3] + alpha * self.img[:, :, 3], 0, 255)
        output = result
        for child in self.childs:
            output = child.draw(output)
        return output

    def rotate(self, angle):
        delta_angle = angle - self.angle
        self.angle = angle
        self.__rotateImage(delta_angle)
        self.__update_childs()

    def move(self, new_pivot_pos):
        delta = (new_pivot_pos[0] - self.pivot[0], new_pivot_pos[1] - self.pivot[1])
        self.pivot = (self.pivot[0] + delta[0], self.pivot[1] + delta[1])
        self.__moveImage(delta)
        self.__update_childs()

def process_image(test_image, model_path = 'model/keras/model.h5'):
    # test_image = 'sample_images/stas3.png' # REMOVE
    # workImg = cv2.imdecode(test_image, cv2.IMREAD_UNCHANGED)  # B,G,R order
    #workImg = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGRA)
    #print(test_image.shape)
    workImg = test_image
    oriImg = cv2.cvtColor(workImg, cv2.COLOR_RGBA2BGR)

    model = PoseEstimatonModel.load_model(model_path)

    param, model_params = config_reader()
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    del heatmap, paf
    print(gc.collect())

    all_peaks = []
    peak_counter = 0

    for part in range(19 - 1):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
              [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
              [55, 56], [37, 38], [45, 46]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    print(gc.collect())
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    print(subset) # REMOVE
    print("========") # REMOVE
    print(np.array(limbSeq[12])) # REMOVE
    print("========") # REMOVE
    print(all_peaks) # REMOVE
    img = workImg.copy()

    # Y = candidate[0, 0]
    # X = candidate[0, 1]
    # Head
    index = subset[0][np.array(limbSeq[12]) - 1]
    X = candidate[index.astype(int), 1]
    mY = np.mean(X) + 10
    headline = ((0, int(mY)), (img.shape[1], int(mY)))
    # LeftHand
    lhandline = ((all_peaks[3][0][0], 0), (all_peaks[3][0][0], img.shape[0]))
    #cv2.line(img, *lhandline, color=(255, 0, 255, 255))

    # RightHand
    rhandline = ((all_peaks[6][0][0], 0), (all_peaks[6][0][0], img.shape[0]))
    #cv2.line(img, *rhandline, color=(255, 0, 255, 255))

    # BottomLine
    index = subset[0][np.array(limbSeq[12]) - 1]
    mx = np.mean((all_peaks[11][0][1], all_peaks[8][0][1]))
    #print(X)
    bottomline = ((0, int(mx),), (img.shape[1], int(mx)))
    #cv2.line(img, *bottomline, color=(255, 0, 255, 255))

    # MidLine
    index = subset[0][np.array(limbSeq[12]) - 1]
    #Y = candidate[index.astype(int), 0]
    mx1 = np.mean((all_peaks[10][0][0], all_peaks[9][0][0]))
    mx2 = np.mean((all_peaks[12][0][0], all_peaks[13][0][0]))
    mx = np.mean((mx1, mx2))
    #print(X)
    midline = ((int(mx), 0), (int(mx), img.shape[0]))
    #cv2.line(img, *midline, color=(255, 0, 255, 255))

    # KneeLine
    index = subset[0][np.array(limbSeq[12]) - 1]
    #Y = candidate[index.astype(int), 0]
    mx = np.mean((all_peaks[12][0][1], all_peaks[9][0][1]))
    #print(X)
    kneeline = ((0, int(mx),), (img.shape[1], int(mx),))
    #cv2.line(img, *kneeline, color=(255, 0, 255, 255))

    # LeftSholder
    lsholderline = ((all_peaks[2][0][0] - 10, 0), (all_peaks[2][0][0] - 10, img.shape[0]))
    #cv2.line(img, *lsholderline, color=(255, 0, 255, 255))

    # RightSholder
    rsholderline = ((all_peaks[5][0][0] + 10, 0), (all_peaks[5][0][0] + 10, img.shape[0]))
    #cv2.line(img, *rsholderline, color=(255, 0, 255, 255))

    # bodymiddleSholder
    mx = np.mean((headline[0][1], bottomline[0][1]))
    bodymiddle = ((0, int(mx)), (img.shape[1], int(mx)))
    #cv2.line(img, *bodymiddle, color=(255, 0, 255, 255))

    img = workImg.copy()

    body = {'rect': (
    (int(lsholderline[0][0] - 10), int(headline[0][1])), (int(rsholderline[0][0] + 10), int(bottomline[0][1]) + 15)),
            'pivot': all_peaks[1][0][0:2], 'angle': calculate_start_angle(*all_peaks[1][0][0:2], *all_peaks[0][0][0:2]),
            'length': calculate_len(all_peaks[1][0][0:2], all_peaks[0][0][0:2])}

    head = {'rect': ((int(lsholderline[0][0]), 0), (int(rsholderline[0][0]), int(headline[0][1]))),
            'pivot': all_peaks[1][0][0:2],
            'angle': calculate_start_angle(*all_peaks[1][0][0:2], *(midline[0][0], bottomline[0][1])),
            'length': calculate_len(all_peaks[1][0][0:2], (midline[0][0], bottomline[0][1]))}

    lsholder = {'rect': (
    (int(lhandline[0][0]) - 5, int(headline[0][1]) - 15), (int(lsholderline[0][0]), int(bodymiddle[0][1]) + 20)),
                'pivot': all_peaks[2][0][0:2],
                'angle': calculate_start_angle(*all_peaks[2][0][0:2], *all_peaks[3][0][0:2]),
                'length': calculate_len(all_peaks[2][0][0:2], all_peaks[3][0][0:2])}

    rsholder = {'rect': (
    (int(rsholderline[0][0]) - 5, int(headline[0][1]) - 15), (int(rhandline[0][0]) + 5, int(bodymiddle[0][1]) + 20)),
                'pivot': all_peaks[5][0][0:2],
                'angle': calculate_start_angle(*all_peaks[5][0][0:2], *all_peaks[6][0][0:2]),
                'length': calculate_len(all_peaks[5][0][0:2], all_peaks[6][0][0:2])}

    lhand = {'rect': ((0, int(headline[0][1]) - 15), ((int(lhandline[0][0]), int(bodymiddle[0][1]) + 20))),
             'pivot': all_peaks[3][0][0:2],
             'angle': calculate_start_angle(*all_peaks[3][0][0:2], *all_peaks[4][0][0:2]),
             'length': calculate_len(all_peaks[3][0][0:2], all_peaks[4][0][0:2])}

    rhand = {'rect': ((int(rhandline[0][0]), int(headline[0][1]) - 15), (img.shape[1], int(bodymiddle[0][1]) + 20)),
             'pivot': all_peaks[6][0][0:2],
             'angle': calculate_start_angle(*all_peaks[6][0][0:2], *all_peaks[7][0][0:2]),
             'length': calculate_len(all_peaks[6][0][0:2], all_peaks[7][0][0:2])}

    lknee = {'rect': ((int(lhandline[0][0]), int(bottomline[0][1]) - 10), (int(midline[0][0]), int(kneeline[0][1]))),
             'pivot': all_peaks[8][0][0:2],
             'angle': calculate_start_angle(*all_peaks[8][0][0:2], *all_peaks[9][0][0:2]),
             'length': calculate_len(all_peaks[8][0][0:2], all_peaks[9][0][0:2])}

    rknee = {'rect': ((int(midline[0][0]), int(bottomline[0][1]) - 10), (int(rhandline[0][0]), int(kneeline[0][1]))),
             'pivot': all_peaks[11][0][0:2],
             'angle': calculate_start_angle(*all_peaks[11][0][0:2], *all_peaks[12][0][0:2]),
             'length': calculate_len(all_peaks[11][0][0:2], all_peaks[12][0][0:2])}

    lleg = {'rect': ((int(lhandline[0][0]), int(kneeline[0][1]) - 10), (int(midline[0][0]), img.shape[0])),
            'pivot': all_peaks[9][0][0:2],
            'angle': calculate_start_angle(*all_peaks[9][0][0:2], *all_peaks[10][0][0:2]),
            'length': calculate_len(all_peaks[9][0][0:2], all_peaks[10][0][0:2])}

    rleg = {'rect': ((int(midline[0][0]), int(kneeline[0][1]) - 10), (int(rhandline[0][0]), img.shape[0])),
            'pivot': all_peaks[12][0][0:2],
            'angle': calculate_start_angle(*all_peaks[12][0][0:2], *all_peaks[13][0][0:2]),
            'length': calculate_len(all_peaks[12][0][0:2], all_peaks[13][0][0:2])}

    llegBP = body_part(workImg, lleg)
    rlegBP = body_part(workImg, rleg)

    lkneeBP = body_part(workImg, lknee, [llegBP])
    rkneeBP = body_part(workImg, rknee, [rlegBP])

    lhandBP = body_part(workImg, lhand)
    rhandBP = body_part(workImg, rhand)

    lsholderBP = body_part(workImg, lsholder, [lhandBP])
    rsholderBP = body_part(workImg, rsholder, [rhandBP])

    headBP = body_part(workImg, head)

    bodyBP = body_part(workImg, body, [lkneeBP, rkneeBP, headBP, lsholderBP, rsholderBP])

    return [rsholderBP, rhandBP, headBP, bodyBP, lsholderBP, lhandBP, rkneeBP, rlegBP, lkneeBP, llegBP]


def angle_transform(angle):
    return 360 - angle - 90

def get_anim(frame_count,body_parts,data,background):
    result = [None]*frame_count
    for frame in range(frame_count):
        result[frame] = get_frame(background,body_parts,data,frame)

    return result

def rebuild_body_parts(body_parts):
    # [rsholderBP, rhandBP, headBP, bodyBP, lsholderBP, lhandBP, rkneeBP, rlegBP, lkneeBP, llegBP]
    llegBP = body_parts[9].copy(list())
    rlegBP = body_parts[7].copy(list())

    lkneeBP = body_parts[8].copy([llegBP])
    rkneeBP = body_parts[6].copy([rlegBP])

    lhandBP = body_parts[5].copy(list())
    rhandBP = body_parts[1].copy(list())

    lsholderBP = body_parts[4].copy([lhandBP])
    rsholderBP = body_parts[0].copy([rhandBP])

    headBP = body_parts[2].copy(list())

    bodyBP = body_parts[3].copy([lkneeBP, rkneeBP, headBP, lsholderBP, rsholderBP])

    return [rsholderBP, rhandBP, headBP, bodyBP, lsholderBP, lhandBP, rkneeBP, rlegBP, lkneeBP, llegBP]


def get_frame(background, body_parts, data, frame):
    frame_bones = rebuild_body_parts(body_parts)
    t = data[data[0] == frame]
    for bone in t.iterrows():
        if not (bone[1][1] == 3 or bone[1][1] == 2):
            frame_bones[bone[1][1]].rotate(angle_transform(bone[1][2]))
    return frame_bones[3].draw(background)





