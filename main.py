import sys, os
sys.path.insert(0, '/home/ubuntu/anaconda3/envs/chainer_p36/lib/python3.6/site-packages/onnx/defs/object_detection')
# sys.path.append("/home/ubuntu/anaconda3/envs/chainer_p36/lib/python3.6/site-packages/onnx/defs/object_detection")

from client_photolab import ClientPhotolab
import os.path
import imageio
from flask import Flask, Response, request, render_template, send_file
import pandas as pd
import cv2
import numpy as np

from PoseAnimator import process_image, get_anim
import urllib.request

app = Flask(__name__,static_url_path='/output/')
animation = pd.read_csv('animation_moves.csv',header = None)

UPLOAD_FOLDER = "input/"
# UPLOAD_FOLDER = "output/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html", title = 'Photohack')
    # return app.send_static_file('index.html')

@app.route('/magic', methods=['POST'])
def magic():
    if 'image' not in request.files:
        return "no file error"
    file = request.files['image']
    if file.filename == '':
        return "no file error"
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # TODO костыль с фоном

    api = ClientPhotolab()
    content_url = "http://3.121.100.121:8888/input/" + filename
    result_url = api.template_process("1001526", [{
        'url': content_url,
        'rotate': 0,
        'flip': 0,
        'crop': '0,0,1,1'
    }])
    resp = urllib.request.urlopen(result_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(image, mask, (0, 0), (0, 0, 0, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    print(image)
    print(image.shape)

    #backgroundImg = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # B,G,R order
    backgroundImg = image.copy()
    backgroundImg = cv2.cvtColor(backgroundImg, cv2.COLOR_RGBA2BGRA)
    # backgroundImg = cv2.imread("sample_images/stas3.png", cv2.IMREAD_UNCHANGED)  # B,G,R order
    backgroundImg *= 0
    # body_parts = process_image(filepath)
    body_parts = process_image(image)
    # body_parts = process_image("sample_images/stas3.png")
    frame_count = 200
    #print(body_parts)
    gif_name = 'output/' + filename + ".gif"
    animations = get_anim(frame_count, body_parts, animation, backgroundImg)

    with imageio.get_writer(gif_name, mode='I', fps=80) as writer:
        for i in range(0,len(animations), 8):
            frame = cv2.cvtColor(animations[i], cv2.COLOR_RGBA2BGRA)
            cv2.imwrite("output/" + str(i) + "_" + filename, frame)
            writer.append_data(animations[i])

    # make gif
    # with imageio.get_writer('output/' + filename + ".gif", mode='I') as writer:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    return gif_name

@app.route('/output/<name>')
def image_output(name=None):
    return send_file('output/' + name)

@app.route('/input/<name>')
def image_input(name=None):
    return send_file('input/' + name)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)
