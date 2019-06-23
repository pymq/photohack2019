PIPELINE:

import pandas as pd
data = pd.read_csv('animation_moves.csv')
body_parts = process_image(img, path_to_model) #Изображение человека и путь к модели (по дефолту model/keras/model.h5)
get_anim(frame_count,body_parts,data,background) #Frame_count - количество кадров в анимации, data - pandas.DataFrame. Background - ищображение фона. Вроде закостылено на то, что оно должно иметь ту же форму что и img