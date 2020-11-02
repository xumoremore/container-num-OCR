#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2019/12/2714:30
#文件名称：draw_points.py

import os
from PIL import Image, ImageDraw
import shutil
import cv2
import numpy as np
image_save_dir = 'psenet_point_check_image/'

def makedir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
makedir( image_save_dir )
def draw_point( image, points ):
    try:
        image = Image.fromarray( image )
    except:
        pass
    imagedraw = ImageDraw.Draw( image )
    image_names = os.listdir(image_save_dir)
    image_name = "%03d"%(len(image_names)+1) + '.jpg'
    print('points:|{}|'.format( points))

    for point in points:
        point_x = point[::2]
        point_y = point[1::2]
        point_xy = list(zip(point_x, point_y))
        print(point_xy)
        imagedraw.polygon( point_xy, fill=None, outline=None,)
        # print("point_x:|{}|, point_y:|{}|".format( point_x, point_y))

    image_save_path = image_save_dir + image_name
    image.save(image_save_path)

if __name__ == '__main__':
    train_data_dir = 'data/psenet_train_data'
    train_datas = os.listdir( train_data_dir )
    image_names = train_datas[::2]
    txt_names = train_datas[1::2]

    for image_name, txt_name in zip( image_names, txt_names ):
        points = []
        print("===================|{}|========================".format( txt_name ))
        image_path = os.path.join( train_data_dir , image_name )
        txt_path = os.path.join( train_data_dir ,txt_name )
        image = Image.open( image_path )
        with open( txt_path, 'r', encoding = 'utf-8' ) as txt_file:
            txt_datas = txt_file.readlines()
        for txt_data in txt_datas:
            point = txt_data.split(',')[:8]
            point = list(map( float, point))
            points.append(point)
        print(points)

        draw_point( image=image, points=points )





