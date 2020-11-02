#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2020/1/1810:34
#文件名称：ocr2labelme.py

import os
from PIL import Image
import cv2
import base64
import json
import shutil

def point_format_change( psenetpoint ):
    x_point = [ i for i in psenetpoint[::2]]
    y_point = [ i for i in psenetpoint[1::2]]

    labelme_point = []
    for xy in zip( x_point, y_point):
        labelme_point.append( [xy[0], xy[1]] )

    pt2 = labelme_point[2]
    labelme_point[2] = labelme_point[3]
    labelme_point[3] = pt2

    return labelme_point


def ocr2labelmejson( texts, points, imagepath, correct_flag='flag'):
    # print("----texts:|{}|-------\n---------points:|{}|----------\n".format(texts, points))

    json_save_dir = 'scripts/data/%s_labelme_image_json/json/'%correct_flag
    image_save_dir = 'scripts/data/%s_labelme_image_json/image/'%correct_flag
    labelme_format = {"shapes": [],
                      "lineColor": [0, 255, 0, 128],
                      "fillColor": [255, 0, 0, 128],
                      "imagePath": "..\\JPEGImages\\001.jpg",
                      "imageData": "",  # base64码
                      "imageHeight": 720,
                      "imageWidth": 1280
                      }


    image_basename = os.path.basename( imagepath )
    json_basename = image_basename.split('.')[0] + '.json'
    # texts_points = zip( texts, points )
    for text, point in zip( texts, points):
        single_shape_dict = { "label":"text", "line_color":None, "fill_color":None, "points":""}
        single_shape_dict['label'] = text
        single_shape_dict['points'] = point_format_change( psenetpoint=point )
        labelme_format['shapes'].append( single_shape_dict )
    labelme_format['imagePath'] = imagepath
    image_read = open( imagepath, "rb").read()
    image_code64 = base64.b64encode( image_read ).decode('utf-8')
    labelme_format['imageData'] = image_code64
    image = Image.open(imagepath)
    labelme_format['imageWidth'] = image.width
    labelme_format['imageHeight'] = image.height
    image_save_path = os.path.join( image_save_dir, image_basename )
    image.save( image_save_path )
    json_save_path = os.path.join( json_save_dir, json_basename )
    json_file = open( json_save_path, 'w')
    labelme_json_result = json.dumps(labelme_format, indent=4)
    json_file.write( labelme_json_result)


