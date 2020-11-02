# -*- coding:utf-8 -*-
#image_64code里面的编码：1-9是合同类型1。 后面的是合同类型2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import csv
import cv2
import base64
import datetime
import sys
import re
import collections
# from demo import comment_wps_demo, no_comment_wps_demo
from demo import flask_demo
type = sys.getfilesystemencoding()
# import ocr_deeptext_flask as ocr_deeptext
# import ocr_deeptext_2 as ocr_deeptext
# from preprocessing.img_detect_2 import LineCircleDetect
# from preprocessing.contract_information import get_information
import argparse
sys.path.append(os.path.abspath('.'))
from glob import glob
from flask import Flask, request, jsonify
app = Flask(__name__)


FLAGS = None
NERCH = False
POSINFERENCE = True

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN inference')
    parser.add_argument('--model', dest='model', help='Model path', default=None)  # required=True, default=None)
    parser.add_argument('--input_dir', dest='image_path', help='Directory where the inference images are',
                        default=None)
    parser.add_argument('--image_file', dest='image_file', help='Particuarl image file to run inference upon',
                        default=None)
    parser.add_argument('--output_dir', dest='output_dir', help='Where to put the image with bbox drawed',
                        default='img_output')  # , required=True
    parser.add_argument('--output_file', dest='output_file', help='', default=None)
    args = parser.parse_args()
    return args

def container_flask_demo(image_base64_code,):
    """Detect object classes in an image using pre-computed object proposals."""
    result = {}
    strs = image_base64_code
    decode_img = base64.b64decode(strs)  # 读取命令行中输入的参数，即base64字符串
    img_array = np.fromstring(decode_img, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    if img is None:
        msg = u"""其他-无法识别"""
        code = str(1004)
        data_result = ''
    else:
        h, w, c = img.shape
        imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
        if int(imageVar) < 50:
            msg = u"""图片模糊"""
            code = str(1003)
            data_result = ''
        elif h > 4096 or w > 4096:
            msg = u"""图片过大"""
            code = str(1002)
            data_result = ''
        else:
            data_result = flask_demo( image = img )
            msg = u"""识别成功"""
            code = str(1001)
    results = {"code": code, "msg": msg, "result": data_result}
    return results

# def demo_data(image_base64_code):
#     """Detect object classes in an image using pre-computed object proposals."""
#     strs = image_base64_code
#     decode_img = base64.b64decode(strs)  # 读取命令行中输入的参数，即base64字符串
#     img_array = np.fromstring(decode_img, np.uint8)  # 转换np序列
#     img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
#     # print(img)
#     # if len(im_files) > 1:
#     if img is None:
#         msg = u"""其他-无法识别"""
#         code = str(1004)
#         mtf_name = ""
#         mtf_tel = ""
#         mtf_item = ""
#     else:
#         h, w, c = img.shape
#         imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
#         if int(imageVar) < 50:
#             msg = u"""图片模糊"""
#             code = str(1003)
#             mtf_name = ""
#             mtf_tel = ""
#             mtf_item = ""
#         elif h > 4096 or w > 4096:
#             msg = u"""图片过大"""
#             code = str(1002)
#             mtf_name = ""
#             mtf_tel = ""
#             mtf_item = ""
#         else:
#             data_result = comment_wps_demo( image = img )
# #             # 先切图
# #             result = ocr_deeptext.model(img)
#             msg = u"""识别成功"""
#             code = str(1001)
# #             mtf_name = mtf
# #             mtf_tel = tel
# #             mtf_item = item
#     # results = {"msg": msg, "code": code, "data": data}
#     results = {"code": code, "msg": msg, "result": data_result}
#     return results


@app.route('/api/v1/ocr/contract', methods=['POST'])  # 添加路由
def http_data():
     #===================
    _image_code = request.form.get('image')
    _image_code = eval(_image_code)
    if not _image_code:
        print('----------_image_code is None!----->|   _image_code:|{}|   |<----'.format( _image_code ))
    result_json = container_flask_demo( image_base64_code = _image_code )
#     if product == "1" or product == 'pg.e2e.ii.cmnets':
#         result_json = demo_internet(_image_code)
#     elif product == '2' or product == 'pg.e2e.ii.cmnetcs':
#         result_json = demo_data(_image_code)
    return jsonify(result_json)

@app.route('/test', methods=['POST', 'GET'])  # 添加路由
def http_test():
    return jsonify("hello world!")

if __name__ == '__main__':
    app.config["JSON_AS_ASCII"] = False
    # 不排序输出
    app.config['JSON_SORT_KEYS'] = False
    app.run(host='0.0.0.0', port=4982, debug=False, use_reloader=False)
