#-*- coding:utf-8 -*-
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import cv2
from math import *
import math
import numpy as np
from PIL import Image
# sys.path.append(os.getcwd() + '/tensorflow_PSENet')
# from densenet.model import predict as keras_densenet
from text_rec.predict import predict as deeptext_predict
import shutil

# sys.path.append(os.getcwd() + '/tf_yolov3')
def sort_box(box):
    """ 
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)#转换矩阵
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))#仿射变换
    print("imgRotation:")
#     print(imgRotation)
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]
    return imgOut

def select_pic(img,img_pt1,img_h):
      #过滤右下角红色水印
    if list(img_pt1)[1]>img_h*0.9:
        return np.array([])
    else:
        return img

def charRec(img, text_recs, adjust=False):
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])
        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度
        print(degree)
        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue
        if partImg.shape[0] == 0:
            pass
        else:
            image = Image.fromarray(partImg).convert('L')
            text = keras_densenet(image)
            if len(text) > 0:
                results[index] = [rec]
                results[index].append(text)  # 识别文字
            else:
                print('字符长度不够------')
    return results


def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4,
):
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    if angle > 45:
        angle = 90 -angle
        angle = -angle
    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]
    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    return imgOut  # rotated image

def charRecNew(img, text_recs, adjust_result_dir, original_image_name, adjust=False):
    xDim, yDim = img.shape[1], img.shape[0]
    image_list = []
    image_path_list = []
    text = []
    for index, rec in enumerate(text_recs):
        image_name =  original_image_name[:-4] + "_" + str(index) + '.jpg'
        adjust_result_path = os.path.join( adjust_result_dir, image_name )
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])
        partImg = rotate(img, pt1, pt2, pt3, pt4)
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue
        if partImg.shape[0] == 0:
            pass
        else:
            image = Image.fromarray(partImg)
            image_path_list.append(adjust_result_path)
            image_list.append(image)
            image.save( adjust_result_path )
        #print("------ocr.py image_list:{},-----image_path_list:{}".format( image_list, image_path_list ) )
    return image_list, image_path_list

def getaxis(point):
    x_point = point[::2]
    y_point = point[1:8:2]
    # x_float = [float(i) for i in x_point]
    # y_float = [float(i) for i in y_point]
    # xmin = int(min(x_float))
    # ymin = int(min(y_float))
    # xmax = int(max(x_float))
    # ymax = int(max(y_float))
    xmin = int(abs(floor(min(x_point))))
    xmax = int(abs(ceil(max(x_point))))
    ymin = int(abs(floor(min(y_point))))
    ymax = int(abs(ceil(max(y_point))))
    return (xmin, ymin, xmax, ymax)

def get_rotate_psenet_target(image, text_recs, adjust_result_dir, original_image_name, adjust=False ):
    ori_image = image
    # 计算旋转角度
    image_list = []
    image_path_list = []
    # results = {}
    xDim, yDim = ori_image.shape[1], ori_image.shape[0]
    for index, rec in enumerate(text_recs):
#         print("index:", index, "rec:", rec)
        if adjust:
            xlength = int((rec[6] - rec[0]) * 0.1)
            ylength = int((rec[7] - rec[1]) * 0.2)
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])
        withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
        heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
        # heightRect = math.sqrt((pt3[0] - pt1[0]) ** 2 + (pt3[1] - pt1[1]) ** 2)  # 矩形框的宽度
        # withRect = math.sqrt((pt4[0] - pt3[0]) ** 2 + (pt4[1] - pt3[1]) ** 2)
        #  angle = acos((pt4[0] - pt3[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
        flag = 1
        # if angle > 45:
        #     angle = 90 - angle  # 不同角度的进行测试
        if angle > 45:
            angle = -90 + angle
        elif angle < -45:
            angle = 90 + angle
        if abs(angle) < 10:  # 不旋转
            flag = 0
#         print("rotate_angle:", angle)
        img_xmin = np.min([pt1[0], pt2[0], pt3[0], pt4[0]])
        img_xmax = np.max([pt1[0], pt2[0], pt3[0], pt4[0]])
        img_ymin = np.min([pt1[1], pt2[1], pt3[1], pt4[1]])
        img_ymax = np.max([pt1[1], pt2[1], pt3[1], pt4[1]])
        angle_limited = 13
        img_xmin_adj = img_xmin if abs(angle) < angle_limited else img_xmin + 15
        img_xmax_adj = img_xmax if abs(angle) < angle_limited else img_xmax - 15
        img_ymin_adj = img_ymin if abs(angle) < angle_limited else img_ymin + 5
        img_ymax_adj = img_ymax if abs(angle) < angle_limited else img_ymax + 5
#         print("image_xmin_adj:|{}|,image_xmax_adj:|{}|,image_ymin_adj:|{}|,image_ymax_adj:|{}|".format( img_xmin_adj, img_xmax_adj,
#                                                                    img_ymin_adj, img_ymax_adj))
        img = ori_image[img_ymin_adj: img_ymax_adj, img_xmin_adj: img_xmax_adj]
#         print("rotate and crop img:|{}|".format( img ))
        if flag == 1:
            # print("rotate_angle:", angle)
            # # 旋转
            # # crop_img = rotate_H(img, angle)
            # crop_img = rotate_dst(img, angle)
            height = img.shape[0]
            width = img.shape[1]
            hr = height
            wr = width
            center_w = int(width / 2)
            center_h = int(height / 2)
            rotateMat = cv2.getRotationMatrix2D((center_w, center_h), angle, 1)  # 按angle角度旋转图像
            heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
            widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
            rotateMat[0, 2] += (widthNew - width) // 2
            rotateMat[1, 2] += (heightNew - height) // 2
            try:
                imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
                n_h, n_w = imgRotation.shape[:2]
                n_center_w = n_w / 2
                n_center_h = n_h / 2
                w_eps = wr / 10
                xmin = int(abs(floor(n_center_w - (wr / 2) - w_eps)))
                xmax = int(abs(ceil(n_center_w + (wr / 2) + w_eps)))
                # 坐标调整
                h_eps = hr / 10 + 5
                ymin = int(abs(floor(n_center_h - (hr / 2) - h_eps)))
                ymax = int(abs(ceil(n_center_h + (hr / 2) + h_eps)))
                crop_img = imgRotation[ymin:ymax, xmin:xmax]
            except:
                print("--！----！----！--用psenet坐标做图片旋转错误")
                print("image.shape:|{}|, rotatemat:|{}|, (widthnew,heightnew):|{}|".format(img, rotateMat, (widthNew, heightNew)))
                crop_image = img
        else:
            crop_img = img
        # partImg = rotate_Hough(img, pt1, pt2, pt3, pt4)
#         if crop_img.shape[0] < 1 or crop_img.shape[1] < 1:  # or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
#             print("图片异常")
#             continue
        if crop_img.shape[0] == 0:
            print("crop_img:|{}|".format(crop_img))
            print("crop_image 等于零")
            pass
        else:
            # image = Image.fromarray(crop_img).convert('L')
            # text = keras_densenet(image)
            image_name =  original_image_name.replace(".", "") + "_" + str(index) + '.jpg'
            adjust_img_path = os.path.join( adjust_result_dir, image_name )
            # print(adjust_img_path)
#             image = Image.fromarray(crop_img).convert('L')
            image = Image.fromarray(crop_img)
            # print("保存调整charrecnew里面调整过后的照片-----------")
#             print("-------------------asjust_image_path:|{}|".format( adjust_img_path ))
            #@@@@@@
            # image.save(adjust_img_path)         #处理十万图片不输出
            image_path_list.append(adjust_img_path)
            image_list.append(image)
    return image_list, image_path_list


def model(image, text_recs, adjust_result_dir, original_image_name, adjust=False):
    """
    @img: 图片(是已经读取后的数据)
    @adjust: 是否调整文字识别结果
    """
#     image_list, image_path_list = charRecNew(img, text_recs, adjust_result_dir, original_image_name, adjust=False)
#     for point in text_recs:
#         gen_point = gen_point_not_predicted( base_point=point )
#     print("model text_recs:|{}|".format( text_recs ))
    image_list, image_path_list = get_rotate_psenet_target( image, text_recs, adjust_result_dir, original_image_name )
#     image_list, image_path_list = get_text_recs_image(img, text_recs, adjust_result_dir, original_image_name,)
    text,cfd_scores = deeptext_predict( image_list, image_path_list )
    return text, cfd_scores



    


    