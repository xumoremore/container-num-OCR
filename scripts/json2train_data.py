#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2020/1/1819:29
#文件名称：json2train_data.py

import os
from PIL import Image
import cv2
import base64
import json
from draw_points import draw_point
import shutil

def get_rotate_psenet_target(image, text_recs, adjust_result_dir, original_image_name, adjust=False):
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

        # print("rotate_angle:", angle)

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
                print("image.shape:|{}|, rotatemat:|{}|, (widthnew,heightnew):|{}|".format(img, rotateMat,
                                                                                           (widthNew, heightNew)))
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
            image_name = original_image_name.replace(".", "") + "_" + str(index) + '.jpg'
            adjust_img_path = os.path.join(adjust_result_dir, image_name)

            # print(adjust_img_path)
            #             image = Image.fromarray(crop_img).convert('L')
            image = Image.fromarray(crop_img)
            # print("保存调整charrecnew里面调整过后的照片-----------")
            #             print("-------------------asjust_image_path:|{}|".format( adjust_img_path ))
            image.save(adjust_img_path)
            image_path_list.append(adjust_img_path)

            image_list.append(image)

    return image_list, image_path_list


def point_format_change( json_point ):
    psenet_point = []
    for xy in json_point:
        psenet_point.append( xy[0] )
        psenet_point.append( xy[1] )
    psenet_point = list( map(str, psenet_point ))
    return psenet_point

def getaxis(point):
    x_point = point[::2]
    y_point = point[1:8:2]
    x_float = [float(i) for i in x_point]
    y_float = [float(i) for i in y_point]
    xmin = int(min(x_float))
    ymin = int(min(y_float))
    xmax = int(max(x_float))
    ymax = int(max(y_float))
    return (xmin, ymin, xmax, ymax)

def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def json2point():
    
    json_dir = 'data/box_image_json_objectid2/json'
    ori_image_dir = 'data/box_image_json_objectid2/image'
    psenet_result_dir = './data/psenet_td'
    deeptext_result_dir = './data/deeptext_td'
    makedir(json_dir)
    makedir(ori_image_dir)
    makedir(psenet_result_dir)
    makedir(deeptext_result_dir)
    deeptext_label_path = os.path.join(deeptext_result_dir, 'tmp_labels.txt')
    deeptext_label_file = open( deeptext_label_path, 'w', encoding='utf-8')
    json_names = os.listdir( json_dir )
    for json_name in json_names:
        print("===================|{}|=======================".format( json_name ))
        image_name = json_name.replace( '.json', '.jpg' )
        txt_name = json_name.replace( '.json', '.txt')
        ori_image_path = os.path.join( ori_image_dir, image_name )
        image = Image.open( ori_image_path )
        psenet_image_save_path = os.path.join( psenet_result_dir, image_name )
        image.save( psenet_image_save_path )
        json_path = os.path.join( json_dir, json_name )

        psenet_txt_path = os.path.join( psenet_result_dir, txt_name )
        psenet_txt_file = open( psenet_txt_path, 'w', encoding='utf-8')

        with open( json_path, 'r') as json_file:
            json_result = json.load( json_file )
        json_shapes = json_result['shapes']
        for index, shape in enumerate(json_shapes):
            crop_image_name = image_name.replace( '.jpg', '_%s.jpg'%index)
            label = shape['label']
            json_point = shape['points']
            psenet_point = point_format_change( json_point = json_point )
            print("psenet_point:|{}|".format( psenet_point ))
            psenet_label = ','.join(psenet_point) + "," + label
            psenet_txt_file.write( psenet_label + '\n')

            crop_axis = getaxis( point= psenet_point )
            crop_image_save_path = os.path.join( deeptext_result_dir, crop_image_name )
            crop_image = image.crop( crop_axis )
            crop_image.save( crop_image_save_path )
            deeptext_label_file.write( crop_image_name + " " + label + '\n')
    deeptext_label_file.close()
    psenet_txt_file.close()

if __name__ == '__main__':
    json2point()


