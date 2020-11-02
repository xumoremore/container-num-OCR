import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import csv
import cv2
import sys
# from yolov3.yolov3_demo import yolov3_predict
from text_detect.eval import psenet_predict
from deeptext.predict import predict as deeptext_predict
type = sys.getfilesystemencoding()
from get_container_num import container_num_check
import get_container_num
from scripts.statistics import statistics
from scripts.ocr2labelme import ocr2labelmejson
import math
import json





"""
psenet训练数据
标准号：50000
长号:25000*2
年后调整为15000*2
datapower的强度也需要再调小
"""



def makedir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def crop_image( bbox, image ):
    objectid = int(bbox[5])
    if objectid != 0:
        xmin = abs(int(bbox[0]) - 20)
        ymin = abs(int(bbox[1]) - 20)
        xmax = abs(int(bbox[2]) + 20)
        ymax = abs(int(bbox[3]) + 20)
    else :
        xmin = abs(int(bbox[0]) - 10)
        ymin = int(bbox[1])
        xmax = abs(int(bbox[2]) + 10)
        ymax = int(bbox[3])
    confidence = int(bbox[4])
    crop_axis = (xmin, ymin, xmax, ymax)
    crop_image = image[ ymin:ymax, xmin:xmax ]
    height,width = crop_image.shape[:2]
    # crop_image = cv2.resize( crop_image, (int(0.9*height),int(0.9*width)))
#     if objectid == 2 or objectid == 3:
#         crop_image = rotate_bound_white_bg( image=crop_image, angle=270)
    return crop_image, confidence, objectid, crop_axis 

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    image = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    return image

def rotate_xy( center_xy, wait_rot_xy, angle):
    valuex = wait_rot_xy[0]
    valuey = wait_rot_xy[1]
    pointx = center_xy[0]
    pointy = center_xy[1]
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    # y_len = pointy-valuey
    # x_len = pointx-valuex
    # xiebian = math.sqrt( y_len*y_len + x_len*x_len )
    # x_dec = x_len - xiebian*math.cos( angle )
    # y_dec = x_len - xiebian*math.sin( angle )
    # sRotatex = x_len + x_dec
    # sRotatey = y_len + y_dec
    angle = angle*math.pi/180
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return int(sRotatex), int(sRotatey)

def rotate_psenet_points( image, text1_points, angle ):
    # center_xy = (np.shape( image )[1]//2, np.shape(image)[0]//2)
    bg_image = Image.fromarray(image)
    rot_angle = angle
    center_xy = (bg_image.size[0]//2, bg_image.size[1]//2)
    rot_text1_points = []
    # print(" text_points:|{}|".format( text_points ))
    for point in text1_points:
        # print("point:|{}|".format( point ))
        rot_point = []
        for x,y in zip(point[0::2], point[1::2]):
            wait_rotxy = (x, y)
            # print( 'wait_rotxy:|{}|'.format( wait_rotxy ))
            rotated_xy = rotate_xy(center_xy, wait_rotxy, rot_angle)
            # print("rotated_xy:|{}|".format( rotated_xy ))
            # image_draw.point(( rotated_xy[0], rotated_xy[1]), 'red')
            rot_point.append( rotated_xy[0] )
            rot_point.append( rotated_xy[1] )
        rot_text1_points.append( rot_point )
    return rot_text1_points

test_image_dir = './datas/all_photos/'
yolov3_result_dir = './datas/yolov3_result/'
makedir(yolov3_result_dir)
psenet_result_dir = './datas/psenet_result/'
makedir(psenet_result_dir)
psenet_points_result_path = './datas/psenet_result/psenet_points.txt'
ocr_result_path = './datas/ocr_result.txt'
adjust_main_dir = "datas/psenet2ocr_adjust/"
container_num_path = './datas/container_num.txt'
error_station_path = './datas/error_station.txt'
result2json_dir = './datas/ocr_json_result'
error_detect_result_path = './datas/error_detect_result.txt'
error_detect_image_dir = './datas/error_detect_image/'
yolo_corp_image_dir = './datas/yolo_crop_image'
makedir( yolo_corp_image_dir)
makedir( result2json_dir )
makedir( error_detect_image_dir )
makedir(adjust_main_dir)
error_det_result_file = open( error_detect_result_path, 'w', encoding='UTF-8')
error_file = open( error_station_path, 'w', encoding='UTF-8')
psenet_point_file = open( psenet_points_result_path, 'w', encoding='UTF-8')
ocr_result_file = open(ocr_result_path, 'w', encoding='UTF-8')
container_num_file = open(container_num_path, 'w', encoding='UTF-8')


def gen_point_not_predicted(psenet_points, direct=0):
    """
    direct:代表获取的方向，0为水平方向
    p1-----p2----gen_p1
    |     |ssssssss
    p3-----p4
    """
    psenet_points = list( psenet_points )
    if len(psenet_points) == 3:
        len_texts = []
        for point in psenet_points:
            point_x = point[::2]
            len_text = max(point_x) - min(point_x)
            len_texts.append(len_text)
        base_point = psenet_points[len_texts.index(max(len_texts))]
        print("gen point!!!!!!!!!!!!!!!!!!!!")
    else:
        print("do not gen point!!!!!!!!!!!!!!!!!!!")
        return psenet_points
    pt1 = (max(1, base_point[0]), max(1, base_point[1]))
    pt2 = (base_point[2], base_point[3])
    pt3 = (base_point[6], base_point[7])
    pt4 = (base_point[4], base_point[5])
    # widthRect = math.sqrt((pt3[0] - pt4[0]) ** 2 + (pt3[1] - pt4[1]) ** 2)  # 矩形框的宽度
    # heightRect = math.sqrt((pt3[0] - pt4[0]) ** 2 + (pt3[1] - pt4[1]) ** 2)
    angle_width = pt3[0] - pt4[0]
    angle_height = pt3[1] - pt4[1]
    xiebian = math.sqrt( angle_width ** 2 + angle_height ** 2 )
    # angle = math.acos((pt4[0] - pt1[0]) / widthRect) * (180 / math.pi)  # 矩形框旋转角度
    angle = math.acos(angle_width/xiebian) * (180 / math.pi)  # 矩形框旋转角度
    heightRect = math.sqrt((pt3[0] - pt2[0]) ** 2 + (pt3[1] - pt2[1]) ** 2)
    gen_point_width = heightRect
    gen_point_height = heightRect
#     print("gen_point_width:|{}|".format( gen_point_width ))
#     print("angle--:|{}|".format(angle))
    if direct:
        pass
        #可能有错
        # height_distance = heightRect / 2
        # gen_pt1 = (pt3[0] + height_distance * math.sin(angle*(math.pi/180)), pt3[1] + height_distance * math.cos(angle*(math.pi/180)))
        # gen_pt3 = (gen_pt1[0] + gen_point_width * math.sin(angle*(math.pi/180)), gen_pt1[1] * math.cos(angle*(math.pi/180)))
        # gen_pt2 = (pt4[0] + height_distance * math.sin(angle*(math.pi/180)), pt4[1] + height_distance * math.cos(angle*(math.pi/180)))
        # gen_pt4 = (gen_pt3[0] + gen_point_width * math.sin(angle*(math.pi/180)), gen_pt3[1] * math.cos(angle*(math.pi/180)))
    else:
        width_distance = heightRect / 3
        gen_pt1 = (pt2[0] + width_distance * math.cos(angle*(math.pi/180)), pt2[1] - width_distance * math.sin(angle*(math.pi/180)))
        gen_pt2 = (gen_pt1[0] + gen_point_width * math.cos(angle*(math.pi/180)), gen_pt1[1] - gen_pt1[1] * math.sin(angle*(math.pi/180)))
        gen_pt4 = (pt3[0] + width_distance * math.cos(angle*(math.pi/180)), pt3[1] - width_distance * math.sin(angle*(math.pi/180)))
        gen_pt3 = (gen_pt4[0] + gen_point_width * math.cos(angle*(math.pi/180)), gen_pt4[1] - gen_pt4[1] * math.sin(angle*(math.pi/180)))
    gen_point = [gen_pt1[0], gen_pt1[1], gen_pt2[0], gen_pt2[1], gen_pt3[0], gen_pt3[1], gen_pt4[0], gen_pt4[1]]
    gen_point = list( map( int, gen_point ))
#     print("-------------------the original point is:|{}|".format( base_point ))
#     print("-------------------the gen point is:|{}|".format(gen_point))
    gen_point_format2 = [gen_pt1[0], gen_pt1[1], gen_pt2[0], gen_pt2[1], gen_pt4[0], gen_pt4[1], gen_pt3[0], gen_pt3[1]]
    gen_point_format2 = list( map(int, gen_point_format2 ))
    gen_point_format2 = list(map(abs, gen_point_format2 ))
    psenet_points.append( gen_point_format2 )
    return psenet_points

def label_change( ocr_result ):
    ori_ocr_results = ocr_result
    new_ocr_results = []
    new_key_list = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ori_key_list = list('abcdefghijklmnopqrstuvwxyz零一二三四五六七八九')
    try:
        for ori_ocr_result in ori_ocr_results:
            print("================ori_ocr_result:|{}|==================".format(ori_ocr_result))
            new_ocr_result = ''
            for alpha in ori_ocr_results:
                index = ori_key_list.index( alpha )
                new_alpha = new_key_list[ index ]
                new_ocr_result += new_alpha
            new_ocr_results.append( new_ocr_result )
    except:
        pass
    return new_ocr_results

def container_type( text_recs ):
    if not len(text_recs) :
        return 
    xs = []
    ys = []
    for point in text_recs:
        point_x = point[::2]
        point_y = point[1::2]
        xs.extend(point_x)
        ys.extend(point_y)
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    width = xmax - xmin
    height = ymax - ymin
    wh_rate = width / height
    if wh_rate > 0.8 and wh_rate < 4.5 :
        # 宽大于高
        return 0
    elif wh_rate > 4.5:
        return 1
    elif wh_rate < 0.3:
        return 2
    else: #在0.8到0.3之间
        return 3

def demo(image, image_name='initial_name.jpg'):

    psenet_start_time = time.time()
    psenet_text_recs, psenet_draw_image = psenet_predict( image )
    psenet_end_time = time.time()
    print("psenet耗时：|{}|".format( psenet_end_time-psenet_start_time ))
    psenet_image_name = image_name[:-4] + '_psenet' + ".jpg"
    print("psenet_image_name:|{}|,psenet_result_text_recs:|{}|".format(psenet_image_name, psenet_text_recs))
    objectid = container_type( text_recs=psenet_text_recs )
    psenet_result_path = os.path.join(psenet_result_dir, psenet_image_name)
    #@@@@@@
    Image.fromarray(psenet_draw_image).save(psenet_result_path)  #处理十万图片不输出
    # ==================
    # deeptext 文字识别 #deeptext的image传入ndarray类型，但在识别的时候需要PIL类型
    # ===================
    adjust_result_dir = os.path.join(adjust_main_dir, psenet_image_name[:-4])
    makedir(adjust_result_dir)
    ocr_result = []
    ocr_result, cfd_scores = ocr.model(image=image,
                           text_recs=psenet_text_recs,
                           adjust_result_dir=adjust_result_dir,
                           original_image_name=psenet_image_name, )
    print("psenet_image_name:|{}|,ocr_result:|{}|".format(psenet_image_name, ocr_result))
    hscore_text_recs = []
    hscore_ocr_result = []
    # print("====cfd_score:|{}|".format( cfd_scores ))
    for index, confidence_score in enumerate( cfd_scores ):
        # print( "=======confidence:|{}|".format( confidence_score))
        if confidence_score >= 0.1:
            hscore_text_recs.append( psenet_text_recs[index] )
            hscore_ocr_result.append( ocr_result[index] )
        else:
            print("confidence_score:|{}| < 0.1 mv".format( confidence_score ))
    ocr_result_file.writelines(psenet_image_name + ' ' + str(ocr_result) + ',' + '\n')
    psenet_point_file.writelines( psenet_image_name + ':' + str(psenet_text_recs) + '\n')
    return hscore_text_recs, hscore_ocr_result, objectid

def pick_container_num( psenet_text_recs, ocr_result, objectid ):
    container_num = {'masterid': [], "boxid": [], "checknum": [], "boxtype": []}
    print("objectid:|{}|".format(objectid))
    if (not ocr_result) or len(ocr_result)==1:
        pass
    if objectid == 0:
        container_num = get_container_num.boxid0(text_recses=psenet_text_recs, ocr_results=ocr_result)
    elif objectid == 1:
        container_num = get_container_num.boxid1(text_recses=psenet_text_recs, ocr_results=ocr_result)
    elif objectid == 2:
        container_num = get_container_num.boxid2(text_recses=psenet_text_recs, ocr_results=ocr_result)
    elif objectid == 3:
        container_num = get_container_num.boxid3(text_recses=psenet_text_recs, ocr_results=ocr_result)
    else:
        warning = "-warning----!----!----!---没有object"
        print(warning)
        error_file.writelines( warning + '\n' )
    return container_num

def flask_demo( image ):
    psenet_text_recs, ocr_result, objectid = demo( image )
    container_num = pick_container_num(psenet_text_recs=psenet_text_recs, ocr_result=ocr_result, objectid=objectid)
    best_container_num, best_boxtype, correct_flat = get_container_num.choice_container_num(container_num_dict=container_num)
    result_dict = {'container_num':best_container_num, 'boxtype': best_boxtype, 'correct_flat': correct_flat } 
    print("flask result dict:|{}|".format( result_dict ))
    result_json = json.dumps( result_dict, ensure_ascii=False)
    return result_json

if __name__ == '__main__':
    test_image_dir = "datas/valid_image"
    image_names = os.listdir( test_image_dir )
    image_names.sort(key=lambda x:int(x.split('_')[0]))
    # image_names = image_names[20000:21000]
    all_ocr_char_results = []
    for image_name in sorted(image_names):
        print('<=================正在处理新图片================================================>|', image_name, '||')
        image_path = os.path.join( test_image_dir ,image_name )
        try:
            print("image_path:{}".format(image_path))
            PIL_image = Image.open(image_path).convert('RGB')
            image = cv2.imread( image_path )[:, :, ::-1]
        except:
            error = '图片 |{}| 读取错误'.format( image_name )
            print( error )
            error_file.writelines( error + '\n' )
            continue
        # print("》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》 |   yolov3目标检测     || ")
        demo_result_dict = {'image_name':'','psenet_text_recs':"", "ocr_result":"", "objectid":""}
        demo_result_dict['image_name'] = image_name
        psenet_text_recs, ocr_result, objectid = demo( image, image_name=image_name )
        np_psenet_text_recs = np.array( psenet_text_recs )
        demo_result_dict['psenet_text_recs'] = np_psenet_text_recs.tolist()
        demo_result_dict['ocr_result'] = ocr_result
        demo_result_dict['objectid'] = objectid
        result_json = json.dumps( demo_result_dict, ensure_ascii=False)
        json_save_path = os.path.join( result2json_dir, image_name.split('.')[0] + '.json' )
        #@@@@@@
        # with open( json_save_path, 'w') as json_file:
        #     json_file.write(result_json)
        if psenet_text_recs == [] or ocr_result == []:
            error = '-error-图片|{}|检测的psenet_text_recs:|{}| 和ocr_result:|{}|为空'.format( image_name, psenet_text_recs, ocr_result )
            print( error )
            error_file.writelines( error + '\n' )
            continue
        container_num = pick_container_num(psenet_text_recs=psenet_text_recs, ocr_result=ocr_result, objectid=objectid)
        best_container_num, best_boxtype, correct_flat = get_container_num.choice_container_num(container_num_dict=container_num)
        #@@@@@@
        # if correct_flat == 'F':
        #     error_det_result_file.writelines(image_name + ' ' + best_container_num + correct_flat + 'type:|{}|'.format(objectid) + '\n')
        #     error_image_save_path = os.path.join( error_detect_image_dir, image_name )
        #     PIL_image.save( error_image_save_path )
        container_num_file.writelines(image_name + ' ' + best_container_num + correct_flat + 'type:|{}|'.format(objectid) + '\n')
        print( "best_container_num:|{}|".format(container_num))
        if objectid is not None:
            ocr2labelmejson(texts=ocr_result, points=np_psenet_text_recs.tolist(), imagepath=image_path, correct_flag=correct_flat)
        else:
            print('------------objectid is none')
        #yolov3的image传入ndarray类型cv读
    error_file.close()
    container_num_file.close()
    ocr_result_file.close()
    statistics()
