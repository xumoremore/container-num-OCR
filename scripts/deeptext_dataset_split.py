#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2020/3/99:55
#文件名称：deeptext_dataset_split.py

import os
import random
import numpy as np
from PIL import Image
import shutil

def makedir( dir ):
    if os.path.exists( dir ):
        shutil.rmtree( dir )
    os.mkdir( dir )

# main_image_dir = '/data02/shan_dong/xzc/self_bill/bill_images200000'
# main_txt_path = '/data02/shan_dong/xzc/self_bill/bill_images200000/tmp_labels.txt'


main_image_dir = '/data02/shan_dong/xzc/sever_connect/yolov3_psenet_ocr_train/deeptext/data/all_container_deeptext_td/image'
main_txt_path = '/data02/shan_dong/xzc/sever_connect/yolov3_psenet_ocr_train/deeptext/data/all_container_deeptext_td/label.txt'

result_dir = 'deeptext/data/train_split_label_v1/'

train_image_dir = result_dir + 'train_images'
train_label_path = result_dir + 'train_labels.txt'

read_image_dir = result_dir + 'read_images/'
read_label_path = read_image_dir + 'read_labels.txt'

valid_image_dir = result_dir + 'valid_images'
valid_label_path = result_dir + 'valid_labels.txt'
# makedir( train_image_dir)
makedir( read_image_dir )
# makedir( valid_image_dir )


train_rate = 0.75
read_rate = 0.0001
valid_rate = 0.2459


image_names = os.listdir( main_image_dir )
np.random.shuffle( image_names )

with open( main_txt_path, 'r', encoding='utf-8' ) as main_txt_file:
    labels = main_txt_file.readlines()
    

np.random.shuffle( labels )

labellen = len( labels )
train_len = int( train_rate*labellen )
read_len = int( read_rate*labellen )
valid_len = int( valid_rate*labellen )

print("总的图片数量：|{}|".format( labellen ))


train_label_file = open( train_label_path, 'w', encoding='utf-8' )
for label in labels[0:train_len]:
    # image_name = label.split(' ')[0]
    # image_path = os.path.join( main_image_dir, image_name )
    # train_image_path = os.path.join( train_image_dir, image_name )
    # image = Image.open( image_path )
    # image.save( train_image_path )

    # label = label.replace(' ', '.jpg ')

    train_label_file.write( label )
train_label_file.close()
print("训练集数量：|{}|".format( train_len ))



valid_label_file = open(valid_label_path, 'w', encoding='utf-8')
for label in labels[train_len+read_len:]:
    # image_name = label.split(' ')[0]
    # image_path = os.path.join(main_image_dir, image_name)
    # valid_image_path = os.path.join(valid_image_dir, image_name)

    # image = Image.open(image_path)
    # image.save(valid_image_path)

    # label = label.replace(' ', '.jpg ')
    valid_label_file.write( label )
valid_label_file.close()
print("验证集数量：|{}|".format( valid_len ))




read_label_file = open(read_label_path, 'w', encoding='utf-8')
for label in labels[train_len:train_len+read_len]:
    image_name = label.split(' ')[0]
    image_path = os.path.join(main_image_dir, image_name)
    read_image_path = os.path.join(read_image_dir, image_name)
    image = Image.open(image_path)
    image.save(read_image_path)

    # label = label.replace(' ', '.jpg ')
    read_label_file.write( label )
read_label_file.close()
print("查看集数量:|{}|".format( read_len ))






















########################################################

"""
#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2020/3/99:55
#文件名称：deeptext_dataset_split.py

import os
import random
import numpy as np
from PIL import Image
import shutil

def makedir( dir ):
    if os.path.exists( dir ):
        shutil.rmtree( dir )
    os.mkdir( dir )

main_image_dir = '/data02/shan_dong/xzc/self_bill/bill_images200000'
main_txt_path = '/data02/shan_dong/xzc/self_bill/bill_images200000/tmp_labels.txt'


# main_image_dir = '/data02/shan_dong/xzc/TextRecognitionDataGenerator-master/trdg/datas/out/images'
# main_image_dir = '/data02/shan_dong/xzc/TextRecognitionDataGenerator-master/trdg/datas/out/labels.txt'

result_dir = 'deeptext/data/ori_datasets/'

train_image_dir = result_dir + 'train_images'
train_label_path = result_dir + 'train_labels.txt'

test_image_dir = result_dir + 'test_images'
test_label_path = result_dir + 'test_labels.txt'

valid_image_dir = result_dir + 'valid_images'
valid_label_path = result_dir + 'valid_labels.txt'
# makedir( train_image_dir)
# makedir( test_image_dir )
# makedir( valid_image_dir )


train_rate = 0.7
test_rate = 0.299
valid_rate = 0.001


image_names = os.listdir( main_image_dir )
np.random.shuffle( image_names )

with open( main_txt_path, 'r', encoding='utf-8' ) as main_txt_file:
    labels = main_txt_file.readlines()
np.random.shuffle( labels )

labellen = len( labels )
train_len = int( train_rate*labellen )
test_len = int( test_rate*labellen )
valid_len = int( valid_rate*labellen )

print("总的图片数量：|{}|".format( labellen ))


train_label_file = open( train_label_path, 'w', encoding='utf-8' )
for label in labels[0:train_len]:
    image_name = label.split(' ')[0] + '.jpg'
    image_path = os.path.join( main_image_dir, image_name )
    train_image_path = os.path.join( train_image_dir, image_name )
    image = Image.open( image_path )
    image.save( train_image_path )
    deeptext_label = label.replace(' ', '.jpg ')
    train_label_file.write( deeptext_label )
train_label_file.close()
print("训练集数量：|{}|".format( train_len ))





test_label_file = open(test_label_path, 'w', encoding='utf-8')
for label in labels[train_len:train_len+test_len]:
    image_name = label.split(' ')[0] + '.jpg'
    image_path = os.path.join(main_image_dir, image_name)
    test_image_path = os.path.join(test_image_dir, image_name)
    image = Image.open(image_path)
    image.save(test_image_path)
    deeptext_label = label.replace(' ', '.jpg ')
    test_label_file.write( deeptext_label )
test_label_file.close()
print("测试集数量:|{}|".format( test_len ))




valid_label_file = open(valid_label_path, 'w', encoding='utf-8')
for label in labels[train_len+test_len:]:
    image_name = label.split(' ')[0] + '.jpg'
    image_path = os.path.join(main_image_dir, image_name)
    valid_image_path = os.path.join(valid_image_dir, image_name)

    image = Image.open(image_path)
    image.save(valid_image_path)
    deeptext_label = label.replace(' ', '.jpg ')
    valid_label_file.write( deeptext_label )
valid_label_file.close()
print("验证机数量：|{}|".format( valid_len ))
"""