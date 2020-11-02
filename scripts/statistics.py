#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2019/12/2820:48
#文件名称：statictis.py


boxid0_container_num_path = 'datas/boxid0_container_num.txt'
boxid1_container_num_path = 'datas/boxid1_container_num.txt'





def statistics():
    container_result_path = 'datas/container_num.txt'
    with open( container_result_path, 'r', encoding='UTF-8') as container_result_file:
        container_nums = container_result_file.readlines()
    len_container_nums = len(container_nums)
    correct_count = 0
    boxid0_correct_count = 0
    boxid0_num_count = 0
    boxid0_lensix = 0
    boxid0_container_num_file = open(boxid0_container_num_path, 'w', encoding='UTF-8')
    boxid1_container_num_file = open(boxid1_container_num_path, 'w', encoding='UTF-8')
    for ori_container_num in container_nums:
        container_num = ori_container_num.replace("|", "").rstrip('\n')
        container_num = container_num.replace("type:", " ")
        container_num_list = container_num.split(" ")
        try:
            image_name, container_T, objectid = container_num_list
            print('=========================process image:| {} |====================='.format( image_name ))

            correct_flat = container_T[-1]

            if correct_flat == 'T':
                correct_count += 1
                if objectid == '0':
                    print('-----')
                    boxid0_correct_count += 1
            if objectid == '0':
                boxid0_num_count += 1
                boxid0_container_num_file.write(ori_container_num)
                print(len(container_T))
                if len(container_T) == 10:
                    boxid0_lensix += 1
            else:
                boxid1_container_num_file.write(ori_container_num)

        except:
            pass
    statict = '图片总数：|{}|，\n验证正确数量：|{}|，\n识别正确里面标准号正确数量：|{}|, \n' \
              '总的标准号数量：|{}|,\n, 标准号长度为6的数量：|{}|' \
              ''.format( len_container_nums, correct_count, boxid0_correct_count, boxid0_num_count, boxid0_lensix)
    print(statict)
    boxid0_container_num_file.close()
    boxid1_container_num_file.close()




def statistics_50051():
    container_result_path = 'datas/container_num.txt'
    with open( container_result_path, 'r') as container_result_file:
        container_nums = container_result_file.readlines()
    len_container_nums = len(container_nums)
    correct_count = 0
    boxid0_correct_count = 0
    boxid0_num_count = 0
    objectid = 0
    print("================statistics=======================")
    for container_num in container_nums:
        container_num_list = container_num.rstrip('\n').split(" ")
        try:
            image_name, container_T = container_num_list
            # print('=========================process image:| {} |====================='.format( image_name ))

            correct_flat = container_T[-1]
            if correct_flat == 'T':
                correct_count += 1
                if objectid == '0':
                    print('-----')
                    boxid0_correct_count += 1
            if objectid == '0':
                boxid0_num_count += 1

        except:
            pass
    statict = '图片总数：|{}|，\n验证正确数量：|{}|，\n识别正确里面标准号正确数量：|{}|, \n' \
              '总的标准号数量：|{}|'.format( len_container_nums, correct_count, boxid0_correct_count, boxid0_num_count)
    print(statict)


if __name__ == '__main__':
    
    statistics()
