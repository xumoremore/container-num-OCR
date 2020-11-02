#_*_coding utf-8 _*_
#开发者：xzc
#开发时间：2019/12/2022:39
#文件名称：get_boxid_and_type.py

"""
isalpha:可用于判断字符串是否存在英文字母
isalnum:可用于判断字符串是否都是字母数字
seriel_str.encode('utf-8').isalpha()
"""
import copy
import re


error_station_path = './datas/error_station.txt'
error_file = open( error_station_path, 'a', encoding='UTF-8')

def get_center_xy( text_recses):
    center_xys = []
    for point in text_recses:
        x_point = point[::2]
        y_point = point[1:8:2]
        x_float = [float(i) for i in x_point]
        y_float = [float(i) for i in y_point]
        xmin = int(min(x_float))
        ymin = int(min(y_float))
        xmax = int(max(x_float))
        ymax = int(max(y_float))
        center_x = (xmin+xmax)//2
        center_y = (ymin+ymax)//2
        center_xys.append( [center_x, center_y])
    return center_xys

# def correct_boxtype( possible_boxtype ):
#     boxtype = possible_boxtype
#     error_alpha_dict = {'6':'G', '0':'U', }
#     error_endnum_dict = {'I':'1','i':'1'}
#     new_boxtype = possible_boxtype
#     if len(possible_boxtype) == 4:
#         for key,value in error_alpha_dict.items():
#             if boxtype[2] == key:
#                 new_boxtype = boxtype[0:2] + value + boxtype[3]
#         for key,value in error_endnum_dict.items():
#             if new_boxtype[3] == key:
#                 new_boxtype = new_boxtype[0:3] + value
#         return new_boxtype
#     elif len(possible_boxtype) != 4:
#         return None

"""
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
0 1 2 3 4 5 6 7 8 9 
"""
def correct_checknum( possible_checknum ):
    #不用确定长度
    check_nums = []
    if len(possible_checknum) < 1:
        return check_nums
    check_num = possible_checknum
    error_num_dict = {"A": "4", "B": "8", "C": "0", "D": "0", "E": "5", "F": "5", "G": "6", "H": "5", "I": "1",
                      "J": "wait", "K": "wait", "L": "1", "M": "wait", "N": "wait","O":"0", "P": "9", "Q": "0", "R": "0",
                      "S": "5", "T": "1", "U": "0", "V": "0", "W": "wait", "X": "wait", "Y": "wait", "Z": "2"}
    for singlenum in check_num:
        if singlenum in error_num_dict.keys():
            check_nums.append( error_num_dict[singlenum] )
        else:
            check_nums.append( singlenum )
    return check_nums

def correct_boxtype( possible_boxtype ):
    boxtypes = []
    boxtype = possible_boxtype
    error_alpha_dict = {'6':'G', '0':'U', '8':'B'}
    error_endnum_dict = {'I':'1','i':'1'}
    len_boxtype = len(boxtype)
    if len_boxtype == 4:
        for key,value in error_alpha_dict.items():
            if boxtype[2] == key:
                boxtype = boxtype[0:2] + value + boxtype[3]
        for key,value in error_endnum_dict.items():
            if boxtype[3] == key:
                boxtype = boxtype[0:3] + value
        boxtypes.append(boxtype)
    elif len_boxtype == 3:
        if boxtype=='22G' or boxtype=='226':
            boxtype = '22G1'
        boxtypes.append( boxtype )
    return boxtypes

def correct_boxid( possible_boxid ):
    boxids = []
    if len(possible_boxid) != 6:
        return boxids
    boxid = possible_boxid
    error_num_dict = {"A": "4", "B": "8", "C": "0", "D": "0", "E": "5", "F": "5", "G": "6", "H": "5", "I": "1",
                      "J": "待", "K": "待", "L": "1", "M": "待", "N": "待","O":"0", "P": "待", "Q": "0", "R": "0",
                      "S": "5", "T": "1", "U": "0", "V": "0", "W": "待", "X": "待", "Y": "待", "Z": "2"}
    for singleid in boxid:
        if singleid in error_num_dict.keys():
            print("correct boxid num: |{}| to |{}| ".format( singleid, error_num_dict[singleid] ))
            boxid = boxid.replace(singleid,error_num_dict[singleid])
            print("矫正之后的correct boxid为：|{}|".format(boxid))
    boxids.append( boxid )
    return boxids

def correct_masterid( possible_masterid ):
    masterids = []
    if len(possible_masterid) != 4:
        return masterids
    masterid = possible_masterid
    error_alpha_dict = {"0":"O", "1":"I", "2":"Z", "3":"B", "4":"A", "5":"待", "6":"G", "7":"Z", "8":"B", "9":"待"}
    #UZJ
    error_endalpha_dict = {"L":"U", "O":"U", "I":"U", "V":"U", "Y":"U", "0":"U", }
    for singleid in masterid:
        if singleid in error_alpha_dict.keys():
            print("correct masterid n_um: |{}| to |{}| ".format( singleid, error_alpha_dict[singleid] ))
            masterid = masterid.replace( singleid, error_alpha_dict[singleid] )
            print("矫正之后的masterid为:|{}|".format(masterid))

    if masterid[3] in error_endalpha_dict.keys():
        masterid = masterid[0:-1] + error_endalpha_dict[masterid[3]]
    masterids.append(masterid)
    if masterid[2] == 'D':
        masterid2 = masterid.replace( masterid[2], 'O')
        masterids.append( masterid2 )
    if masterid[2] == 'O':
        masterid3 = masterid.replace( masterid[2], 'D')
        masterids.append( masterid3 )
    return masterids

def master_box_check_split( container_num_str ):
    # print('------len container_num_str:|{}|'.format( len(container_num_str)))
    len_container_num = len( container_num_str )
    masterid = ""
    boxid = ""
    checknum = ""
    if len_container_num == 11:
        masterid = container_num_str[0:4]
        boxid = container_num_str[4:10]
        checknum = container_num_str[-1]
    elif len_container_num == 10:
        masterid = container_num_str[0:4]
        boxid = container_num_str[4:10]
    elif len_container_num > 11:
        # print("cotainer_num_str > 11")
        # print("container_num_str:|{}|".format(container_num_str))
        masterid = re.findall(r"[a-zA-Z]{4}", container_num_str)
        boxid = re.findall(r"\d{6}", container_num_str)
        checknum = container_num_str[-1]
        # print(len(masterid))
        if len(masterid) == 1:
            masterid = masterid[0]
        if len(boxid) == 1:
            boxid = boxid[0]
        # print("masterid:|{}|, boxid:|{}|, checknum:|{}|".format(masterid, boxid, checknum))
    elif len_container_num < 10 and len_container_num > 1:
        # print("container_num_str:|{}|".format( container_num_str ))
        masterid = re.findall(r"[a-zA-Z]{4}", container_num_str)
        boxid = re.findall(r"\d{6}", container_num_str)
        if len(masterid) == 1:
            masterid = masterid[0]
        if len(boxid) == 1:
            boxid = boxid[0]
        checknum = container_num_str[-1]
        # print(masterid, boxid, check_num)
    return masterid, boxid, checknum

"""
str.isalnum()  所有字符都是数字或者字母，为真返回 Ture，否则返回 False。
str.isalpha()   所有字符都是字母(当字符串为中文时, 也返回True)，为真返回 Ture，否则返回 False。
str.isdigit()     所有字符都是数字，为真返回 Ture，否则返回 False。
str.islower()    所有字符都是小写，为真返回 Ture，否则返回 False。
str.isupper()   所有字符都是大写，为真返回 Ture，否则返回 False。
str.istitle()      所有单词都是首字母大写，为真返回 Ture，否则返回 False。
str.isspace()   所有字符都是空白字符，为真返回 Ture，否则返回 False。
"""

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def ocr_result_clean( text_recses, ocr_results ):
    mv_indexs = []
    new_ocr_results = []
    new_text_recses = list(text_recses)
    for index, ocr_result in enumerate(ocr_results):
        if is_contain_chinese(ocr_result):
            new_ocr_result = ''
        else:
            new_ocr_result = "".join(filter(str.isalnum, ocr_result))
        if new_ocr_result == '':
            mv_indexs.append( index )
        new_ocr_results.append(new_ocr_result)
        # print("ocr_result:|{}|, new_ocr_result:|{}|".format(ocr_result, new_ocr_result))
        #         if is_contain_chinese( ocr_result ):
        #             mv_index.append( index )
    for index, new_ocr_result in enumerate( new_ocr_results ):
        if index in mv_indexs:
            new_ocr_results.pop( index )
            new_text_recses.pop( index )
    return new_text_recses, new_ocr_results

def container_num_check( masterid, boxid, checknum ):
    """
    :param masterid: 要保证是四位大写字母
    :param boxid:  要保证是六位数字
    :param checknum:  要保证是一位数字
    :return: 
    """
    try:
        if (not masterid.encode('utf-8').isupper()) and len(masterid) != 4 :
            return False
        if not boxid.isdigit() and len(boxid) != 6 :
            return False
        if not checknum.isdigit() and len(checknum) != 1:
            return False
        the_sum = 0
        equal_value_table = {'A':10, 'B':12, "C":13, "D":14, "E":15, "F":16, "G":17,
                             "H":18, "I":19, "J":20, "K":21, "L":23, "M":24, "N":25,
                             "O":26, "P":27, "Q":28, "R":29, "S":30, "T":31, "U":32,
                             "V":34, "W":35, "X":36, "Y":37, "Z":38, "0":0, "1":1,
                             "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9}
        container_num_str = masterid + boxid
        for index, container_char in enumerate(container_num_str):
            the_sum += equal_value_table[container_char] * ( 2**index )
        remainder = the_sum%11
        #10=0
        if remainder == int(checknum) or remainder == int(checknum) + 10:
            return True
        else:
            return False
    except:
        return False

def boxid0( text_recses, ocr_results):
    # container
    # number
    text_recses, ocr_results = ocr_result_clean(text_recses=text_recses, ocr_results=ocr_results)
    container_num_dict = {'masterid':[], "boxid":[], "checknum":[],"boxtype":[]}
    mv_index = []
    possible_boxtype = ""
    possible_masterid = ""
    possible_boxid = ""
    possible_checknum = ""
    center_xys = get_center_xy(text_recses)
    #确定boxtype
    if len(text_recses) > 3:
        max_y = max(center_xys, key=lambda x: x[1])
        max_y_index = center_xys.index( max_y )
        possible_boxtype = ocr_results[ max_y_index]
        mv_index.append(max_y_index)
    else:
        for index, ocr_result in enumerate(ocr_results):
            if index == center_xys.index( max(center_xys, key=lambda x: x[1]) ):
                if len(ocr_result) == 4:
                    if ocr_result[0].isdigit and ocr_result[-2].isalpha():
                        possible_boxtype = ocr_result
                        mv_index.append(index)
                    else:
                        pass
                elif len(ocr_result) == 3:
                    if ocr_result[0].isdigit() and ocr_result[-1].isalpha():
                        possible_boxtype = ocr_result
                        mv_index.append( index )
                    else:
                        pass
                elif len( ocr_result) == 5:
                    if ocr_result[2].isdigit():
                        possible_boxtype = ocr_result
                        mv_index.append( index )
                    else:
                        pass
#                 if len(ocr_result) == 4 and ocr_result[-1].isdigit() and ocr_result[-2].isalpha():
#                     possible_boxtype = ocr_result
#                     mv_index.append( ocr_results.index( ocr_result ) )
#                     break;

    #确定masterid
    for index,ocr_result in enumerate(ocr_results):
        len_ocr_result = len(ocr_result)
        isalpha_flat = ocr_result.encode('utf-8').isalpha()
        if isalpha_flat and len_ocr_result==4:
            possible_masterid = ocr_result
            mv_index.append( index )
            break
        elif len_ocr_result==4:
            alpha_num = 0
            for i in ocr_result:
                if i.isalpha():
                    alpha_num +=1
            if alpha_num >= 3:
                possible_masterid = ocr_result
                mv_index.append( index )
        elif len_ocr_result >= 3 and ocr_result.isalpha():
            mv_index.append( index )
    boxid_checknum_x_ocr_list = []
    for index, center_xy in enumerate(center_xys):
        if index not in mv_index:
            center_x = center_xy[0]
            ocr_result = ocr_results[index]
            # if len(ocr_result) == 1 and ocr_result.isdigit():
            #     possible_check_num2 = ocr_result
            boxid_checknum_x_ocr_list.append( (center_x, ocr_result))
    sorted_boxid_checknum_x_ocr_list = sorted( boxid_checknum_x_ocr_list,key=lambda x: x[0] )
    sorted_boxid_checknum_ocr_list = [ i[1] for i in sorted_boxid_checknum_x_ocr_list]
    # possible_checknum3 = sorted_boxid_checknum_ocr_list[-1]
    if len(sorted_boxid_checknum_ocr_list) == 2:
        if len(sorted_boxid_checknum_ocr_list[0]) == 6:
            possible_boxid = sorted_boxid_checknum_ocr_list[0]
        if len(sorted_boxid_checknum_ocr_list[1]) == 1 or len(sorted_boxid_checknum_ocr_list[1]) == 2:
            possible_checknum = sorted_boxid_checknum_ocr_list[1]
    elif len(sorted_boxid_checknum_ocr_list) == 1:
        boxid_checknum_str = sorted_boxid_checknum_ocr_list[0]
        len_id_num = len(boxid_checknum_str)
        if len_id_num == 6:  #没有校验码
            possible_boxid = boxid_checknum_str[0:6]
        elif len_id_num == 7 or len_id_num == 8:
            possible_boxid = boxid_checknum_str[0:6]
            possible_checknum = boxid_checknum_str[-1]
        elif len_id_num == 1:
            possible_checknum = boxid_checknum_str[-1]
    elif sorted_boxid_checknum_ocr_list != []: #有三个
        boxid_checknum_str = ''.join(sorted_boxid_checknum_ocr_list)
        len_id_num = len(boxid_checknum_str)
        if len_id_num == 6:
            possible_boxid = boxid_checknum_str
        if len_id_num == 7 or len_id_num == 8:
            possible_boxid = boxid_checknum_str[0:6]
            possible_checknum = boxid_checknum_str[-1]
    container_num_dict['boxtype'] = correct_boxtype(possible_boxtype=possible_boxtype)
    container_num_dict['boxid'] = correct_boxid( possible_boxid = possible_boxid )
    container_num_dict['masterid'] = correct_masterid(possible_masterid=possible_masterid)
    container_num_dict['checknum'] = correct_checknum(possible_checknum=possible_checknum)
    return container_num_dict


def boxid1( text_recses, ocr_results):
    text_recses, ocr_results = ocr_result_clean(text_recses=text_recses, ocr_results=ocr_results)
    container_num_dict = {'masterid': [], "boxid": [], "checknum": [], "boxtype": []}
    center_xys = get_center_xy( text_recses )
    center_xs = [ i[0] for i in center_xys]
    x_ocr_list = zip(center_xs, ocr_results )
    sorted_x_ocr_list = sorted( x_ocr_list, key=lambda x:x[0] )
    sorted_ocr_list = [ i[1] for i in sorted_x_ocr_list ]
    container_num_str = ''.join( sorted_ocr_list )
    masterid,boxid,checknum = master_box_check_split( container_num_str=container_num_str)
    container_num_dict['boxid'] = correct_boxid( possible_boxid = boxid )
    container_num_dict['masterid'] = correct_masterid(possible_masterid=masterid)
    container_num_dict['checknum'] = correct_checknum(possible_checknum=checknum)
    return container_num_dict

def boxid2( text_recses, ocr_results):
    text_recses, ocr_results = ocr_result_clean(text_recses=text_recses, ocr_results=ocr_results)
    container_num_dict = {'masterid': [], "boxid": [], "checknum": [], "boxtype": []}
    center_xys = get_center_xy( text_recses )
    center_ys = [ i[1] for i in center_xys]
    y_ocr_list = zip(center_ys, ocr_results )
    sorted_y_ocr_list = sorted( y_ocr_list, key=lambda x:x[0] )
    sorted_ocr_list = [ i[1] for i in sorted_y_ocr_list ]
    container_num_str = ''.join( sorted_ocr_list )
    masterid,boxid,checknum = master_box_check_split( container_num_str=container_num_str)
    container_num_dict['boxid'] = correct_boxid( possible_boxid = boxid )
    container_num_dict['masterid'] = correct_masterid(possible_masterid=masterid)
    container_num_dict['checknum'] = correct_checknum(possible_checknum=checknum)
    return container_num_dict


def boxid3( text_recses, ocr_results):
    text_recses, ocr_results = ocr_result_clean(text_recses=text_recses, ocr_results=ocr_results)
    container_num_dict = {'masterid': [], "boxid": [], "checknum": [], "boxtype": []}
    boxtype_row_y_ocr = []
    id_row_y_ocr = []
    center_xys = get_center_xy(text_recses)
    center_xs = [ i[0] for i in center_xys]
    aver_x = sum(center_xs) / len(center_xs)
    for index, center_xy in enumerate(center_xys):
        center_x = center_xy[0]
        if center_x < aver_x:
            id_row_y_ocr.append( (center_xy[1], ocr_results[index]))
        else :
            boxtype_row_y_ocr.append( (center_xy[1], ocr_results[index]))
    sorted_id_row_y_ocr = sorted( id_row_y_ocr, key=lambda x:x[0])
    sorted_boxtype_row_y_ocr = sorted( boxtype_row_y_ocr, key=lambda x:x[0])
    sorted_id_row_ocr = [i[1] for i in sorted_id_row_y_ocr]
    sorted_boxtype_row_ocr = [i[1] for i in sorted_boxtype_row_y_ocr]
    container_num_str = "".join(sorted_id_row_ocr)
    possible_boxtype = "".join(sorted_boxtype_row_ocr)
    masterid,boxid,checknum = master_box_check_split( container_num_str=container_num_str)
    container_num_dict['boxtype'] = correct_boxtype(possible_boxtype=possible_boxtype)
    container_num_dict['boxid'] = correct_boxid( possible_boxid = boxid )
    container_num_dict['masterid'] = correct_masterid(possible_masterid=masterid)
    container_num_dict['checknum'] = correct_checknum(possible_checknum=checknum)
    return container_num_dict

def choice_container_num( container_num_dict ):
    correct_flat = 'F'
    masterids = container_num_dict['masterid']
    boxids = container_num_dict['boxid']
    checknums = container_num_dict['checknum']
    boxtypes = container_num_dict['boxtype']
    for masterid in masterids:
        for boxid in boxids:
            for checknum in checknums:
                print("masterid:|{}|, boxid:|{}|, checknum:|{}|".format(masterid, boxid, checknum))
                if container_num_check(masterid, boxid, checknum):
                    best_container_num = masterid + boxid + checknum
                    print("====================================")
                    print("|| 集装箱号码校验正确:|{}| ||".format( masterid+boxid+checknum ))
                    print("====================================")
                    correct_flat = 'T'
                    break;
    if correct_flat == 'F':
        try:
            best_container_num = masterid + boxid + checknum
        except:
            try:
                best_container_num = masterid + boxid 
            except:
                best_container_num = ''
    if boxtypes != []:
        best_boxtype = boxtypes[0]
    else:
        best_boxtype = ''
    return best_container_num, best_boxtype, correct_flat


if __name__ == '__main__':
    boxid0_text_recses = [[201,49,201,18,216,49,216,18],
                 [15,52,15,22,66,52,66,22],
                 [102,49,102,22,180,49,180,22],
                 [98,87,98,59,148,87,148,59],]
    boxid0_ocr_results = ["0","DRYU","262004","22G1"]
    boxid0_container_num = boxid0( text_recses= boxid0_text_recses, ocr_results=boxid0_ocr_results )
    check_flat = container_num_check( masterid = boxid0_container_num['masterid'],
                                      boxid = boxid0_container_num['boxid'],
                                      checknum = boxid0_container_num['checknum'])
    print(boxid0_container_num)
    print(check_flat)
    boxid1_text_recses = [[201,49,201,18,216,49,216,18],
                 [15,52,15,22,66,52,66,22],
                 [102,49,102,22,180,49,180,22],]
    boxid1_ocr_results = ["0","DRYU","22004"]
    boxid1_container_num = boxid1( text_recses= boxid1_text_recses, ocr_results=boxid1_ocr_results )
    check_flat = container_num_check( masterid = boxid1_container_num['masterid'],
                                      boxid = boxid1_container_num['boxid'],
                                      checknum = boxid1_container_num['checknum'])
    print(boxid1_container_num)
    print(check_flat)
    #



















