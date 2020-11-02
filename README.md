# 集装箱号码识别


## 目录
- [简介](#简介)
- [环境](#环境)
- [展示](#展示)
- [使用](#使用)
    - [demo](#demo)
    - [后台运行](#后台运行)
    - [目录结构](#目录结构)

## 简介

项目为港口集装箱号码的检测识别；算法分为两阶段，先检测再识别。集装箱号包括:
* 箱主代号：由四位因为字母组成；如，CCIU。
- 箱体注册码：由六位数字组成；如，335231。
* 校验码：由一位数字组成；如，1。
- 箱型编号：其四位编码由数字字母混合组成；如，22G1。

其在集装箱上的布局有五种形状，如下图所示。算法可对任意的集装箱号做检测识别，并且其校验准确率达到百分之九十四。由于工作原因代码模型不能做开源，此处只做展示，模型输出可见文件夹 ./datas/nohup/nohuo.out。

关键词：`集装箱` , `OCR`。

## 环境

    rehat7.1
    python3.6
    tensorflow==1.13.1
    torch==0.4.1

## 展示

![识别结果一](https://github.com/xumoremore/container-num-OCR/blob/main/datas/test_result/12643_5_psenet.jpg)
![识别结果二](https://github.com/xumoremore/container-num-OCR/blob/main/datas/test_result/14145_7_psenet.jpg)
![识别结果三](https://github.com/xumoremore/container-num-OCR/blob/main/datas/test_result/16481_4_psenet.jpg)
![识别结果四](https://github.com/xumoremore/container-num-OCR/blob/main/datas/test_result/19040_5_psenet.jpg)
![识别结果五](https://github.com/xumoremore/container-num-OCR/blob/main/datas/test_result/19825_7_psenet.jpg)

## 使用

### demo

    cd ./container-num-OCR
    python demo.py
    
### 后台运行

    cd ./container-num-OCR
    python flask.py

### 目录结构



