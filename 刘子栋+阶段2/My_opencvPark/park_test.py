# -*- coding: gbk -*-
from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os,glob
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from Parking import Parking
import pickle


cwd = os.getcwd()

def img_process(test_images,park):
    #ȥ������
    white_yellow_images = list(map(park.select_rgb_white_yellow,test_images))  # map(func,case) ��caseʹ��func
    park.show_images(white_yellow_images)
    #�Ҷ�ͼ
    gray_images = list(map(park.convert_gray_scale,white_yellow_images))
    park.show_images(gray_images)
    #��Ե���
    edge_images = list(map(park.detect_edges,gray_images))
    park.show_images(edge_images)
    #ȥ������Ҫ�����򣬱���ͣ��������
    roi_images = list(map(park.select_region,edge_images))
    park.show_images(roi_images)
    # ���ֱ�ߣ�����ֱ�������ͣ��λλ��
    list_of_lines = list(map(park.hough_lines,roi_images))
    # ����ͣ��λֱ��
    line_iamges = []
    for image,lines in zip(test_images,list_of_lines):
        line_iamges.append(park.draw_lines(image,lines))
    park.show_images(line_iamges)
    # ����ͣ��λÿ�еľ��ο򣬲���ͣ��λ�����
    rect_images = []   # ���õ�ͼ
    rect_coords = []   # ÿ�е�λ��
    for image,lines in zip(test_images,list_of_lines):
        new_image,rects = park.identify_blocks(image,lines)
        rect_images.append(new_image)
        rect_coords.append(rects)
    park.show_images(rect_images)
    # �з�ÿ�е�ͣ��λ
    delineated = []
    spot_pos = []  #��λ���
    for image,rects in zip(test_images,rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)
    park.show_images(delineated)

    final_spot_dict = spot_pos[1]
    #����λ������λ�ü���ű���
    with open('spot_dict.pickle','wb') as f:
        pickle.dump(final_spot_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    # ׼��ѵ������ĳ�λ���ݼ�
    park.save_images_for_cnn(test_images[0],final_spot_dict)

    return final_spot_dict

def keras_model(weights_path):
    model = load_model(weights_path)
    return model

def img_test(test_images,final_spot_dict,model,class_dict):
    for i in range(len(test_images)):
        predicted_images = park.predict_on_image(test_images[i],final_spot_dict,model,class_dict)

def video_test(video_name,final_spot_dict,model,class_dictionary):
    name = video_name
    # cap = cv2.VideoCapture(name)
    park.predict_on_video(name,final_spot_dict,model,class_dictionary,ret=True)

if __name__ == "__main__":
    # ׼������ͼƬ
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    # ����ѵ���õ�ģ���е�Ȩ�ز���
    weights_path = 'car1.h5'

    video_name = 'parking_video.mp4'

    # ͣ��λ��𣺿�orռ��
    class_dict = {}
    class_dict[0] = 'empty'
    class_dict[1] = 'occupied'

    park = Parking()
    # �򿪲���ͼ��
    park.show_images(test_images)

    # ͼ��Ԥ����
    final_spot_dict = img_process(test_images,park)

    #��ģ��ѵ��
    model = keras_model(weights_path)
    img_test(test_images,final_spot_dict,model,class_dict)
    video_test(video_name,final_spot_dict,model,class_dict)