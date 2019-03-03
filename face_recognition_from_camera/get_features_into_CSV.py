# 从人脸图像文件中提取人脸特征存入csv


#增加录入多涨人脸到CSV到功能

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd


#
path_photos_from_camera = "../data/data_faces_from_camera/"

#
path_csv_from_photos = "../data/data_csvs_from_camera/"

detector = dlib.get_frontal_face_detector()

#人脸预测器
predictor = dlib.shape_predictor("../data/dlib/shape_predictor_5_face_landmarks.dat")

#人脸识别模型
#face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("../data/dlib/dlib_face_recognition_resnet_model_v1.dat")

#返回单张图像到128d特征
def return_128d_features(path_img):
    img= io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 1) #增加第二个参数会改变画质，检测速度会减慢

    print("检测到人脸到图像：",path_img, "\n")

    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img, shape)
    else:
        face_descriptor = 0
        print("no face")

    print(face_descriptor)
    return face_descriptor

#
def write_into_csv(path_faces_personX, path_csv_from_photos):
    photos_list = os.listdir(path_faces_personX)
    with open(path_csv_from_photos, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if photos_list:
            for i in range(len(photos_list)):
                print("正在读到人脸图像：", path_faces_personX + "/" + photos_list[i])
                features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
                if features_128d ==0:
                    i+=1
                else:
                    writer.writerow(features_128d)
        else:
            print("Warning: Empty photos in " + path_faces_personX+'/')
            writer.writerow("")


#读到某人所有到人脸图像的数据，写入person_X.csv
faces  = os.listdir(path_photos_from_camera)
faces.sort()

for person in faces:
    print("##### " + person +"###### ")
    print(path_csv_from_photos + person + ".csv")
    write_into_csv(path_photos_from_camera+ person, path_csv_from_photos+ person +".csv")
print('\n')

#从CSV读取数据，计算128D特征的均值
def compute_the_mean(path_csv_from_photos):
    column_names = []

    #128d 特征
    for features_num in range(128):
        column_names.append("features_" + str(features_num +1))

    #利用pandas 读取csv
    rd = pd.read_csv(path_csv_from_photos, names = column_names)

    if rd.size !=0:
        #存放128d特征的均值
        feature_mean_list = []

        for feature_num in range(128):
            tmp_arr = rd["features_" + str(feature_num + 1)]
            tmp_arr = np.array(tmp_arr)

            #
            tmp_mean= np.mean(tmp_arr)
            feature_mean_list.append(tmp_mean)
    else:
        feature_mean_list = []
    return feature_mean_list


#
path_csv_from_photos_feature_all = "../data/features_all.csv"

#
# path_csv_from_photos = "../data/data_csvs_from_camera/"

with open(path_csv_from_photos_feature_all, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    csv_rd = os.listdir(path_csv_from_photos)
    csv_rd.sort()
    print("#### 得到的特征均值 / The generated average values of features sorted in :####")
    for i in range(len(csv_rd)):
        feature_mean_list = compute_the_mean(path_csv_from_photos + csv_rd[i])
        print(path_csv_from_photos + csv_rd[i])
        writer.writerow(feature_mean_list)


