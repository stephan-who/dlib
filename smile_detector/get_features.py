import dlib
import numpy as np
import cv2
import os
import csv

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/dlib/shape_predictor_68_face_landmarks.dat")


## 输入图像文件所在路径，返回一个41维数组（包含提取到的40维特征和1维输出标记）
def get_features(img_rd):
    img = cv2.imread(img_rd)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #
    positions_68_arr = []
    faces = detector(img_gray, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[0]).parts()])

    for idx, point in enumerate(landmarks):
        pos = (point[0,0], point[0,1])
        positions_68_arr.append(pos)

    positions_lip_arr = []
    #将点49-68写入csv
    for i in range(48, 68):
        positions_lip_arr.append(positions_68_arr[i][0])
        positions_lip_arr.append(positions_68_arr[i][1])

    return positions_lip_arr

#
path_images_with_smiles ="../data/data_imgs/database/smiles/"
path_images_no_smiles = "../data/data_imgs/database/no_smiles/"


#
imgs_smiles = os.listdir(path_images_with_smiles)
imgs_no_smiles = os.listdir(path_images_no_smiles)

#
path_csv = "../data/data_csvs/"


#write the features into csv
def write_into_csv():
    with open(path_csv+"data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        print("###### with smiles ##########")
        for i in range(len(imgs_smiles)):
            print(path_images_with_smiles+imgs_smiles[i])

            #append "1" means "with smiles"
            features_csv_smiles = get_features(path_images_with_smiles+imgs_smiles[i])
            features_csv_smiles.append(1)
            print("position of lips:", features_csv_smiles, "\n")

            #
            writer.writerow(features_csv_smiles)

        #处理不带微笑点图片
        print("#########no smiles ##########")
        for i in range(len(imgs_no_smiles)):
            print(path_images_no_smiles+imgs_no_smiles[i])
            features_csv_no_smiles = get_features(path_images_no_smiles+imgs_no_smiles[i])
            features_csv_no_smiles.append(0)

            #
            writer.writerow(features_csv_no_smiles)

#写入csv
write_into_csv()

