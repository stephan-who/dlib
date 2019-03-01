import dlib
import numpy as np
import cv2
import os

#读取图像的路径
path_read = "data/images/faces_for_test"
img = cv2.imread(path_read + "test_faces_3.jpg")

#用来存储生成的单张人脸的路径
path_save = "data/images/faces_seperated/"


#Delete old images
def clear_images():
    imgs = os.listdir(path_save)

    for img in imgs:
        os.remove(path_save + img)

    print("clean finish", '\n')


clear_images()

#Dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

#DLib检测
faces = detector(img, 1)

print("人脸数：", len(faces), '\n')

for k, d in enumerate(faces):
    #计算矩形大小
    pos_start = tuple([d.left(), d.top()])
    pos_end = tuple([d.right(), d.bottom()])

    #计算矩形框的大小
    height = d.bottom()-d.top()
    width = d.right() - d.left()

    #根据人脸大小生成空的图像
    img_blank = np.zeros((height, width, 3), np.uint8 )

    for i in range(height):
        for j in range(width):
            img_blank[i][j] = img[d.top()+i][d.left()+j]

    #存到本地
    print("save to:", path_save+"img_face_"+str(k+1)+".jpg")
    cv2.imwrite(path_save+"img_face_"+str(k+1)+".jpg", img_blank)



