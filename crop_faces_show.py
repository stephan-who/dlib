import dlib
import numpy as np
import cv2

#Dlib 检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

#读取图像
path = "data/images/faces_for_test"
img=cv2.imread(path+"test_faces_1.jpg")

faces = detector(img,1)

print("人脸数:", len(faces), "\n")

#记录人脸矩阵大小
height_max = 0
width_sum = 0

for k, d in enumerate(faces):
    pos_start = tuple([d.left(), d.top()])
    pos_end = tuple([d.right(), d.bottom()])

    #计算矩形框大小
    height = d.bottom() - d.top()
    width = d.right() - d.left()

    #处理宽度
    width_sum += width

    #处理高度
    if height > height_max:
        height_max = height
    else:
        height_max = height_max

#绘制用来显示人脸的图像的大小
print("窗口大小：",
      '\n',"高度/ height:", height_max,
      '\n', "宽度 / width:", width_sum)

#空的图像
img_blank = np.zeros((height_max, width_sum, 3), np.uint8)

#记录每次开始写入人脸像素的宽度位置
blank_start = 0

#将人脸填充到img_blank
for k, d in enumerate(faces):
    height = d.bottom() - d.top()
    width = d.right() - d.left()

    #填充
    for i in range(height):
        for j in range(width):
            img_blank[i][blank_start+j] = img[d.top()+i][d.left()+j]

    #调整图像
    blank_start += width


cv2.namedWindow("img_faces")
cv2.imshow("img_faces",img_blank)
cv2.waitKey(0)




