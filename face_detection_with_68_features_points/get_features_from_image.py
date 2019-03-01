import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

img_rd = cv2.imread("data/samples/face_2.jpg")
img_gray= cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

faces = detector(img_gray, 0)

#
font = cv2.FONT_HERSHEY_SIMPLEX

#标68个点
if len(faces) != 0:
    for i in range(len(faces)):
        #取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            #68点的坐标
            pos = (point[0, 0], point[0,1])

            #利用cv.circle 给每个特征点画一个圈，共68个
            cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
            #利用cv2.putText
            cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255,255), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "faces:" + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

else:
    cv2.putText(img_rd, "no face", (20, 40), font, 1, (0,0,0), 1, cv2.LINE_AA)

#参数为0 可以拖动缩放窗口，为1不可以
cv2.namedWindow("image", 1)
cv2.imshow("image", img_rd)

cv2.waitKey(0)