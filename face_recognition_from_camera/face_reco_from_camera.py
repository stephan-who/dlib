#摄像头实时人脸识别

import dlib
import numpy as np
import cv2
import cv2
import pandas as pd

facerec = dlib.face_recognition_model_v1("../data/dlib/dlib_face_recognition_resnet_model_v1.dat")

#计算两个128d向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)

    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print('e_distance:', dist)

    if dist> 0.6:
        return "diff"
    else:
        return "same"


path_features_known_csv = "../data/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

features_known_arr =[]

#读取已知人脸数据
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i,:][j])
    features_known_arr.append(features_someone_arr)
print("Faces in Database:", len(features_known_arr))

#
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/dlib/shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

cap.set(3, 480)

#
def get_128d_features(img_gray):
    faces = detector(img_gray, 1)
    if len(faces) != 0:
        face_des = []
        for i in range(len(faces)):
            shape = predictor(img_gray, faces[i])
            face_des.append(facerec.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des

#
while cap.isOpened():

    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    #
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    # 人脸数 faces
    faces = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_COMPLEX

    # 存储当前摄像头中捕获到的所有人脸的坐标/名字
    pos_namelist = []
    name_namelist = []

    # 按下 q 键退出
    if kk == ord('q'):
        break
    else:
        # 检测到人脸
        if len(faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

            for k in range(len(faces)):
                name_namelist.append("unknown")
                #每个人脸的名字坐标
                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() +(faces[k].bottom() -faces[k].top())/4 )]))

            for i in range(len(features_known_arr)):
                print("with person_", str(i+1), "the", end="")
                #将某张人脸与存储的所有人脸数据进行比对
                compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                if compare == "same":
                    if i==0:
                        name_namelist[k] = "Person 1"
                    elif i==1:
                        name_namelist[k] = "Person 2"
                    elif i==2:
                        name_namelist[k] = "Person 3"
            for kk, d in enumerate(faces):
                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0,255,255), 2)

        #  在人脸框下面写人脸名字
        for i in range(len(faces)):
            cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0,255,255), 1, cv2.LINE_AA)

    print("Name list now:", name_namelist, "\n")

    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("camera", img_rd)

cap.release()

cv2.destroyAllWindows()

