import dlib
import cv2
import time

#储存截图的目录
path_screenshots = "data/images/screenshots/"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/dlib/shape_predictor_68_face_landmarks.dat")

#
cap = cv2.VideoCapture(0)

#
cap.set(3, 960)

# 截图screenshots的计数器
ss_cnt = 0

while cap.isOpened():
    flag, img_rd = cap.read()

    #每帧数据延时 1ms, 延时为0读取的是静态帧
    k = cv2.waitKey(1)

    #取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    #人脸数
    faces = detector(img_gray, 0)

    #待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    #按下'q'键退出
    if k == ord('q'):
        break
    else:
        #检测到人脸
        if len(faces) != 0:
            #记录每次开始写入人脸像素的位置：
            faces_start_width = 0

            for face in faces:
                cv2.rectangle(img_rd, tuple([face.left(), face.top()]), tuple([face.right(), face.bottom()]), (0, 255, 255), 2)

                height = face.bottom() - face.top()
                width = face.right() -face.left

                #进行人脸裁剪
                #如果没有超出摄像头边界
                if (face.bottom() < 480) and (face.right() < 640) and ((face.top() + height < 480) and (face.left() + width) < 640):
                    # 填充
                    for i in range(height):
                        for j in range(width):
                            img_rd[i][faces_start_width+j] = img_rd[face.top()+i][face.left()+j]

                    #更新faces_start_width的坐标
                    faces_start_width += width

            cv2.putText(img_rd, 'Faces in all:' + str(len(faces)), (20, 350), font, 0.8, (0,0,0), 1,cv2.LINE_AA)
        else:
            #没有检测到人脸
            cv2.putText(img_rd, "no face", (20,350), font, 0.8, (0,0,0), 1, cv2.LINE_AA)

        #添加说明
        img_rd = cv2.putText(img_rd, "Press 'S': Screen shot",(20,400), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
        img_rd = cv2.putText(img_rd, "Press 'Q': Quit", (20, 450), font, 0.8, (255,255,255), 1, cv2.LINE_AA)

    #按下“s”键保存
    if k == ord('s'):
        ss_cnt +=1
        print(path_screenshots + "screenshot" + "_" + str(ss_cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%s", time.localtime()) + ".jpg")

        cv2.imwrite(path_screenshots + "screenshot" + "_" +str(ss_cnt) + "_" +time.strftime("%Y-%m-%d-%H-%M-%s", time.localtime()) + ".jpg", img_rd)


    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", img_rd)

cap.release()

cv2.destroyAllWindows()


