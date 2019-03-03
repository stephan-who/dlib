import dlib
import numpy as np
import cv2
import time
import timeit
import statistics

#
path_screenshots = "../data/screenshots/"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../data/dlib/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

cap.set(3, 480)

cnt = 0

time_cost_list = []

while cap.isOpened():
    
    flag, img_rd = cap.read()
    
    k =cv2.waitKey(1)
    
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    
    start = timeit.default_timer()

    faces = detector(img_gray, 0)

    #
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 标68个点
    if len(faces) != 0:
        for i in range(len(faces)):
            #取
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
            for idx, point in enumerate(landmarks):
                #68点的坐标
                pos = (point[0, 0], point[0,1])

                #利用cv.circle 给每个特征点画一个圈，共68个
                cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
                #利用cv2.putText
                cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255,255), 1, cv2.LINE_AA)

        cv2.putText(img_rd, "faces:" + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        stop = timeit.default_timer()
        time_cost_list.append(stop -start)
        print("%-15s %f" % ("Time cost:", (stop - start)))
    else:
        cv2.putText(img_rd, "no face", (20, 40), font, 1, (0,0,0), 1, cv2.LINE_AA)


    #添加说明
    img_rd = cv2.putText(img_rd, "press 'S': screenshot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA )
    img_rd = cv2.putText(img_rd, "press 'Q': quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    if k==ord('s'):
        cnt +=1
        print(path_screenshots + "screenshot" + "_"+ str(cnt) + "_" +time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        cv2.imwrite(path_screenshots + "screenshot" + "_" +str(cnt) +"_" +time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+ ".jpg",img_rd)

    if k== ord('q'):
        break

    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", img_rd)

cap.release()
cv2.destroyAllWindows()

print("%-15s" % "Result:")
print("%-15s %f" % ("Max time:", (max(time_cost_list))))
print("%-15s %f" % ("Min time:", (min(time_cost_list))))
print("%-15s %f" % ("Average time:", (statistics.mean(time_cost_list))))

