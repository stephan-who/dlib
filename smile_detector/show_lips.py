import dlib
import cv2
from get_features import get_features

path_test_img = "../data/data_imgs/test_imgs/i064rc-mn.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/dlib/shape_predictor_68_face_landmarks.dat")


# get lip's positions of features points
positions_lip = get_features(path_test_img)

img_rd = cv2.imread(path_test_img)

#Draw on the lip points
for i in range(0, len(positions_lip), 2):
    print(positions_lip[i], positions_lip[i+1])
    cv2.circle(img_rd, tuple([positions_lip[i], positions_lip[i+1]]), radius=1, color=(0, 255, 0))


cv2.namedWindow("img_read", 2)
cv2.imshow("img_read", img_rd)
cv2.waitKey(0)




