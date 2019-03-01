import cv2


cap = cv2.VideoCapture(0)

cap.set(3, 480)

print(cap.isOpened())

while cap.isOpened():
    ret_flag, img_camera = cap.read()
    cv2.imshow("camera", img_camera)

    k = cv2.waitKey(1)

    if k ==ord('s'):
        cv2.imwrite('test.jpg', img_camera)

    if k== ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


