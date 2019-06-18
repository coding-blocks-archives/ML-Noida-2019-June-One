import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("../../datasets/haarcascade_frontalface_default.xml")

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # print(type(gray))
    faces = classifier.detectMultiScale(gray)

    if len(faces) > 0:
        f = faces[0]
        x, y, w, h = tuple(f)
        face = gray[y:y+h, x:x+w]
        print(f)
        print(face.shape)
        cv2.imshow("face", face)

    # cv2.imshow("video", gray)

    if cv2.waitKey(1) > 30:
        cv2.imwrite("class.jpg", gray)
        break

cap.release()
cv2.destroyAllWindows()