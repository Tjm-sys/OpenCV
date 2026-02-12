import cv2
import numpy
import sys

videoCapture = cv2.VideoCapture(0)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

while True:
    ret, frame = videoCapture.read(0)
    if not ret: break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    finalFrame = blurFrame

    circles = cv2.HoughCircles(finalFrame,cv2.HOUGH_GRADIENT,5,100,
        param1=100,param2=50,minRadius=20,maxRadius=60)

    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) > dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i

        cv2.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 255, 0), 3)
        prevCircle = chosen

    cv2.imshow("circles", frame)

    if cv2.waitKey(1) == 27: 
        break

videoCapture.release()
cv2.destroyAllWindows()
