import cv2

img = cv2.VideoCapture("t5_trim.mp4")

ret, frame = img.read()

cv2.imwrite("res/images/screenshot.png",frame)
