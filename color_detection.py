# Read the image

import numpy as np
import cv2

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, dsize=(300,300),interpolation = cv2.INTER_AREA)


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 400)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 25, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 7, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 93, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    path = "C:/Users/rashi/Desktop/Elevator/overlap2/6.jpg"
    image = cv2.imread(path)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    Sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    Sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    Val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    Val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, Sat_min, Sat_max, Val_min, Val_max)

    lower = np.array([h_min, Sat_min, Val_min])
    upper = np.array([h_max, Sat_max, Val_max])
    mask = cv2.inRange(HSV, lower, upper)
    imgResult = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("Original", image)
    cv2.imshow("HSV", HSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked", imgResult)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# if __name__ == "__main__":

