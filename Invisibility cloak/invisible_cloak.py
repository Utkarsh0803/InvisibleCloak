import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
background = cv2.imread('./image.jpg')

while cap.isOpened():
    #capture the live frame
    ret, current_frame = cap.read()
    if ret:
        #converting from rgb to hsv color space
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # ------ YELLOW COLOR RANGE ------
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Noise removal
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        # substituting the yellow portion with background image
        part1 = cv2.bitwise_and(background, background, mask=yellow_mask)

        # detecting the non-yellow part
        yellow_free = cv2.bitwise_not(yellow_mask)

        # if cloak is not present, show the current image
        part2 = cv2.bitwise_and(current_frame, current_frame, mask=yellow_free)

        # final output
        cv2.imshow("cloak", part1 + part2)

        cv2.imshow("yellow cloak", part1)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
