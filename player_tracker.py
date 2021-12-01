import cv2
import numpy as np

class PlayerTracker:
    def __init__(self):
        pass

    def get_jersey_color(self, frame):
        def mouseRGB(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
                colorsB = frame[y, x, 0]
                colorsG = frame[y, x, 1]
                colorsR = frame[y, x, 2]
                colors = frame[y, x]
                print("Red: ", colorsR)
                print("Green: ", colorsG)
                print("Blue: ", colorsB)
                print("BRG Format: ", colors)
                print("Coordinates of pixel: X: ", x, "Y: ", y)

        cv2.imshow("result", frame)

        cv2.setMouseCallback('result', mouseRGB)
        cv2.waitKey(0)

    def locate_bodies(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # green range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        # blue range
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Red range
        lower_red = np.array([0, 31, 255])
        upper_red = np.array([176, 255, 255])

        # white range
        lower_white = np.array([137, 163, 208])
        upper_white = np.array([157, 183, 218])

        # Define a mask ranging from lower to uppper
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Do masking
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # convert to hsv to gray
        res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # Defining a kernel to do morphological operation in threshold image to
        # get better output.
        kernel = np.ones((13, 13), np.uint8)
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # find contours in threshold image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        prev = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Detect players
            if (h >= (1.5) * w):
                if (w > 15 and h >= 15):
                    idx = idx + 1
                    player_img = frame[y:y + h, x:x + w]
                    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                    # If player has blue jersy
                    mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)
                    # If player has red jersy
                    mask2 = cv2.inRange(player_hsv, lower_white, upper_white)
                    res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                    nzCountred = cv2.countNonZero(res2)

                    if (nzCount >= 20):
                        # Mark blue jersy players as france
                        cv2.putText(frame, 'France', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    else:
                        pass
                    if (nzCountred >= 20):
                        # Mark red jersy players as belgium
                        cv2.putText(frame, 'Belgium', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, 'Belgium', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        pass
            if ((h >= 1 and w >= 1) and (h <= 30 and w <= 30)):
                player_img = frame[y:y + h, x:x + w]

                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                # white ball  detection
                mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                if (nzCount >= 3):
                    # detect football
                    cv2.putText(frame, 'football', (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return frame




