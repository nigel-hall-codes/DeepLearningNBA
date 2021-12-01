import cv2
import numpy as np

class CourtIdentifier:
    def __init__(self):
        pass

    def display_lines(self, img, lines):
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def display_line(self, img, line):
        line_image = np.zeros_like(img)
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def canny(self, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = 5
        # blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
        # canny = cv2.Canny(blur, 50, 150)
        return img

    def region_of_interest(self, canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)

        triangle = np.array([[
            (200, height),
            (550, 250),
            (1100, height), ]], np.int32)

        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image

    def get_lines(self, frame):
        image = frame
        lane_image = np.copy(image)
        lane_canny = self.canny(lane_image)
        # cropped_canny = self.region_of_interest(lane_canny)
        lines = cv2.HoughLinesP(lane_canny, 2, np.pi/180, 20, np.array([]), minLineLength=.1, maxLineGap=5)
        return lines

    def make_points(self, image, line):
        slope, intercept = line
        y1 = int(image.shape[0])  # bottom of the image
        y2 = int(y1 * 3 / 5)  # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        if lines is None:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope > 0:  # y is reversed in image
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        # add more weight to longer lines
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_points(image, left_fit_average)
        right_line = self.make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines

    def medium_edge_map(self, image):
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        mid = cv2.Canny(blurred, 50, 150)

        return mid

    def small_edge_map(self, image):

        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        mid = cv2.Canny(blurred, 30, 375)

        return mid

    def find_baselines(self, image):


        lines = self.get_lines(image)

        baselines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            length = pow((x2 - x1), 2) + pow((y2 - y1), 2)

            if slope < 1 and slope > .4 and length > 6500:
                baselines.append(line)

        return baselines

    def three_point_line_equation_x(sel, y):
        return

    def find_three_point_arc(self):
        # Equation 0.000384x^2 - 0.01891y + 300.057
        pass



