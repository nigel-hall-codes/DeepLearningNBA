import court_identifier
import cv2
import time
import numpy as np
import numpy.polynomial.polynomial as poly
import player_tracker

class Tests:

    def __init__(self):
        self.court_identifier = court_identifier.CourtIdentifier()
        self.image = r"D:\PycharmProjects\NBACV\test_screenshots\half-court-setup.PNG"
        self.image = cv2.imread(self.image)
        self.player_tracker = player_tracker.PlayerTracker()
        self.x_coords = []
        self.y_coords = []
        self.p = [2.61526062e-04, -8.09734422e-02,  2.97225722e+02]

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw circle here (etc...)
            self.x_coords.append(x)
            self.y_coords.append(y)

            coefs = poly.polyfit(self.x_coords, self.y_coords, 4)
            ffit = poly.polyval(1019, coefs)

            print(np.polyfit(self.x_coords, self.y_coords, 2))

            print('x = %d, y = %d' % (x, y))

    def get_polyfit_with_click(self):
        mid = self.court_identifier.small_edge_map(self.image)
        cv2.imshow("result", mid)
        cv2.setMouseCallback('result', self.onMouse)
        cv2.waitKey(0)
        return np.polyfit(self.x_coords, self.y_coords, 5)


    def test_draw_3pt_line_from_baseline(self):

        p = self.p
        f = np.poly1d(p)

        mid = self.court_identifier.small_edge_map(self.image)
        baselines = self.court_identifier.find_baselines(mid)
        for baseline in baselines:
            new_x = np.linspace(baseline[0][0], baseline[0][2], 50)
            new_y = f(new_x)

            red = [255, 255, 0]

            for x, y in zip(new_x, new_y):
                print(x, y)
                cv2.circle(mid, (int(x), int(y)), 1, red, 1)

        line_image = self.court_identifier.display_lines(mid, baselines)
        combo_image = cv2.addWeighted(mid, 0.8, line_image, 1, 1)

        cv2.imshow("result", combo_image)
        cv2.setMouseCallback('result', self.onMouse)
        cv2.waitKey(0)


    def test_show_image(self):
        image = r"D:\PycharmProjects\NBACV\test_screenshots\half-court-setup.PNG"
        frame = cv2.imread(image)
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)



        cv2.imshow("result", image)

        cv2.setMouseCallback('result', self.onMouse)
        cv2.waitKey(0)

    def test_draw_three_point_line(self):
        image = r"D:\PycharmProjects\NBACV\test_screenshots\half-court-setup.PNG"
        frame = cv2.imread(image)
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)



        cv2.imshow("result", image)
        cv2.waitKey(0)


    def test_key_outline(self):

        image = r"D:\PycharmProjects\NBACV\test_screenshots\clipboard.jpg"
        frame = cv2.imread(image)
        frame = cv2.resize(frame, (0,0), fx=0.9, fy=0.9)
        lines = self.court_identifier.get_lines(frame)

        averaged_lines = self.court_identifier.average_slope_intercept(frame, lines)
        line_image = self.court_identifier.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)

        cv2.waitKey(0)

    def test_edge_detection(self):
        image = r"D:\PycharmProjects\NBACV\test_screenshots\half-court-setup.PNG"
        frame = cv2.imread(image)
        mid = self.court_identifier.medium_edge_map(frame)
        cv2.imshow("result", mid)
        cv2.waitKey(0)

    def test_get_lines_from_edge_map(self):
        mid = self.court_identifier.small_edge_map(self.image)

        lines = self.court_identifier.get_lines(mid)

        long_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if pow((x2 - x1), 2) + pow((y2 - y1), 2) > 1600:
                long_lines.append(line)
                # line_image = self.court_identifier.display_line(mid, line)

        line_image = self.court_identifier.display_lines(mid, long_lines)
        combo_image = cv2.addWeighted(mid, 0.8, line_image, 1, 1)

        cv2.imshow("result", combo_image)
        cv2.waitKey(0)

    def test_get_base_lines_from_edge_map(self):

        mid = self.court_identifier.small_edge_map(self.image)
        baselines = self.court_identifier.find_baselines(mid)

        line_image = self.court_identifier.display_lines(mid, baselines)
        combo_image = cv2.addWeighted(mid, 0.8, line_image, 1, 1)

        cv2.imshow("result", combo_image)
        cv2.waitKey(0)

    def test_play_video_and_search_baseline(self):
        cap = cv2.VideoCapture(r"D:\PycharmProjects\NBACV\test_videos\Steph2Wise.mp4")
        while (cap.isOpened()):
            _, frame = cap.read()
            mid = self.court_identifier.small_edge_map(frame)
            baselines = self.court_identifier.find_baselines(mid)

            line_image = self.court_identifier.display_lines(mid, baselines)
            combo_image = cv2.addWeighted(mid, 0.8, line_image, 1, 1)

            cv2.imshow("result", combo_image)
            cv2.waitKey(50)

    def test_locate_players(self):
        frame = self.image
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        frame = self.player_tracker.locate_bodies(frame)
        cv2.imshow("result", frame)
        cv2.waitKey(0)

    def test_get_images_for_clustering(self):
        import deep_clustering_algorithm

        algo = deep_clustering_algorithm.ClusteringAlgorithm()
        print(algo.found_player_images())


    def test_person_locator(self):
        from human_tracking import detect
        frame = self.image
        frame = detect(frame)

        cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('output', frame)
        print(frame)
        cv2.waitKey(0)

    def test_extractor(self):
        import player_image_extractor

        extractor = player_image_extractor.Extractor()
        extractor.extract_players_from_video(r"D:\PycharmProjects\NBACV\test_videos\Steph2Wise.mp4")


    def run_trainer(self):
        from trainer import Trainer

        t = Trainer()
        print(t.train())

    def test_extract_features(self):
        from trainer import Trainer
        img = r'D:\PycharmProjects\DeepLearningNBA\found_players\Steph2Wise\player_7.jpg'
        t = Trainer()
        print(t.extract_features(img))


    def test_get_labels(self):
        from trainer import Trainer
        t = Trainer()
        t.get_labels()

t = Tests()
t.test_get_labels()
# t.test_get_images_for_clustering()