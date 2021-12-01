import cv2
from human_tracking import detect
import os

class Extractor:

    def __init__(self):
        self.clip = None
        self.images = None
        self.found_players_dir = r'D:\PycharmProjects\DeepLearningNBA\found_players'
        self.image_no = 1

    def extract_players_from_video(self, video_path):
        self.clip = video_path
        cap = cv2.VideoCapture(video_path)
        self.video_path_dir = video_path.split("\\")[-1].split(".")[0]
        if self.video_path_dir not in os.listdir(r'D:\PycharmProjects\DeepLearningNBA\found_players'):
            os.mkdir(os.path.join(r'D:\PycharmProjects\DeepLearningNBA\found_players', self.video_path_dir))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_analyzed = 0

        print(f"Total frames for processsing {total}")

        while (cap.isOpened()):
            frames_analyzed += 1
            ret, frame = cap.read()
            print("Frames analyzed: ", frames_analyzed)
            self.extract_from_frame(frame)


    def extract_from_frame(self, frame):
        try:
            HOGCV = cv2.HOGDescriptor()
            HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(20, 20), padding=(5, 5), scale=1.03)
            print(f"working frame found {len(bounding_box_cordinates)} coordinates")

            for x, y, w, h in bounding_box_cordinates:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                crop_frame = frame[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(r'D:\PycharmProjects\DeepLearningNBA\found_players', self.video_path_dir, f"player_{self.image_no}.jpg"), crop_frame)

                self.image_no += 1

            return frame

        except Exception as e:
            print("Frame Extraction failed")
            print(e)