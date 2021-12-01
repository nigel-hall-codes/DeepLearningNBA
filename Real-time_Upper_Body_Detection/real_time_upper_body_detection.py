import cv2
import imutils

haar_upper_body_cascade = cv2.CascadeClassifier("data/haarcascade_fullbody.xml")

# Uncomment this for real-time webcam detection
# If you have more than one webcam & your 1st/original webcam is occupied,
# you may increase the parameter to 1 or respectively to detect with other webcams, depending on which one you wanna use.

# video_capture = cv2.VideoCapture(0)

# For real-time sample video detection
video_capture = cv2.VideoCapture(r"D:\PycharmProjects\NBACV\test_videos\Steph2Wise.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)

frame_no = 0

while True:


    ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale

    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.05,
        minNeighbors = 1,
        minSize = (5, 40), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )



    # Draw a rectangle around the upper bodies
    for i, (x, y, w, h) in enumerate(upper_body):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "Upper Body Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
        # roi = frame[y:y + h, x:x + w]
        # cv2.imwrite(r'D:\PycharmProjects\DeepLearningNBA\found_players\frame{}-ub{}.jpg'.format(frame_no, i), roi)
    cv2.imshow('Video', frame) # Display video

    frame_no += 1

    # stop script when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
video_capture.release()
cv2.destroyAllWindows()
