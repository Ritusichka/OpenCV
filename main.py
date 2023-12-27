import cv2
import numpy as np
import time

video_capture = cv2.VideoCapture('/Users/ritusicka/Desktop/pythonProject3/3.mp4')

ret, first_frame = video_capture.read()
if not ret:
    exit(0)

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

last_detection_time = time.time()
detection_interval = 5

detected = False
detected_box = None

output_path = '/Users/ritusicka/Desktop/pythonProject3/output_video3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 45.0, (first_frame.shape[1], first_frame.shape[0]))

while True:
    ret, frame = video_capture.read()

    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frame_diff = cv2.absdiff(first_gray, gray)


    threshold = 30
    _, thresholded = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if time.time() - last_detection_time > detection_interval:
        detected = True if contours else False
        if detected:

            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(max_contour)
                detected_box = (x, y, x + w, y + h)
            last_detection_time = time.time()


    if detected and detected_box is not None:
        x1, y1, x2, y2 = detected_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, 'lost', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Static Object Detection', frame)
    out.write(frame)

    key = cv2.waitKey(1)
    if key != -1:
        break


video_capture.release()
out.release()
cv2.destroyAllWindows()
