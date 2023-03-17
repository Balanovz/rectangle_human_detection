import cv2
import math
import time
import mediapipe as mp
from ultralytics import YOLO

# for box tracking
model = YOLO('../YOLOWeights/yolov8n.pt')
# for human skeleton
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)


def getHumanSkeletonCoordinates(frame, show=False):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    lmList = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))

        if show:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        return lmList


def getBoxTrackingCoordinates(frame, show=False):
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # if name is person
            if box.cls[0] == 0:
                print(f'x1: {x1} y1: {y1} x2: {x2} y2:{y2}')
                if show:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (163, 83, 224), 1)
                return x1, y1, x2, y2


def startMeasurements(camera_ip: str, real_height: float, show=False):
    cap = cv2.VideoCapture(camera_ip)
    cap.set(3, 640)
    cap.set(4, 480)
    shoulder_lengths, hip_lengths, human_widths, human_heights = [], [], [], []
    timing = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
        cv2.rectangle(frame, (0, 20), (550, 60), (255, 255, 255), -1)

        if (time.time() - timing) < 10:
            cv2.putText(frame, f'PREPARE FRONT (10 sec) {round(time.time() - timing, 2)}', (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.75, (0, 0, 0), 2)

        elif 10 <= (time.time() - timing) < 20:
            # front view
            print('estimating pose part')
            cv2.putText(frame, f'REC FRONT (10 sec) {round(time.time() - timing - 10, 2)}', (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.75, (0, 0, 255), 2)
            lmList = getHumanSkeletonCoordinates(frame, show)
            if lmList:
                shoulder_lengths.append(math.hypot(lmList[12][0] - lmList[11][0], lmList[12][1] - lmList[11][1]))
                hip_lengths.append(math.hypot(lmList[24][0] - lmList[23][0], lmList[24][1] - lmList[23][1]))

        elif 20 <= (time.time() - timing) < 30:
            cv2.putText(frame, f'PREPARE SIDE (10 sec) {round(time.time() - timing - 20, 2)}', (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.75, (0, 0, 0), 2)

        elif 30 <= time.time() - timing < 40:
            # side view, 2
            print('estimating width and height part')
            cv2.putText(frame, f'REC SIDE (10 sec) {round(time.time() - timing - 30, 2)}', (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.75, (0, 0, 255), 2)
            box_coord = getBoxTrackingCoordinates(frame, show)
            if box_coord:
                human_widths.append(abs(box_coord[2] - box_coord[0]))
                human_heights.append(abs(box_coord[3] - box_coord[1]))
        else:
            break

        cv2.imshow('ai_clothing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not shoulder_lengths or hip_lengths or human_widths or human_heights:
        print("one of the measurement isn't completed")
        return len(shoulder_lengths), len(hip_lengths), len(human_widths)

    average_shoulders = sum(shoulder_lengths) / len(shoulder_lengths)
    average_hips = sum(hip_lengths) / len(hip_lengths)
    average_widths = sum(human_widths) / len(human_widths)
    average_heights = sum(human_heights) / len(human_heights)

    pix_to_cm = real_height / average_heights
    ROUND_COEFFICIENT = 0.62940333494

    shoulders_cm = pix_to_cm * average_shoulders
    hips_cm = pix_to_cm * average_hips
    width_cm = pix_to_cm * average_widths * ROUND_COEFFICIENT

    return shoulders_cm, hips_cm, width_cm


def getHumanInfo(shoulders: float, hips: float, width: float):
    print(f'length of shoulders:\t{shoulders} cm')
    print(f'length of hips:\t{hips} cm')
    print(f'length of widths:\t{width} cm')


def getMatchPercentInfo(first_data: list, second_data: list):
    shoulders_match = min(first_data[0], second_data[0]) / max(first_data[0], second_data[0]) * 100 if first_data[0] and second_data[0] else 0
    hips_match = min(first_data[1], second_data[1]) / max(first_data[1], second_data[1]) * 100 if first_data[1] and second_data[1] else 0
    width_match = min(first_data[2], second_data[2]) / max(first_data[2], second_data[2]) * 100 if first_data[2] and second_data[2] else 0
    print(f'\nmatch percentage of shoulders:\t {shoulders_match} %')
    print(f'match percentage of hips:\t  {hips_match} %')
    print(f'match percentage of width:\t {width_match} %')


def main():
    ip = 'http:/192.168.1.106:4747/video'
    height = float(input('Enter first person height:\t'))
    first_shoulders, first_hips, first_width = startMeasurements(ip, height, True)

    print('\nfirst person measurements:')
    getHumanInfo(first_shoulders, first_hips, first_width)

    height = float(input('Enter second person height:\t'))
    second_shoulders, second_hips, second_width = startMeasurements(ip, height, True)

    print('\nsecond person measurements:')
    getHumanInfo(second_shoulders, second_hips, second_width)

    getMatchPercentInfo([first_shoulders, first_hips, first_width], [second_shoulders, second_hips, second_width])


if __name__ == '__main__':
    main()
