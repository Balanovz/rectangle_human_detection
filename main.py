import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)


def findPose(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        if draw:
            mpDraw.draw_landmarks(img, results.pose_landmarks,
                                  mpPose.POSE_CONNECTIONS)
    return img


def findPosition(img, draw=True, bboxWithHands=False):
    lmList = []
    bboxInfo = {}
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

        # Bounding Box
        ad = abs(lmList[12][1] - lmList[11][1]) // 2
        if bboxWithHands:
            x1 = lmList[16][1] - ad
            x2 = lmList[15][1] + ad
        else:
            x1 = lmList[12][1] - ad
            x2 = lmList[11][1] + ad

        y2 = lmList[29][2] + ad
        y1 = lmList[1][2] - ad
        bbox = (x1, y1, x2 - x1, y2 - y1)
        cx, cy = bbox[0] + (bbox[2] // 2), \
                 bbox[1] + bbox[3] // 2

        bboxInfo = {"bbox": bbox, "center": (cx, cy)}

        if draw:
            cv2.rectangle(img, bbox, (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    return lmList, bboxInfo


cap = cv2.VideoCapture('http:/192.168.1.106:4747/video')
cap.set(3, 640)
cap.set(4, 480)
real_height = 183

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
    frame = findPose(frame)
    lmList, bboxInfo = findPosition(frame)
    cv2.imshow('output', frame)
    if bboxInfo:
        unit = real_height / bboxInfo['bbox'][3]
        # print(f'length of shoulders:\t{abs(lmList[11][1] - lmList[12][1]) * unit}')
        print(f"length profile:\t{bboxInfo['bbox'][2] * unit}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
