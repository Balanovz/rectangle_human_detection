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
            mpDraw,
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


def findStraightMeasurement(lmList, unit):
    shoulders_length = abs(lmList[12][1] - lmList[11][1]) * unit
    hips_length = abs(lmList[23][1] - lmList[24][1]) * unit
    return {'shoulders': shoulders_length, 'hips': hips_length}


def findProfileMeasurement(bbox, unit):
    return bbox[2] * unit


def startRecord(ip: str):
    cap = cv2.VideoCapture(ip)
    cap.set(3, 640)
    cap.set(4, 480)
    real_height = float(input('Enter person height:\t'))
    is_straight_view = True
    shoulder_lengths = []
    hip_lengths = []
    human_widths = []

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
            if is_straight_view:
                human_body_part = findStraightMeasurement(lmList, unit)
                print(f'shoulders:\t{human_body_part["shoulders"]}')
                shoulder_lengths.append(human_body_part["shoulders"])
                print(f'hips:\t{human_body_part["hips"]}')
                hip_lengths.append(human_body_part["hips"])
            else:
                width = findProfileMeasurement(bboxInfo['bbox'], unit)
                human_widths.append(width)
                print(f'human width:\t{width}')
        # TODO исправить изменение режима съемки нажатия на кнопку на другой сигнал
        if cv2.waitKey(1) & 0xFF == ord(' '):
            input()
            is_straight_view = not is_straight_view
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return shoulder_lengths, hip_lengths, human_widths


def getHumanInfo(shoulder_average: float, hip_average: float, width_average: float):
    print(f'average length of shoulders:\t{shoulder_average}')
    print(f'average length of hips:\t{hip_average}')
    print(f'average length of widths:\t{width_average}')


def getAverageMeasurements(shoulder_lengths: list, hip_lengths: list, widths: list):
    return dict(shoulders=sum(shoulder_lengths) / len(shoulder_lengths),
                hips=sum(hip_lengths) / len(hip_lengths),
                widths=sum(widths) / len(widths))


def getMatchPercentInfo(first_data: dict, second_data: dict):
    print(f'match percentage of shoulders:\t {min(first_data["shoulders"], second_data["shoulders"]) / max(first_data["shoulders"], second_data["shoulders"]) * 100} %')
    print(f'match percentage of hips:\t  {min(first_data["hips"], second_data["hips"]) / max(first_data["hips"], second_data["hips"]) * 100} %')
    print(f'match percentage of width:\t {min(first_data["widths"], second_data["widths"]) / max(first_data["widths"], second_data["widths"]) * 100} %')


if __name__ == '__main__':
    camera_ip = 'http:/192.168.1.106:4747/video'
    shoulders_lengths, hips_lengths, widths = startRecord(camera_ip)
    first_average = getAverageMeasurements(shoulders_lengths, hips_lengths, widths)

    print('\nfirst person measurement:')
    getHumanInfo(first_average['shoulders'], first_average['hips'], first_average['widths'])

    shoulders_lengths, hips_lengths, widths = startRecord(camera_ip)
    second_average = getAverageMeasurements(shoulders_lengths, hips_lengths, widths)
    print('\nsecond person:')
    getHumanInfo(second_average['shoulders'], second_average['hips'], second_average['widths'])

    getMatchPercentInfo(first_average, second_average)
