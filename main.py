import cv2


def detect_human(frame):
    bounding_box_coordinates, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(1, 1), scale=1.03)
    for x, y, w, h in bounding_box_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


def detect_by_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = detect_human(frame)
        cv2.imshow('output', cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


cv2.startWindowThread()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
detect_by_camera()
