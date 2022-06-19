import mediapipe as mp
import cv2 as cv
import hand_tracking_module as htm
import time

capture = cv.VideoCapture(0)
detector = htm.detector(max_hands=1)
ptime = 0

overlay = []
track_id = [8, 12, 16, 20]

for i in range(0, 6):
    img = cv.imread(f"{i + 1}.png")
    overlay.append(img)

while True:
    success, frame = capture.read()
    frame = cv.resize(frame, (540, 420))
    frame = detector.draw_hands(frame)
    lm_list = detector.find_pos(frame, draw=False)

    fingers = []

    if len(lm_list) != 0:
        if lm_list[4][1] < lm_list[3][1]:
            fingers.append(0)
        else:
            fingers.append(1)

        for id in track_id:
            if lm_list[id][2] < lm_list[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        fingers_stand = fingers.count(1)

        overlay[fingers_stand - 1] = cv.resize(overlay[fingers_stand - 1], (150, 150))
        frame[0:150, 0:150] = overlay[fingers_stand - 1]

        frame[250:400, 0:150] = (0, 255, 0)
        cv.putText(frame, str(fingers_stand), (25, 380), cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), thickness=8)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv.putText(frame, f"FPS : {str(int(fps))}%", (350, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), thickness=2)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
