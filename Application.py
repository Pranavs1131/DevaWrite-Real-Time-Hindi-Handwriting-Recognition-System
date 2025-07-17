import cv2
import numpy as np
from keras.models import load_model
from collections import deque

# Load trained model
model = load_model(r'C:\Users\PRANAV SINGH\Desktop\Handwriting Classification\CNN-Hindi-Handwriting-Recognition-master\devanagari_model.h5')

# Label mapping
letter_count = {
    0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: '06_cha',
    7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna', 11: '11_taa', 12: '12_thaa', 13: '13_daa',
    14: '14_dhaa', 15: '15_adna', 16: '16_ta', 17: '17_tha', 18: '18_da', 19: '19_dha', 20: '20_na',
    21: '21_pa', 22: '22_pha', 23: '23_ba', 24: '24_bha', 25: '25_ma', 26: '26_yaw', 27: '27_ra',
    28: '28_la', 29: '29_waw', 30: '30_saw', 31: '31_petchiryakha', 32: '32_patalosaw', 33: '33_ha',
    34: '34_chhya', 35: '35_tra', 36: '36_gya', 37: 'CHECK'
}

# Webcam and config
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
Lower_blue = np.array([100, 150, 0])
Upper_blue = np.array([140, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# State
pts = deque(maxlen=1024)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
preview_digit = np.zeros((200, 200), dtype=np.uint8)
pred_class = 0
pred_probab = 0.0
no_pen_counter = 0
DRAWING_TIMEOUT_FRAMES = 30

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, Lower_blue, Upper_blue)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    center = None

    if len(cnts) > 0:
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                pts.appendleft(center)
                cv2.circle(img, center, 8, (0, 255, 255), -1)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 24, lineType=cv2.LINE_AA)
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 8, lineType=cv2.LINE_AA)

            no_pen_counter = 0

    else:
        if len(pts) != 0:
            no_pen_counter += 1
            if no_pen_counter > DRAWING_TIMEOUT_FRAMES:
                gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # ✅ FIXED: use THRESH_BINARY (white on black)
                thresh_digit = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                cnts = cv2.findContours(thresh_digit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(cnts) > 0:
                    cnt = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 1000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        pad = 30
                        x = max(x - pad, 0)
                        y = max(y - pad, 0)
                        w = min(w + 2 * pad, gray.shape[1] - x)
                        h = min(h + 2 * pad, gray.shape[0] - y)
                        roi = thresh_digit[y:y + h, x:x + w]

                        roi_resized = cv2.resize(roi, (32, 32)).astype("float32") / 255.0
                        roi_input = roi_resized.reshape(1, 32, 32, 1)

                        pred = model.predict(roi_input)[0]
                        pred_class = np.argmax(pred)
                        pred_probab = np.max(pred)

                        preview_digit = cv2.resize(roi, (200, 200))
                        print(f"Predicted: {letter_count[pred_class]} | Confidence: {pred_probab:.2f}")

                pts.clear()
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                no_pen_counter = 0
        else:
            no_pen_counter = 0

    # ✅ No confidence filtering for now
    label = f"{letter_count[pred_class]} ({pred_probab*100:.1f}%)"
    cv2.putText(img, "Prediction: " + label, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show windows
    cv2.imshow("Live Feed", img)
    cv2.imshow("Pen Mask", thresh)
    cv2.imshow("Stroke Preview", preview_digit)

    # Controls
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        pts.clear()
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        preview_digit = np.zeros((200, 200), dtype=np.uint8)
        pred_class = 0
        pred_probab = 0.0
        print("🔁 Reset")

cap.release()
cv2.destroyAllWindows()
