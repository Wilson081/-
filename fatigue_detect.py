import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from datetime import datetime, timedelta

# 加载模型
model = load_model('test_model.h5')

# Gamma值
gamma = 0.9

# 載入OpenCV的預訓練人臉和眼睛分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 檢查攝像頭是否打開
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # 從攝像頭讀取影像
    ret, frame = cap.read()

    # 檢查是否成功讀取幀
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # 轉換成灰階圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 在人臉區域內檢測眼睛
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Gamma校正
            gamma_corrected = np.array(255 * (eye_img / 255) ** gamma, dtype='uint8')
            # 轉換為RGB圖像並擴展通道維度
            rgb_img = cv2.merge([gamma_corrected] * 3)
            rgb_img = np.expand_dims(rgb_img, axis=0)
            rgb_img = preprocess_input(rgb_img)
            # # 使用模型進行預測

            prediction = model.predict(rgb_img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            if predicted_class == 0:
                eye_state = "Closed"
            else:
                eye_state = "Open"

            # print(predicted_class)
    try:
        cv2.putText(frame, f'Eyes: {eye_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2, cv2.LINE_AA)
    except:
        pass

    # 顯示影像
    cv2.imshow('Face and Eye Detection', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉所有視窗
cap.release()
cv2.destroyAllWindows()

        
