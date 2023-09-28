import cv2
import json
import requests

def detect_eyes_from_webcam():
    # Haar cascades 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    cap = cv2.VideoCapture(0)  # 웹캠 열기

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_data = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eyes_data.append({"x": x + ex, "y": y + ey, "width": ew, "height": eh})

        # 서버에 데이터 전송
        send_to_server(eyes_data)

        # 결과 프레임 표시
        cv2.imshow('Frame', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def send_to_server(data):
    server_url = "http://localhost:5000/endpoint"  
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, data=json.dumps(data), headers=headers)

    # 응답 확인
    print(response.text)

detect_eyes_from_webcam()
