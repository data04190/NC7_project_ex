import cv2
import json
import requests

def send_to_server(data):
    server_url = "http://localhost:5000/endpoint"  
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, data=json.dumps(data), headers=headers)

    # 응답 확인
    print(response.text)


def save_to_local(data, filename="eyes_data.json"):
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write("\n")  # 한 줄에 하나의 JSON 객체를 저장하기 위한 개행


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
                eyes_data.append({"x": int(x + ex), "y": int(y + ey), "width": int(ew), "height": int(eh)})

        # 서버에 데이터 전송
        save_to_local(eyes_data)

        # 결과 프레임 표시
        cv2.imshow('Frame', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_function():
	print("Testing fuction!")

test_function()
detect_eyes_from_webcam()
