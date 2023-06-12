import cv2
import numpy as np

# 칼만 필터 초기화
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]], np.float32) * 0.05

# 칼만 필터 추적 실습
# 동영상 파일 읽기
cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

#첫 프레임에서 추적할 객체 선택
ret, frame = cap.read()
print(ret, frame) 
bbox_info = cv2.selectROI("Select object", frame, False, False)
print(f"box info : {bbox_info}")

# 객체 추적을 위한 초기 추정 위치 설정
# 객체의 x 좌표, 객체의 y 좌표, 객체의 x 방향속도(초기 0), 객체의 y 방향속도(초기 0)
kalman.statePre = np.array([[bbox_info[0]], [bbox_info[1]], [0], [0]], np.float32)

"""
"객체의 속도"라는 표현은 일반적으로 동영상에서 객체의 움직임 속도를 의미합니다. Kalman 필터를 사용하여 객체의 위치와 속도를 추정하는 경우, 이는 객체가 프레임 간에 이동하는 속도를 의미합니다.
일반적으로 속도는 "픽셀 단위/프레임" 또는 "미터 단위/초"와 같은 형태로 표현됩니다.
"""

while True : 
    # 프레임 읽기 
    ret, frame = cap.read()
    if not ret :
        print("프레임 읽기 실패")
        break

    # 칼만 필터를 사용하여 객체 위치 추정
    # 객체의 바운딩 박스 중심의 x와 y좌표로 구성된다
    kalman.correct(np.array([[np.float32(bbox_info[0] + bbox_info[2] / 2)], [np.float32(bbox_info[1] + bbox_info[3] / 2)]]))
    kalman.predict()

    # 칼만 필터로 추정된 객체 위치 
    predicted_bbox = tuple(map(int, kalman.statePost[:2, 0]))

    # 추정된 객체 위치를 사각형으로 표시
    cv2.rectangle(frame, (predicted_bbox[0] - bbox_info[2] // 2, predicted_bbox[1] - bbox_info[3] // 2),
                  (predicted_bbox[0] + bbox_info[2] // 2, predicted_bbox[1] + bbox_info[3] // 2),
                  (0,255,0), 2)
    
    # 프레임 출력
    cv2.imshow("Kalman Filter Tracking", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q') : 
        break

# 자원 해제 
cap.release()
cv2.destroyAllWindows()