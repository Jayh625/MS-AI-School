import cv2

# 동영상 파일 읽기
cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

# SIFT 객체 생성
sift = cv2.SIFT_create()

while True : 
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret : break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 특징점 그리기 
    frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # flages 매개변수에 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 를 지정하면 특징점의 크기와 방향을 고려하여 시각화합니다. 특징점의 크기에 따라 원의 반지름이 조정되고, 방향 정보에 따라 라인이 그려집니다. 이렇게 그려진 특징점은 시각적으로 더 풍부한 정보를 제공할 수 있습니다.
    
    # 프레임 출력
    cv2.imshow("SIFT", frame)

    # q키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q') : 
        break

cap.release()
cv2.destroyAllWindows()