import cv2
import numpy as np

def mode1(cap) :
    track_window = None
    roi_hist = None
    trem_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    ret, frame = cap.read()
    x, y, w, h = cv2.selectROI("selectROI", frame, False, False)
    print(f"선택한 박스 좌표 : ({x}, {y}, {w}, {h})")

    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    track_window = (x,y,w,h)

    while True :
        ret, frame = cap.read()
        if not ret :
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        _, track_window = cv2.meanShift(dst, track_window, trem_crit)

        x, y, w, h = track_window
        print(f"추적 결과 좌표 : ({x}, {y}, {w}, {h})")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("Mean Shift Tracking", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
def mode2(cap):
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
    ret, frame = cap.read()
    print(ret, frame) 
    bbox_info = cv2.selectROI("Select object", frame, False, False)
    print(f"box info : {bbox_info}")

    kalman.statePre = np.array([[bbox_info[0]], [bbox_info[1]], [0], [0]], np.float32)
    while True : 
        ret, frame = cap.read()
        if not ret :
            print("프레임 읽기 실패")
            break
        kalman.correct(np.array([[np.float32(bbox_info[0] + bbox_info[2] / 2)], [np.float32(bbox_info[1] + bbox_info[3] / 2)]]))
        kalman.predict()
        predicted_bbox = tuple(map(int, kalman.statePost[:2, 0]))

        cv2.rectangle(frame, (predicted_bbox[0] - bbox_info[2] // 2, predicted_bbox[1] - bbox_info[3] // 2),
                    (predicted_bbox[0] + bbox_info[2] // 2, predicted_bbox[1] + bbox_info[3] // 2),
                    (0,255,0), 2)
        
        cv2.imshow("Kalman Filter Tracking", frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q') : 
            break
    cv2.destroyAllWindows()

def mode3(cap):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    print(feature_params)
    lk_params = dict(winSize=(15,15), 
                    maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_corner = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    prev_points = prev_corner.squeeze()
    color = (0,255,0)
    while True :
        ret, frame = cap.read()
        if not ret : 
            print("프레임 읽기 실패")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        for i, (prev_point, next_point) in enumerate(zip(prev_points, next_points)) :
            x1, y1 = prev_point.astype(int)
            x2, y2 = next_point.astype(int)
            cv2.line(frame, (x1,y1), (x2,y2), color, 2)
            cv2.circle(frame, (x2,y2), 3, color, -1)
        cv2.imshow("Feature-based Tracking", frame)
        prev_gray = gray.copy()
        prev_points = next_points
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def mode4(cap) :
    sift = cv2.SIFT_create()
    while True : 
        ret, frame = cap.read()
        if not ret : break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SIFT", frame)
        if cv2.waitKey(30) & 0xFF == ord('q') : 
            break
    cv2.destroyAllWindows()

def mode5(cap) :
    sift = cv2.SIFT_create(contrastThreshold=0.02)
    max_keypoints = 100 
    while True :
        ret, frame = cap.read()
        if not ret : break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # print(keypoints, descriptors)

        if len(keypoints) > max_keypoints :
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]
        frame = cv2.drawKeypoints(
            frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow("SIFT", frame)
        if cv2.waitKey(30) & 0xFF == ord('q') : 
            break
    cv2.destroyAllWindows()

def mode6(cap) : 
    orb = cv2.ORB_create()
    while True : 
        ret, frame = cap.read()
        if not ret : break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = orb.detect(gray, None)
        frame = cv2.drawKeypoints(frame, keypoints, None, (0,150,220), flags=0)
        cv2.imshow("0RB", frame)
        if cv2.waitKey(30) & 0xFF == ord('q') :
            break
    cv2.destroyAllWindows()

def mode7(cap) :
    orb = cv2.ORB_create()
    min_keypoint_size = 10
    duplicate_threshold = 10
    while True : 
        ret, frame = cap.read()
        if not ret : 
            print("프레임 읽기 실패")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = orb.detect(gray, None)
        keypoints = [kp for kp in keypoints if kp.size > min_keypoint_size]
        mask = np.ones(len(keypoints), dtype=bool)
        for i, kp1 in enumerate(keypoints) :
            if mask[i] :
                for j, kp2 in enumerate(keypoints[i+1 :]) :
                    if (
                        mask[i+j+1]
                        and np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
                        < duplicate_threshold
                    ) :
                        mask[i+j+1] = False
        keypoints = [kp for i, kp in enumerate(keypoints) if mask[i]]
        frame = cv2.drawKeypoints(frame, keypoints, None, (0,200,150), flags=0)
        cv2.imshow("0RB", frame)
        if cv2.waitKey(30) & 0xFF == ord('q') :
            break
    cv2.destroyAllWindows()

cap1 = cv2.VideoCapture('./data/slow_traffic_small.mp4')
cap2 = cv2.VideoCapture('./data/vtest.avi')
for i in range(1, 5):
    eval(f"mode{i}(cap1)")
    
for i in range(5, 8):
    eval(f"mode{i}(cap2)")

cap1.release()
cap2.release()
cv2.destroyAllWindows()