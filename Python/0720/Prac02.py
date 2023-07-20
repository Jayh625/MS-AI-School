import cv2 

# 이미지 로드 
image1 = cv2.imread("1.png")
image2 = cv2.imread("2.png")

image1 = cv2.resize(image1, (500,500))
image2 = cv2.resize(image2, (500,500))

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 이미지2 -> 90 회전
image2_rotated = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
gray2_rotated = cv2.rotate(gray2, cv2.ROTATE_90_CLOCKWISE)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# Keypoint 검출 및 특징 Discriptor 계산
keypoint1, descriptor1 = sift.detectAndCompute(gray1, None)
keypoint2, descriptor2 = sift.detectAndCompute(gray2_rotated, None)

# 키포인트 매칭
matcher = cv2.BFMatcher()
matchers = matcher.match(descriptor1, descriptor2)
matchers = sorted(matchers, key=lambda x: x.distance)

for match in matchers[:10] :
    print(f"Distance : {match.distance}")
    print("Keypoint 1 : (x=%d, y=%d)" % (int(keypoint1[match.queryIdx].pt[0]),
                                         int(keypoint1[match.queryIdx].pt[1])
                                         ))
    print("Keypoint 2 : (x=%d, y=%d)" % (int(keypoint2[match.trainIdx].pt[0]),
                                         int(keypoint2[match.trainIdx].pt[1])
                                         ))

matched_image = cv2.drawMatches(image1, keypoint1, image2_rotated, keypoint2,
                                matchers[:10], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    

cv2.imshow("Matched_image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()