import cv2

# image load
image1 = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)

gray2_rotated = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

# image size 
image1 = cv2.resize(image1, (400,400))
image2 = cv2.resize(gray2_rotated, (400,400))

# ORB 객체 생성
orb = cv2.ORB_create()

# keypoints, descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# BRIFE 디스크립터 매칭
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matchers = matcher.match(descriptors1, descriptors2)

# 매칭 결과 정렬
matchers = sorted(matchers, key=lambda x:x.distance)

# 상위 N개의 매칭 결과 시각화
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                matchers[:10], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    

cv2.imshow("BRIEF Matching", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
