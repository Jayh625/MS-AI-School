import cv2 

def mode1(img1,img2) : 
    img1 = cv2.resize(img1, (500, 800))
    img2 = cv2.resize(img2, (500, 800))
    orb = cv2.ORB_create()
    keypoint01, descriptor01 = orb.detectAndCompute(img1, None)
    keypoint02, descriptor02 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor01, descriptor02)
    matches = sorted(matches, key=lambda x:x.distance)
    result = cv2.drawMatches(img1, keypoint01, img2, keypoint02, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("Matches", result)
    cv2.waitKey(0)
     
    num_matches = len(matches)
    print(num_matches) 
    num_good_matches = sum(1 for m in matches if m.distance < 50) # 적절한 거리 임계값 설정
    matching_percent = (num_good_matches / num_matches) * 100
    print("매칭 퍼센트 : %.2f%%" % matching_percent)
    cv2.destroyAllWindows()

img1 = cv2.imread('./data/cat1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/cat2.png', cv2.IMREAD_GRAYSCALE)
mode1(img1, img2)
