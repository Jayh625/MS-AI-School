import cv2
import numpy as np

img = cv2.imread("imgs\\mango\\0000_mango.png")
print(img.shape)

cv2.imshow("",img)
cv2.waitKey()
cv2.destroyAllWindows()