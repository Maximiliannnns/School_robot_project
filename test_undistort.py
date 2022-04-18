import sys
import numpy as np
import cv2

img0 = 'photo00.jpg'
img = cv2.imread(img0)
#cv2.imshow('img_undist1', img)
cv2.waitKey(0)
dist_coef = np.array([[-1.51948828e+00,  1.95329695e+02, -1.04202010e-02, -3.03130271e-03,
  -7.54902603e+03]])
#camera_matrix = np.array([[640, 0, 480], [0, 640, 480], [0, 0, 1]])
camera_matrix = np.array([[3.32555825e+03, 0.00000000e+00, 9.49095102e+02],
 [0.00000000e+00, 3.20972678e+03, 5.61755191e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


img_undist = cv2.undistort(img, cameraMatrix=camera_matrix, distCoeffs=dist_coef)

cv2.imshow('img_undist', img_undist[450:950, 550:1300])

cv2.waitKey()
cv2.destroyAllWindows()