import os
import cv2
import numpy as np
import imageio

base_path = "/home/kyle/USR_SNU_MODULE/SNU_Integrated_v2/src/snu_module/DATA/US_2019_POHANG/01/MV12_0604_d_fin/rgbdepth/"

color_img_path = os.path.join(base_path, "color/")
disparity_img_path = os.path.join(base_path, "depth/")
thermal_img_path = os.path.join(os.path.dirname(os.path.dirname(base_path)), "thermal/")

color_img_name_list = sorted(os.listdir(color_img_path))
disparity_img_name_list = sorted(os.listdir(disparity_img_path))
thermal_img_name_list = sorted(os.listdir(thermal_img_path))

color_img_list = [color_img_path+filename for filename in color_img_name_list]
disparity_img_list = [disparity_img_path+filename for filename in disparity_img_name_list]
thermal_img_list = [thermal_img_path+filename for filename in thermal_img_name_list]

# Test for first frame Image
rgb_img_imageio = imageio.imread(color_img_list[0])
rgb_img = cv2.cvtColor(cv2.imread(color_img_list[0]), cv2.COLOR_BGR2RGB)
rgb_any_img = cv2.imread(color_img_list[0], cv2.IMREAD_ANYCOLOR)
disparity_img8 = cv2.imread(disparity_img_list[0], 0)
thermal_img = cv2.imread(thermal_img_list[0], 0)

disparity_img16 = cv2.imread(disparity_img_list[0], cv2.IMREAD_ANYDEPTH)
disparity_img16_2 = cv2.imread(disparity_img_list[0], cv2.IMREAD_UNCHANGED)

thermal_mono8 = cv2.imread(thermal_img_list[0], cv2.IMREAD_GRAYSCALE)

# imshow
cv2.imshow("RGB Image", rgb_img)
# cv2.imshow("Disparity Image", disparity_img)
# cv2.imshow("Thermal Image", thermal_img)

cv2.waitKey(1)
