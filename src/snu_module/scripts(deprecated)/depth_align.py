import cv2
import numpy as np
import scipy.misc

IMG_PATH = './data/rgb_image/rgb_image__%09d.png'
DEPTH_NPZ = './data/aligned_depth_image/aligned_depth_ndarray.npz'
depth_npz = np.load(DEPTH_NPZ)
lt_ratio = (0.175, 0.200)
rb_ratio = (0.810, 0.825)


def align_depth_map(depth_map, lt_ratio, rb_ratio):
    h, w = depth_map.shape[:2]
    l, t = int(lt_ratio[0] * w), int(lt_ratio[1] * h)
    r, b = int(rb_ratio[0] * w), int(rb_ratio[1] * h)
    aligned_depth_map = cv2.resize(depth_map[t:b, l:r], dsize=(w, h))
    return aligned_depth_map


print(len(depth_npz.files))
for i in range(0, 2400, 2):
    print('[%09d]' % i, IMG_PATH % (i + 1), 'arr_%d' % i)
    img = cv2.imread(IMG_PATH % (i+1))
    depth_img = depth_npz['arr_%d' % i]
    depth_img = align_depth_map(depth_npz['arr_%d' % i], lt_ratio, rb_ratio)

    cv2.imshow('img', img)
    cv2.imshow('depth', depth_img)

    depth_img = np.clip(depth_img.astype(float), 0.0, 65000.0) / 260.0
    depth_img = np.expand_dims(depth_img, 2).astype(np.uint8)
    synth_img = (depth_img * 0.6 + img * 0.4).astype(np.uint8)
    
    cv2.imshow('synth', synth_img)
    cv2.waitKey(0)



