"""Feature extraction functions (Gabor, texture stats)."""

import numpy as np
import cv2
from skimage.filters import gabor




def build_gabor_kernels(ksize=31, frequencies=[0.1,0.2,0.3], thetas=[0, np.pi/4, np.pi/2]):
kernels = []
for freq in frequencies:
for theta in thetas:
kernels.append((freq, theta))
return kernels


def extract_gabor_features_gray(img_gray, kernels=None):
if kernels is None:
kernels = build_gabor_kernels()
feats = []
for (freq, theta) in kernels:
real, imag = gabor(img_gray, frequency=freq, theta=theta)
feats.append(real.mean())
feats.append(real.std())
return np.array(feats)




def extract_basic_stats(img):
# img: HxW or HxWxC
arr = img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
return np.array([arr.mean(), arr.std(), arr.min(), arr.max()])




if __name__ == '__main__':
print('Feature extraction utilities loaded')
