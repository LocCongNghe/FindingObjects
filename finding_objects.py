import cv2
import numpy as np
import os

img = cv2.imread("image.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
darkened_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

template_dir = "objects"
templates = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith('.png')]

threshold = 0.6

for template_path in templates:
    template = cv2.imread(template_path, 0) 
    best_match_val = -1
    best_match_loc = None
    best_match_scale = None
    best_w, best_h = None, None

    for scale in np.linspace(0.5, 1.5, 20):
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        w, h = scaled_template.shape[::-1]  

        res = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_match_scale = scale
            best_w, best_h = w, h

    if best_match_val >= threshold:
        top_left = best_match_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)

        roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        darkened_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

        cv2.rectangle(darkened_img, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow('Detected Objects', darkened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg', darkened_img)