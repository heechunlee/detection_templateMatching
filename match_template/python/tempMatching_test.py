import numpy as np
import cv2
import imutils
from tempMatching_functions import read_file, return_dim, rotate_template, preprocess, return_contourDims, plt_template, crop_template
from tempMatching import set_params, generate_templates, detect_template, draw_match


# Testing TM functions
test_canvas = cv2.imread('')
test_template = cv2.imread('')
#cv2.imshow('Canvas', test_canvas)
#cv2.imshow('Template', test_template)

test_params = set_params(1, 2, 0.1, 2.0)
temp_list = generate_templates(test_template, test_params)

result = detect_template(test_canvas, temp_list, test_params)
print(result)

draw_match(test_canvas, temp_list, result)

cv2.waitKey(0)
