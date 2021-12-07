'''
Given a rotated square, the method is able to match the template, but during template preprocessing, some of the cropped templates are not
appropriately cropped. Some are cropped as single pixel values without detectable edges that will obviously match to the left corner of the
canvas regardless.
Requires troubleshooting into the template selection process. More tests must be conducted with other images to determine template matching
model's robustness.
After changes to template matching, CoM detection functions must be added.
'''

import numpy as np
import cv2
import imutils
from tempMatching import read_file, return_dim, rotate_template, preprocess, return_contourDims, plt_template, crop_template

# Sets parameters of template matching: rotation increment, padding of template, minimum and maximum scale values.
def set_params(rotation_increment=20, padding=0, minScale=0.1, maxScale=2.0):
    return (rotation_increment, padding, minScale, maxScale)

# Generates list of rotated templates given specified rotation increment.
def generate_templates(template, params):
    rotated_images = rotate_template(template, params[0])
    temp_list = []
    for img in rotated_images:
        cropped = crop_template(img, padding=params[1])
        processed = preprocess(cropped)
        temp_list.append(processed)
    return temp_list

# Returns value, location, scale ratio and index in temp_list of best match.
def detect_template(canvas, temp_list, params):
    found = None
    index = 0
    for template in temp_list:
        (tH, tW) = return_dim(template)

        for scale in np.linspace(params[2], params[3], 20)[::-1]:
            cResized = imutils.resize(canvas, width=int(canvas.shape[1]*scale))
            ratio = canvas.shape[1] / float(cResized.shape[1])

            if cResized.shape[0]<tH or cResized.shape[1]<tW:
                break

            cEdged = preprocess(cResized)
            result = cv2.matchTemplate(cEdged, template, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal>found[0]:
                found = (maxVal, maxLoc, ratio, index)
                print('Index: ', index)
                print('Scale Size: ', scale)
                print('maxLoc: ', found[1])
                print('maxVal: ', found[0], '\n')
        index+=1
    return found

# found = (maxVal, maxLoc, ratio, index)

# Draws detected template object onto canvas.
def draw_match(canvas, temp_list, found):
    maxVal, maxLoc, ratio, index = found
    (tH, tW) = return_dim(temp_list[index])
    (startX, startY) = (int(maxLoc[0]*ratio), int(maxLoc[1]*ratio))
    (endX, endY) = (int((maxLoc[0]+tW)*ratio), int((maxLoc[1]+tH)*ratio))
    cv2.rectangle(canvas, (startX, startY), (endX, endY), (0,0,255), 5)
    cv2.imshow('Detected Template Result', canvas)
    cv2.waitKey(0)

