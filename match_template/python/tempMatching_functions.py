import numpy as np
import cv2
import imutils

# Returns image object given image filepath
def read_file(filename):
    return cv2.imread(filename)

# Returns height, width dimensions of image.
def return_dim(img):
    return img.shape[:2]

# Returns list of images in array form after each increment of angle rotation.
def rotate_template(img, increment=20):
    img_list = []
    for angle in range(0, 360, increment):
        rotated_img = imutils.rotate_bound(img, angle)
        img_list.append(rotated_img)
    return img_list

# Performs pre-processing on image: color conversion to grayscale, Gaussian blur operation and canny edge detection.
def preprocess(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(grayscale, (5,5), 0)
    canny = cv2.Canny(gaussian, 50, 100)
    return canny

# Finds contours from given input image and returns its location (x, y) and width (w)/height (h).
def return_contourDims(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # largest_contour is list of contours.
    # Must sort through list of contours and index first element (ie. largest contour)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]
    x, y, w, h = cv2.boundingRect(largest_contour)
    return ((x, y), w, h)

def crop_template(img, padding=0):
    (coord, tW, tH) = return_contourDims(preprocess(img))
    template_cropped = img[coord[1]-padding:coord[1]+tH+padding, coord[0]-padding:coord[0]+tW+padding]
    return template_cropped

# Displays cropped template from original canvas given specified padding input.
def plt_template(img, padding=0):
    (coord , tW, tH) = return_contourDims(preprocess(img))
    canvas_dim = return_dim(img)
    cH, cW = canvas_dim[0], canvas_dim[1]
    template_cropped = None
    if tW+2*padding<=cW and tH+2*padding<=cH:
        template_cropped = img[coord[1]-padding:coord[1]+tH+padding, coord[0]-padding:coord[0]+tW+padding]
    cv2.imshow('Cropped Template, Padding={}'.format(padding), template_cropped)

