import cv2
import imutils
from siftHomography_pipeline import setQueryTrain, findKeypoints, findMatches, findGoodMatches, computeHomography, computeQueryCoords, drawQueryMatch, SH_pipeline

q_path = r'C:\Users\jasul\Desktop\heat_transfer_query.png'
t_path = r'C:\Users\jasul\Desktop\heat_transfer_train.jpg'

#q_path = r'C:\Users\jasul\Desktop\cocoa_butter_template.jpg'
#t_path = r'C:\Users\jasul\Desktop\aveeno_train.jpg'

SH_pipeline(q_path, t_path, 10)