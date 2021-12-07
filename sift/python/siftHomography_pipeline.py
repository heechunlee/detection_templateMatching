"""
*** When calculating rotation angle, only rotation about the first principal component (initially ground truth x-axis) is considered. 
    Orientation of principal components are arbitrary as they represent eigenvectors that are subject to scalar multiplication by any 
    scale factor. 
    Thus, rotation of -45 degrees is equivalent to rotation by 135 dgrees. 

Troubleshooting: 
(1) Issues with using SIFT algorithm for detecting simple shapes with little texture gradient.
(2) Computed principal components have poor accuracy when homography matrix confers large distortions when matching query image object 
    to train image. 
(3) Finds principal components regardless of initial query image orientation. 
    Eg. Query image has longer height than width and has unchanged rotation in train image. 
        But since algorithm performs principal component analysis, longer height side is returned as first principal componenet and 
        returns 90 degrees of rotation compared to ground truth axis. 
    ==> Maybe the rotation occurs via the ground truth x-axis, meaning the angle of rotation is the angle required to achieve alignment
        between the objects of the query and train images.  
"""
import math
import numpy as np 
import cv2 

# Reads and returns query and train images given filepath as inputs. 
def setQueryTrain(query_path, train_path):
    query = cv2.imread(query_path)
    train = cv2.imread(train_path)
    return (query, train)
  
# Generates SIFT object to find keypoints and descriptors for query/train images. 
# Returns query_data, train_data dictionaries with keypoints and descriptors as values. 
def findKeypoints(query, train):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(query, None)
    kp_t, des_t = sift.detectAndCompute(train, None)
    query_data = dict(keypoints=kp_q, descriptors=des_q)
    train_data = dict(keypoints=kp_t, descriptors=des_t)
    return (query_data, train_data)

# Matches each keypoint in query image to two most similar corresponding keypoints in train image based on FLANN algorithm. 
# Returns keypoint_matches nested list consisting of two DMatch objects for each query image keypoint. 
def findMatches(query_data, train_data): 
    des_q = query_data['descriptors']
    des_t = train_data['descriptors']

    FLANN_INDEX_KDTREE = 1 
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    keypoint_matches = flann.knnMatch(des_q, des_t, k=2)
    return keypoint_matches

# Brute force keypoint matcher used as alternative to FLANN. 
def findMatchesBF(query_data, train_data):
    des_q = query_data['descriptors']
    des_t = train_data['descriptors']

    bf = cv2.BFMatcher()
    keypoint_matches = bf.knnMatch(des_q, des_t, k=2)
    return keypoint_matches

# For each two train image keypoint matches per query image keypoint, applies Lowe's ratio test to determine detected suitability. 
# Returns list of good keypoint matches. 
def findGoodMatches(matches):
    good_keypoint_matches = []
    for m, n in matches: 
        if m.distance < 0.75*n.distance: 
            good_keypoint_matches.append(m)
    return good_keypoint_matches

# Computes and returns 3x3 homography transformation matrix and binary mask after applying RANSAC to good keypoint matches. 
# Only computes if number of good keypoint matches is greater than specified input threshold amount. 
def computeHomography(query_data, train_data, good_matches, min_match=10):
    homography, mask = None, None
    if len(good_matches) >= min_match: 
        kp_q = query_data['keypoints']
        kp_t = train_data['keypoints']
        query_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        train_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        homography, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        #print(homography)
        mask = mask.ravel().tolist()
        #print(mask)
    else:
        print('Not enough good keypoint matches between query and train images: {}/{}'.format(len(good_matches), min_match))
    return (homography, mask)

# Performs perspective transformation of query image based on homography matrix. 
# Returns edge coordinates of detected query image on train image (Negative values possible). 
def computeQueryCoords(homography, query):
    h,w,d = query.shape
    query_coords_source = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    query_coords_dest = cv2.perspectiveTransform(query_coords_source, homography)
    return query_coords_dest

# Returns center of mass of detected query image object given its four edge coordinates. 
def findCenter(query_coords_dest):
    coords = query_coords_dest.squeeze()
    center_of_mass = np.mean(coords, axis=0)
    return center_of_mass

# Returns coordinates of contour points along detected query images perimeter given its edge coordinates as inputs. 
# Used to perform principal component analysis and determine rotation of object. 
def findContourPoints(query_coords_dest):
    contour_point_set = None 
    coords = query_coords_dest.squeeze().tolist()
    for m,n in [(0,1), (1,2), (2,3), (3,0)]:
        first_point = coords[m]
        second_point = coords[n]
        x_points = np.linspace(first_point[0], second_point[0], 50)
        y_points = np.linspace(first_point[1], second_point[1], 50)
        coord_set = np.column_stack((x_points, y_points))
        if contour_point_set is None: 
            contour_point_set = coord_set
        else: 
            contour_point_set = np.concatenate((contour_point_set, coord_set), axis=0)
    return contour_point_set

# Returns corresponding eigenvectors and eigenvalues given contour points as inputs. 
def computeEigen(contour_points):
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean)
    return (eigenvectors, eigenvalues)

# Computes angle of rotation between eigenvectors. 
# Returns angle of rotation of object in degrees against ground truth x, y axis. 
# Ground truth without rotation has x-axis facing east, y-axis facing south (but second principcal component subject to change). 
# Rotation in clockwise direction returns positive angle, rotation in counterclockwise direction returns negative angle. 
def computeRotation(contour_points):
    edge1, edge2, edge3, edge4 = contour_points[0], contour_points[50], contour_points[100], contour_points[150]
    center = np.mean([edge1, edge2, edge3, edge4], axis=0)
    center_y = center[1]
    top_midpoint_y = contour_points[175][1]

    eigenvectors, eigenvalues = computeEigen(contour_points)
    pca_p1_y = [center[0] + 0.01*eigenvectors[0,0]*eigenvalues[0,0], center[1] + 0.01*eigenvectors[0,1]*eigenvalues[0,0]][1]
    #print('HERE: ', center_y, top_midpoint_y, pca_p1_y)

    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle = (angle*180)/math.pi

    if(top_midpoint_y<center_y and pca_p1_y>center_y):
        angle = angle - 90
    elif(top_midpoint_y>center_y and pca_p1_y>center_y):
        angle = angle + 90 
    elif(top_midpoint_y<center_y and pca_p1_y<center_y):
        angle = angle + 90
    elif(top_midpoint_y>center_y and pca_p1_y<center_y):
        angle = angle + 270

    return angle

# Performs principal component analysis given center of mass, eigenvectors and eigenvalues as inputs. 
# Returns two new points corresponding to a set of rotated principal components when centered around original center of mass. 
def computePCA(center, eigenvectors, eigenvalues):
    pca_p1 = [center[0] + 0.01*eigenvectors[0,0]*eigenvalues[0,0], center[1] + 0.01*eigenvectors[0,1]*eigenvalues[0,0]]
    pca_p2 = [center[0] - 0.01*eigenvectors[1,0]*eigenvalues[1,0], center[1] - 0.01*eigenvectors[1,1]*eigenvalues[1,0]]
    return (pca_p1, pca_p2)

# Draws rotated axes given newly computed principal components, center of mass and scale factor. 
# Length of new axis dependent on the amount of variation along given axis detected contour points. 
# Blue corresponds to first principal component, orange corresponds to second principal component. 
def drawRotatedAxis(train, center, pca_points, scale=1):
    placeholder = train.copy()
    order = 0
    for point in pca_points:
        angle = math.atan2(center[1] - point[1], center[0] - point[0])
        hypotenuse = math.sqrt((center[1] - point[1])**2 + (center[0] - point[0])**2)

        scaled_x = center[0] - scale*hypotenuse*math.cos(angle)
        scaled_y = center[1] - scale*hypotenuse*math.sin(angle)
        labeled_result = cv2.line(placeholder, 
                                pt1=np.int32([center[0],center[1]]), 
                                pt2=np.int32([scaled_x, scaled_y]), 
                                color=(255,128,0) if order==0 else (0,128,255), 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
        order += 1
    return labeled_result

# Draws center of mass of matched query image on train image. 
# If passed as argument, draws ground truth axis centered at object center of mass. 
def drawCenter(train, center, gt_axis=True):
    if gt_axis == True: 
        width = train.shape[1]
        height = train.shape[0]
        labeled_result = cv2.line(train.copy(), 
                                pt1=[np.int32(center[0]), 0],
                                pt2=[np.int32(center[0]), height],
                                color=(200,200,200), 
                                thickness=2, 
                                lineType=cv2.LINE_AA)

        labeled_result = cv2.line(labeled_result, 
                                pt1=[0, np.int32(center[1])],
                                pt2=[width, np.int32(center[1])],
                                color=(200,200,200), 
                                thickness=2, 
                                lineType=cv2.LINE_AA)

        labeled_result = cv2.circle(labeled_result, 
                            center=np.int32(center), 
                            radius=3, 
                            color=(0,0,255), 
                            thickness=2)
    else: 
        labeled_result = cv2.circle(train.copy(), 
                            center=np.int32(center), 
                            radius=3, 
                            color=(0,0,255), 
                            thickness=2)
    return labeled_result

# Draws final result of query image matching: 
# (1) Draws quadrilateral representing detected query image on train image given computed edge coordinates.  
# (2) Draws corresponding good keypoint matches between query image and train image. 
def drawQueryMatch(query, query_data, train, train_data, good_matches, mask, query_coords):
    placeholder = train.copy()
    train_matched = cv2.polylines(placeholder, 
                                pts=[np.int32(query_coords)], 
                                isClosed=True, 
                                color=255, 
                                thickness=2, 
                                lineType=cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = mask, #Only draws inliers after RANSAC. 
                    flags = 2)

    matched_result = cv2.drawMatches(query, 
                                    query_data['keypoints'],
                                    train_matched,
                                    train_data['keypoints'],
                                    good_matches, 
                                    None, 
                                    **draw_params)
    return matched_result

# Pipeline function that runs entire workflow. 
# Query image filepath, train image filepath, good match threshold and T/F of ground truth axis passed as inputs. 
def SH_pipeline(query_path, train_path, match_threshold, gt_axis=True): 
    query_img, train_img = setQueryTrain(query_path, train_path)
    query_dict, train_dict = findKeypoints(query_img, train_img)
    all_matches = findMatches(query_dict, train_dict)
    good_matches = findGoodMatches(all_matches)
    homography, mask = computeHomography(query_dict, train_dict, good_matches, match_threshold)
    query_edge_coords = computeQueryCoords(homography, query_img)
    center_of_mass = findCenter(query_edge_coords)

    contour_points = findContourPoints(query_edge_coords)
    eigenvectors, eigenvalues = computeEigen(contour_points)
    pca_points = computePCA(center_of_mass, eigenvectors, eigenvalues)
    print("Principal Component 1: ", pca_points[0])
    rotation_angle = computeRotation(contour_points)

    print('Matched Query Image Coordinates: \n', query_edge_coords.squeeze())
    print('Center of Mass: ', center_of_mass)
    print('Angle of Rotation: ', rotation_angle)
    
    img = drawCenter(train_img, center_of_mass, gt_axis)
    img = drawRotatedAxis(img, center_of_mass, pca_points)
    
    img = drawQueryMatch(query_img, query_dict, img, train_dict, good_matches, mask, query_edge_coords)
    cv2.imshow('Final Result', img)
    cv2.waitKey(0)