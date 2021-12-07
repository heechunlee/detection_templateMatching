#ifndef siftHomography_H__
#define siftHomography_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <vector>
#include <math.h>
#include <corecrt_math_defines.h>

class siftHomography_pipeline {
public:
    // Finds edge coordinates of template on canvas image. 
    // Input: Template image (Mat), canvas image (Mat)
    // Output: Four coordinates of template detected on canvas (vector<Point2f>) 
    std::vector<cv::Point2f> findEdges(cv::Mat temp, cv::Mat canvas);

    // Finds center of template. 
    // Input: Template image (Mat), canvas image (Mat)
    // Output: Center coordinates of template detected on canvas (Point2f)
    cv::Point2f findCenter(cv::Mat temp, cv::Mat canvas);

    // Finds rotation of template. 
    // Input: Template image (Mat), canvas image (Mat) 
    // Output: Rotation angle of template detected on canvas (double) 
    double findRotation(cv::Mat temp, cv::Mat canvas);

};


#endif  //siftHomography_H__
