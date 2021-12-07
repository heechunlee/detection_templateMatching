#include "siftHomography_pipeline.h"

// Input: Template image (Mat), canvas image (Mat)
// Output: Four coordinates of template detected on canvas (vector<Point2f>) 
std::vector<cv::Point2f> findEdges(cv::Mat template_img, cv::Mat canvas_img) {

	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	std::vector<cv::KeyPoint> kp_temp, kp_canvas;
	cv::Mat des_temp, des_canvas;
	detector->detectAndCompute(template_img, cv::noArray(), kp_temp, des_temp);
	detector->detectAndCompute(canvas_img, cv::noArray(), kp_canvas, des_canvas);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> total_matches;
	matcher->knnMatch(des_temp, des_canvas, total_matches, 2);

	std::vector<cv::DMatch> good_matches; 
	const float threshold = 0.75f; 
	for (size_t i = 0; i < total_matches.size(); i++) {
		if (total_matches[i][0].distance < threshold * total_matches[i][1].distance) {
			good_matches.push_back(total_matches[i][0]);
		}
	}

	cv::Mat homography;
	if (good_matches.size() >= 10) {
		std::cout << "Successfully identified enough good keypoint matches: " << good_matches.size() << "/" << 10 << "\n";
		std::vector<cv::Point2f> temp_pts;
		std::vector<cv::Point2f> canvas_pts;

		for (size_t i = 0; i < good_matches.size(); i++) {
			temp_pts.push_back(kp_temp[good_matches[i].queryIdx].pt);
			canvas_pts.push_back(kp_canvas[good_matches[i].trainIdx].pt);
		}
		homography = cv::findHomography(temp_pts, canvas_pts, cv::RANSAC, 5.0);
	}
	else {
		std::cout << "Not enough good keypoint matches: " << good_matches.size() << "/" << 10 << "\n";
	}

	std::vector<cv::Point2f> edgePts_temp(4);
	std::vector<cv::Point2f> edgePts_canvas(4);
	edgePts_temp[0] = cv::Point2f(0, 0);
	edgePts_temp[1] = cv::Point2f((float)template_img.cols, 0);
	edgePts_temp[2] = cv::Point2f((float)template_img.cols, (float)template_img.rows);
	edgePts_temp[3] = cv::Point2f(0, (float)template_img.rows);
	perspectiveTransform(edgePts_temp, edgePts_canvas, homography);

	return edgePts_canvas;
}

// Input: Template image (Mat), canvas image (Mat)
// Output: Center coordinates of template detected on canvas (Point2f)
cv::Point2f findCenter(cv::Mat template_img, cv::Mat canvas_img) {
	std::vector<cv::Point2f> edgePts_canvas = findEdges(template_img, canvas_img);

	float x_val = 0;
	float y_val = 0;
	for (int i = 0; i < edgePts_canvas.size(); i++) {
		x_val += edgePts_canvas[i].x;
		y_val += edgePts_canvas[i].y;
	}
	x_val = x_val / 4;
	y_val = y_val / 4;
	return cv::Point2f(x_val, y_val);
}

// Input: Template image (Mat), canvas image (Mat) 
// Output: Rotation angle of template detected on canvas (double) 
double findRotation(cv::Mat template_img, cv::Mat canvas_img) {
	std::vector<cv::Point2f> edgePts_canvas = findEdges(template_img, canvas_img);

	std::vector<cv::Point2f> contour_pts;
	for (int i = 0; i < edgePts_canvas.size(); i++) {
		if (i <= 2) {
			float x_increment = (edgePts_canvas[i + 1].x - edgePts_canvas[i].x) / 50;
			float y_increment = (edgePts_canvas[i + 1].y - edgePts_canvas[i].y) / 50;
			for (int j = 0; j < 50; j++) {
				contour_pts.push_back(cv::Point2f(edgePts_canvas[i].x + j * x_increment, edgePts_canvas[i].y + j * y_increment));
			}
		}
		else {
			float x_increment = (edgePts_canvas[0].x - edgePts_canvas[i].x) / 50;
			float y_increment = (edgePts_canvas[0].y - edgePts_canvas[i].y) / 50;
			for (int j = 0; j < 50; j++) {
				contour_pts.push_back(cv::Point2f(edgePts_canvas[i].x + j * x_increment, edgePts_canvas[i].y + j * y_increment));
			}
		}
	}

	cv::Mat contour_mtx = cv::Mat(200, 2, CV_64F);
	for (int i = 0; i < contour_mtx.rows; i++) {
		contour_mtx.at<double>(i, 0) = contour_pts[i].x;
		contour_mtx.at<double>(i, 1) = contour_pts[i].y;
	}

	cv::PCA pca_analysis(contour_mtx, cv::Mat(), cv::PCA::DATA_AS_ROW);
	std::vector<cv::Point2d> eigenvecs(2);
	std::vector<double>  eigenvals(2);
	for (int j = 0; j < 2; j++) {
		eigenvecs[j] = cv::Point2d(pca_analysis.eigenvectors.at<double>(j, 0), pca_analysis.eigenvectors.at<double>(j, 1));
		eigenvals[j] = pca_analysis.eigenvalues.at<double>(j);
	}
	
	double angle = atan2(eigenvecs[0].y, eigenvecs[0].x);
	angle = (angle * 180) / M_PI;
	
	return angle; 
}

int main(void) {

}
