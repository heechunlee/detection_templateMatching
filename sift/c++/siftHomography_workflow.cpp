#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <tuple>
#include <corecrt_math_defines.h>

using namespace std;
using namespace cv; 
using namespace cv::xfeatures2d; 

// setQueryTrain
tuple<Mat, Mat> setQueryTrain(string query_path, string train_path) {
	Mat query = imread(query_path, 0);
	Mat train = imread(train_path, 0);
	tuple <Mat, Mat> image_set = make_tuple(query, train);
	return image_set; 
}

// findKeypoints
//Instead of two dictionaries referencing the query and train image, each keypoint and descriptor as values
//Here the two dictionaries refer to keypoint and descriptor sets, with query and train images values found for each. 
tuple< map<string, vector<KeyPoint>> , map<string, Mat> > findKeypoints(Mat query, Mat train){
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> kp_q, kp_t; 
	Mat des_q, des_t;

	detector->detectAndCompute(query, noArray(), kp_q, des_q);
	detector->detectAndCompute(train, noArray(), kp_t, des_t); 
	
	map<string, vector<KeyPoint>> kp_dict = { {"Query", kp_q}, {"Train", kp_t} };
	map<string, Mat> des_dict = { {"Query", des_q}, {"Train", des_t} };
	return make_tuple(kp_dict, des_dict);
}

// findMatches
vector<vector<DMatch>> findMatches(map<string, Mat> des_dict) {
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	Mat des_q = des_dict["Query"];
	Mat des_t = des_dict["Train"];

	vector<vector<DMatch>> knn_matches; 
	matcher->knnMatch(des_q, des_t, knn_matches, 2);
	return knn_matches; 
}

// findMatchesBF
vector<vector<DMatch>> findMatchesBF(map<string, Mat> des_dict) {
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	Mat des_q = des_dict["Query"];
	Mat des_t = des_dict["Train"];

	vector<vector<DMatch>> knn_matches;
	matcher->knnMatch(des_q, des_t, knn_matches, 2);
	return knn_matches;
}


// findGoodMatches
vector<DMatch> findGoodMatches(vector<vector<DMatch>> all_matches) {
	vector<DMatch> good_keypoint_matches; 
	const float threshold = 0.75f; 
	for (size_t i = 0; i < all_matches.size(); i++) {
		if (all_matches[i][0].distance < threshold * all_matches[i][1].distance) {
			good_keypoint_matches.push_back(all_matches[i][0]);
		}
	}
	return good_keypoint_matches; 
}

// computeHomography 
tuple <Mat, vector<char>> computeHomography(map<string, vector<KeyPoint>> kp_dict, vector<DMatch> good_matches, int min_match=10) {
	Mat homography; 
	vector<char> mask; 
	if (good_matches.size() >= min_match){
		cout << "Successfully identified enough good keypoint matches: " << good_matches.size() << "/" << min_match << "\n";
		vector<KeyPoint> kp_q = kp_dict["Query"];
		vector<KeyPoint> kp_t = kp_dict["Train"];
		vector<Point2f> query_pts; 
		vector<Point2f> train_pts; 

		for (size_t i = 0; i < good_matches.size(); i++) {
			query_pts.push_back(kp_q[good_matches[i].queryIdx].pt);
			train_pts.push_back(kp_t[good_matches[i].trainIdx].pt);
		}
		homography = findHomography(query_pts, train_pts, RANSAC, 5.0, mask);
	}
	else {
		cout << "Not enough good keypoint matches: " << good_matches.size() << "/" << min_match << "\n";
		homography = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
		mask.push_back(0);
	} 
	return make_tuple(homography, mask); 
}

// computeQueryCoords
vector<Point2f> computeQueryCoords(Mat homography, Mat query_img) {
	vector<Point2f> query_coords_source(4); 
	vector<Point2f> query_coords_dest(4);

	query_coords_source[0] = Point2f(0, 0);
	query_coords_source[1] = Point2f((float)query_img.cols, 0);
	query_coords_source[2] = Point2f((float)query_img.cols, (float)query_img.rows);
	query_coords_source[3] = Point2f(0, (float)query_img.rows);

	perspectiveTransform(query_coords_source, query_coords_dest, homography);
	return query_coords_dest; 
}

Point2f findCenter(vector<Point2f> query_coords_dest) {
	int x_val = 0; 
	int y_val = 0; 
	for (int i = 0; i < query_coords_dest.size(); i++) {
		x_val += query_coords_dest[i].x;
		y_val += query_coords_dest[i].y;
	}
	x_val = x_val/4; 
	y_val = y_val/4; 
	return Point2f(x_val, y_val); 
}

// findContourPoints 
vector<Point2f> findContourPoints(vector<Point2f> query_coords_dest) {
	vector<Point2f> contour_point_set; 
	for (int i = 0; i < query_coords_dest.size(); i++) {
		if (i <= 2) {
			float x_increment = (query_coords_dest[i + 1].x - query_coords_dest[i].x) / 50;
			float y_increment = (query_coords_dest[i + 1].y - query_coords_dest[i].y) / 50;
			for (int j = 0; j < 50; j++) {
				contour_point_set.push_back(Point2f(query_coords_dest[i].x + j*x_increment, query_coords_dest[i].y+j*y_increment));
			}
		}
		else {
			float x_increment = (query_coords_dest[0].x - query_coords_dest[i].x) / 50; 
			float y_increment = (query_coords_dest[0].y - query_coords_dest[i].y) / 50; 
			for (int j = 0; j < 50; j++) {
				contour_point_set.push_back(Point2f(query_coords_dest[i].x + j*x_increment, query_coords_dest[i].y + j*y_increment)); 
			}
		}
	}
	return contour_point_set; 
}

// computeEigen 
tuple <vector<Point2d>, vector<double>> computeEigen(vector<Point2f> contour_points) {
	Mat contour_points_mtx = Mat(200, 2, CV_64F);
	for (int i = 0; i < contour_points_mtx.rows; i++) {
		contour_points_mtx.at<double>(i, 0) = contour_points[i].x;
		contour_points_mtx.at<double>(i, 1) = contour_points[i].y; 
	}
	//cout << "M = " << endl << " " << contour_points_mtx << endl << endl;
	PCA pca_analysis(contour_points_mtx, Mat(), PCA::DATA_AS_ROW);

	vector<Point2d> eigenvecs(2);
	vector<double>  eigenvals(2); 
	for (int j = 0; j < 2; j++) {
		eigenvecs[j] = Point2d(pca_analysis.eigenvectors.at<double>(j, 0), pca_analysis.eigenvectors.at<double>(j, 1)); 
		eigenvals[j] = pca_analysis.eigenvalues.at<double>(j);
	}
	return make_tuple(eigenvecs, eigenvals);
}

// computeRotation
double computeRotation(vector<Point2f> contour_points) {
	tuple <vector<Point2d>, vector<double>> eigens = computeEigen(contour_points);
	vector<Point2d> eigenvecs = get<0>(eigens);
	vector<double> eigenvals = get<1>(eigens);
	double angle = atan2(eigenvecs[1].y, eigenvecs[1].x);
	angle = (angle * 180) / M_PI; 
	return angle; 
}

// computePCA 
tuple <Point2f, Point2f> computePCA(Point2f center, vector<Point2d> eigenvecs, vector<double> eigenvals) {
	float pc1_x = center.x + 0.01 * eigenvecs[0].x * eigenvals[0];
	float pc1_y = center.y + 0.01 * eigenvecs[0].y * eigenvals[0]; 

	float pc2_x = center.x + 0.01 * eigenvecs[1].x * eigenvals[1];
	float pc2_y = center.y + 0.01 * eigenvecs[1].y * eigenvals[1];

	return make_tuple(Point2f(pc1_x, pc1_y), Point2f(pc2_x, pc2_y));
}

Mat drawRotatedAxis(Mat train_img, Point2f center, tuple<Point2f, Point2f> pca_points, int scale = 1) {
	Mat placeholder = train_img.clone();

	double angle_pc1 = atan2(center.y - get<0>(pca_points).y, center.x - get<0>(pca_points).x);
	double hypotenuse_pc1 = sqrt(pow(center.y-get<0>(pca_points).y, 2) + pow(center.x-get<0>(pca_points).x, 2));
	double scaled_pc1_x = center.x - scale * hypotenuse_pc1 * cos(angle_pc1);
	double scaled_pc1_y = center.y - scale * hypotenuse_pc1 * sin(angle_pc1); 

	double angle_pc2 = atan2(center.y - get<1>(pca_points).y, center.x - get<1>(pca_points).x);
	double hypotenuse_pc2 = sqrt(pow(center.y - get<1>(pca_points).y, 2) + pow(center.x - get<1>(pca_points).x, 2));
	double scaled_pc2_x = center.x - scale * hypotenuse_pc2 * cos(angle_pc2);
	double scaled_pc2_y = center.y - scale * hypotenuse_pc2 * sin(angle_pc2);

	line(placeholder, center, Point2f(scaled_pc1_x, scaled_pc1_y), Scalar(255, 128, 0), 2, LINE_AA);
	line(placeholder, center, Point2f(scaled_pc2_x, scaled_pc2_y), Scalar(0, 128, 255), 2, LINE_AA);

	return placeholder; 
}

// drawCenter
Mat drawCenter(Mat train_img, Point2f center, bool gt_axis = true) {
	Mat placeholder = train_img.clone();
	if (gt_axis) {
		int width = placeholder.cols;
		int height = placeholder.rows; 

		line(placeholder, Point2f(center.x, 0), Point2f(center.x, height), Scalar(200, 200, 200), 2, LINE_AA);
		line(placeholder, Point2f(0, center.y), Point2f(width, center.y), Scalar(200, 200, 200), 2, LINE_AA);
		circle(placeholder, center, 3, Scalar(0, 0, 255), 2);
	}
	else {
		circle(placeholder, center, 3, Scalar(0, 0, 255), 2); 
	}
	return placeholder; 
}

// drawBoundingBox
Mat drawBoundingBox(Mat train_img, vector<Point2f> query_coords) {
	Mat placeholder = train_img.clone();
	vector<Point> query_coords_point;
	for (int i = 0; i < query_coords.size(); i++) {
		query_coords_point.push_back(query_coords.at(i));
	}

	polylines(placeholder, query_coords_point, true, Scalar(255,0,0), 2, LINE_AA);
	return placeholder; 
}


int main(void) {
	// Read image in GrayScale mode
	Mat query_img = imread("", IMREAD_UNCHANGED); //IMREAD_UNCHANGED, IMREAD_GRAYSCALE
	Mat train_img = imread("", IMREAD_UNCHANGED);

	tuple< map<string, vector<KeyPoint>>, map<string, Mat> > qt_keypoints = findKeypoints(query_img, train_img);
	cout << "Checkpoint #1 - Finding all keypoints. \n";

	map<string, Mat> placeholder_des_dict = get<1>(qt_keypoints);
	map<string, vector<KeyPoint>> placeholder_kp_dict = get<0>(qt_keypoints);
	vector<vector<DMatch>> total_matches = findMatchesBF(placeholder_des_dict);
	cout << "Checkpoint #2 - Finding all keyopint matches. \n";

	vector<DMatch> good_matches = findGoodMatches(total_matches);
	cout << "Checkpoint #3 - Selecting for good keypoint matches. \n";

	tuple<Mat, vector<char>> homography = computeHomography(placeholder_kp_dict, good_matches, 10);
	Mat homography_mtx = get<0>(homography);
	vector<char> homography_mask = get<1>(homography);
	cout << "Checkpoint #4 - Computing homography matrix. \n";

	vector<Point2f> dest_coords = computeQueryCoords(homography_mtx, query_img);
	for (int i = 0; i < dest_coords.size(); i++) {
		cout << "Coordinate #" << i << " " << dest_coords[i] << "\n";
	}
	cout << "Checkpoint #5 - Computing query image edge coordinates on train image. \n";

	Point2f center_coords = findCenter(dest_coords);
	cout << center_coords << "\n";
	cout << "Checkpoint #6 - Finding query image object center. \n";

	vector<Point2f> contour_coords = findContourPoints(dest_coords);
	//for (int i = 0; i < contour_coords.size(); i++) {
	//	cout << contour_coords[i] << "\n";
	//}
	cout << "Checkpoint #7 - Computing coordinates of contour points. \n";

	tuple<vector<Point2d>, vector<double>> eigens = computeEigen(contour_coords);
	cout << "Eigenvalue #1: " << get<1>(eigens)[0] << ", Eigenvalue #2: " << get<1>(eigens)[1] << "\n";
	cout << "M = " << endl << " " << get<0>(eigens) << endl << endl;
	cout << "X Val: " << get<0>(eigens)[0].x << ", Y Val: " << get<0>(eigens)[0].y << "\n";
	cout << "Checkpoint #8 - Computing eigenvectors and eigenvalues of contour points. \n";

	double rotation_angle = computeRotation(contour_coords);
	cout << "Angle of Rotation: " << rotation_angle << "\n";
	cout << "Checkpoint #9 - Computing angle of rotation in degrees. \n";

	tuple <Point2f, Point2f> principal_components = computePCA(center_coords, get<0>(eigens), get<1>(eigens));
	cout << "Principal Component #1: (" << get<0>(principal_components).x << ", " << get<0>(principal_components).y << ") \n";
	cout << "Principal Component #2: (" << get<1>(principal_components).x << ", " << get<1>(principal_components).y << ") \n";
	cout << "Checkpoint #10 - Computing principal components from query image object center. \n";

	Mat placeholder = drawRotatedAxis(train_img, center_coords, principal_components, 1);
	placeholder = drawCenter(placeholder, center_coords, true);
	//imshow("PCA Applied", placeholder);
	cout << "Checkpoint #11 - Labeling train image. \n";

	placeholder = drawBoundingBox(placeholder, dest_coords);
	imshow("Final Touches", placeholder);
	cout << "Checkpoint #12 - Drawing keypoint matches. \n";
	
	waitKey(0);
	return 0;
}