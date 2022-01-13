#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Detector {
public:
	typedef struct {
		float prob;
		std::string name;
		cv::Rect rect;
	} Object;
	bool init(string);
	bool process_frame(Mat& inframe, vector<Object> &detected_objects);
	bool slice_process_frame(Mat& inframe, vector<Object> &detected_objects);
private:
	double sigmoid(double);
	bool parseYolov5(const Blob::Ptr &blob, float cof_threshold,
		std::vector<Rect> &o_rect, std::vector<float> &o_rect_cof,
		std::vector<int> &classId);
	vector<vector<int>> getboxs(int image_height, int image_width, int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio);
	cv::Mat letterBox(Mat src);
	cv::Rect detect2origin(const Rect &det_rect, float rate_to, int top, int left);
	std::vector<int> get_anchors(int net_grid);
	double ratio;
	int topPad, btmPad, leftPad, rightPad;

	ExecutableNetwork _network;
	OutputsDataMap _outputinfo;
	string _input_name;
	string _xml_path;
	double _conf_threshold = 0.1;
	double _nms_area_threshold = 0.1;
	int _slice_height = 1280;
	int _slice_width = 1280;
	double _overlap_height_ratio = 0.1;
	double _overlap_width_ratio = 0.1;
	string className[2] = { "IMB", "IMBA" };
	/***
	string className[80] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		 "hair drier", "toothbrush" };
	***/
};
#endif