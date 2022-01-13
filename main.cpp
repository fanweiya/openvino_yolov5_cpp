#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "detector.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\lib
//C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\include
//C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\bin
int main(int argc, char *argv[]) {
	Detector* detector = new Detector;
	string xml_path = "best-sim.xml";
	detector->init(xml_path);

	Mat src = imread("test_1.jpg");
	Mat or_img = src.clone();
	int slice_number = 23;
	ofstream outfile("result_light_status.txt");
	//Ï¸°û±àºÅ,X×ø±ê,Y×ø±ê,Ö±¾¶
	outfile << "Ï¸°û±àºÅ,"<<"X×ø±ê,"<< "Y×ø±ê," << "Ö±¾¶" << endl;
	auto start = chrono::high_resolution_clock::now();
	vector<Detector::Object> detected_objects;
	detector->slice_process_frame(src, detected_objects);
	cout << "cell num: " << detected_objects.size() << endl;
	vector<int> IMB_area_vecter, IMB_diam_vecter;
	for (int i = 0; i < detected_objects.size(); i++) {
		if (detected_objects[i].name == "IMB") {
			IMB_area_vecter.push_back(detected_objects[i].rect.width*detected_objects[i].rect.height);
			IMB_diam_vecter.push_back(int((detected_objects[i].rect.width + detected_objects[i].rect.height) / 2));
		}
	}
	double IMB_area_means= round(std::accumulate(std::begin(IMB_area_vecter), std::end(IMB_area_vecter), 0.0)/ IMB_area_vecter.size());
	int IMB_diam_means = round(std::accumulate(std::begin(IMB_diam_vecter), std::end(IMB_diam_vecter), 0.0) / IMB_diam_vecter.size());
	vector<String> anysisresults;
	vector<vector<int>> cell_infos;
	for (int i = 0; i < detected_objects.size(); i++) {
		int xmin = detected_objects[i].rect.x;
		int ymin = detected_objects[i].rect.y;
		int width = detected_objects[i].rect.width;
		int height = detected_objects[i].rect.height;
		Rect rect(xmin, ymin, width, height);
		int cricle_x = int(xmin+width/2);
		int cricle_y = int(ymin+height/ 2);
		int diam = int((width + height) / 2);
		cv::Scalar color = Scalar(0, 0, 255);
		if (detected_objects[i].name == "IMB") {
			//color = Scalar(0,0, 255);
			anysisresults.push_back(to_string(cricle_x) + "," +to_string(cricle_y)+","+to_string(diam));
			vector<int> temp = { cricle_x ,cricle_y ,diam };
			cell_infos.push_back(temp);
		}
		if (detected_objects[i].name == "IMBA") {
			int slice_number = round((width * height)/ IMB_area_means);
			for (int j = 0; j < slice_number; j++) {
				anysisresults.push_back(to_string(cricle_x) + "," + to_string(cricle_y) + "," + to_string(IMB_diam_means));
				vector<int> temp = { cricle_x ,cricle_y ,IMB_diam_means };
				cell_infos.push_back(temp);
			}
		}
		//cv::rectangle(src, rect, color, 1, LINE_8, 0);
		cv::circle(src,Point(cricle_x,cricle_y),int(diam/2),color,1);
		//cout << "is: " << detected_objects[i].name << endl;
		/*
		putText(src, detected_objects[i].name+" "+to_string(detected_objects[i].prob),
			cv::Point(xmin, ymin - 10),
			cv::FONT_HERSHEY_SIMPLEX,
			0.5,
			color);
		*/
	}
	//Ð´ÐòºÅ
	for (int i = 0; i < anysisresults.size();i++) {
		outfile << to_string(i+1) + "," << anysisresults[i]<< endl;
	}
	outfile.close();
	
	//·Ö¸îÍ¼Æ¬Ð´½á¹û
	int split_height = round(or_img.rows/slice_number);
	for(int i=0; i < slice_number; i++) {
		imwrite("Chamber1_"+to_string(i+1)+"_BR_light.jpg", src(Rect(0, i*split_height,or_img.cols, split_height)));
		ofstream patch_outfile("Chamber1_" + to_string(i + 1) + "_BR_light_status.txt");
		patch_outfile << "Ï¸°û±àºÅ," << "X×ø±ê," << "Y×ø±ê," << "Ö±¾¶" << endl;
		int z = 1;
		for(auto cell:cell_infos) {
		if ((cell[1]>=(i*split_height))&&(((i + 1)*split_height)>= cell[1])){
			patch_outfile << to_string(z) + "," << to_string(cell[0]) + "," + to_string(cell[1]) + "," + to_string(cell[2]) << endl;
			z++;
		}
		}
		patch_outfile.close();
	}

	auto end = chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	cout << "use " << diff.count() << " s" << endl;
	imwrite("result_light.jpg", src);
	return 0;
}
