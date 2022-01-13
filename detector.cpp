#include "detector.h"

#define IMAGE_LEN 960
#define IMAGE_LEN_F 960.0f
bool Detector::parseYolov5(const Blob::Ptr &blob, float cof_threshold,
	std::vector<Rect> &o_rect, std::vector<float> &o_rect_cof,
	std::vector<int> &classId)
{
	//如果输入图像大小为[1,3,640,640],那么三个输出的尺寸为3*80*80*(4+1)*80 | 40*40 | 20*20
	//利用blob->getTensorDesc().getDims()获取输出头的参数，包括w,h,c,batch_size,class_num
	const int net_grid_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);    // 80/40/20
	const int net_grid_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);    // 80/40/20
	const int batch_size = static_cast<int>(blob->getTensorDesc().getDims()[0]);    // batch_size
	const int anchor_num = static_cast<int>(blob->getTensorDesc().getDims()[1]);    // anchor_num,3
	// item_size为类别个数，对于COCO而言，输出为80
	const int item_size = static_cast<int>(blob->getTensorDesc().getDims()[4]);     // (4+1+class_num)
	//cout << "anchor scale: " << net_grid_h << endl;
	//cout << "item_size: " << item_size << endl;
	//根据输出的尺度，利用get_anchors方法获得相应的锚框
	std::vector<int> anchors = get_anchors(net_grid_h);
	//输出分类内存分配
	InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
	const float *output_blob = blobMapped.as<float *>();
	int net_grid = net_grid_h;
	std::size_t gi = net_grid * item_size;
	std::size_t ggi = net_grid * gi;
	std::size_t anchor_n = anchor_num;
	//输出解析与预测框解码部分
	for (int n = 0; n < anchor_num; ++n)
		for (int i = 0; i < net_grid; ++i)
			for (int j = 0; j < net_grid; ++j) {
				//获取输出置信度
				double box_prob = sigmoid(output_blob[n*ggi + i * gi + j * item_size + 4]);

				if (box_prob < cof_threshold) continue;

				double obj_prob = sigmoid(output_blob[n*ggi + i * gi + j * item_size + 5]);
				// 获取输出目标框
				//注意此处输出为中心点坐标,需要转化为角点坐标
				double x = output_blob[n*ggi + i * gi + j * item_size + 0];
				double y = output_blob[n*ggi + i * gi + j * item_size + 1];
				double w = output_blob[n*ggi + i * gi + j * item_size + 2];
				double h = output_blob[n*ggi + i * gi + j * item_size + 3];
				//获取输出类别索引
				double max_prob = 0;
				int idx = 0;
				for (int t = 5; t < item_size; ++t) {
					double tp = sigmoid(output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + t]);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t - 5;
					}
				}
				float cof = box_prob * max_prob;
				//对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
				if (cof < cof_threshold)  continue;
				//预测框解码
				x = (sigmoid(x) * 2 - 0.5 + j)* IMAGE_LEN_F / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)* IMAGE_LEN_F / net_grid;
				w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
				h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

				double r_x = x - w / 2;
				double r_y = y - h / 2;
				cv::Rect rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));
				//将结果保存在vector中
				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
				classId.push_back(idx);
			}
	if (o_rect.size() == 0) return false;
	else return true;
}

bool Detector::init(std::string xml_path) {
	try {
		_xml_path = xml_path;
		// 加载推理引擎Core，该引擎需要从当前路径加载plugins.xml文件
		InferenceEngine::Core ie;
		//查询支持硬件设备
		std::vector<std::string> availableDev = ie.GetAvailableDevices();
		for (int i = 0; i < availableDev.size(); i++) {
			cout << "supported device name: " << availableDev[i].c_str() << endl;
		}
		//从IR加载检测模型
		InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(_xml_path);
		cnnNetwork.setBatchSize(1);
		// 网络输入头参数设置
		InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
		InferenceEngine::InputInfo::Ptr input = inputInfo.begin()->second;
		_input_name = inputInfo.begin()->first;//输入头的名称
		input->setPrecision(InferenceEngine::Precision::FP32);
		input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
		InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
		InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
		cnnNetwork.reshape(inputShapes);
		//网络输出设置，YOLOV5包含三个输出
		_outputinfo = InferenceEngine::OutputsDataMap(cnnNetwork.getOutputsInfo());
		for (auto& output : _outputinfo) {
			//cout << output.first << endl;
			output.second->setPrecision(InferenceEngine::Precision::FP32);
		}
		//获取可执行网络
		_network = ie.LoadNetwork(cnnNetwork, "CPU");
		return true;
	}
	catch (std::exception & e) {
		return false;
	}
}

bool Detector::process_frame(Mat &curFrame, std::vector<Object> &detectedObj)
{
	cv::Mat resize_img = letterBox(curFrame);
	cout << "input image_size: " << resize_img.size() << endl;
	std::size_t img_size = IMAGE_LEN * IMAGE_LEN;


	//建立推理请求
	InferenceEngine::InferRequest infer_request = _network.CreateInferRequest();
	InferenceEngine::Blob::Ptr frameBlob = infer_request.GetBlob(_input_name);
	InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
	float* blob_data = blobMapped.as<float*>();
	//nchw
	//将输入图像关联到Blob
	for (std::size_t row = 0; row < IMAGE_LEN; row++) {
		for (std::size_t col = 0; col < IMAGE_LEN; col++) {
			for (std::size_t ch = 0; ch < 3; ch++) {
				//遍历输入图像的每个像素保存到blob_data，目前不理解blob_data的索引值
				blob_data[img_size*ch + row * IMAGE_LEN + col] = float(resize_img.at<cv::Vec3b>(row, col)[ch] / 255.0f);
			}
		}
	}
	//执行推理
	infer_request.Infer();

	//获取各层结果,保存到origin_rect,origin_rect_cof，classId
	std::vector<Rect> origin_rect;
	std::vector<float> origin_rect_cof;
	std::vector<int> classId;

	//大规模计算之前先收集指针
	std::vector<InferenceEngine::Blob::Ptr> blobs;
	//blobs保存推理结果，用于输入到parseYolov5方法进行解析
	for (auto &output : _outputinfo) {
		auto output_name = output.first;
		InferenceEngine::Blob::Ptr blob = infer_request.GetBlob(output_name);
		blobs.push_back(blob);
	}
	//cout << "blobs.size: " << blobs.size() << endl;
	//blob的大小为3
	for (int i = 0; i < 3; i++) {
		float th = 0.5;
		//TODO:根据网格大小使用不同阈值，可自己调配
		if (i == 0) { th = 0.55; }  //小目标严格要求
		if (i == 1) { th = 0.45; }  //大目标放宽要求
		if (i == 2) { th = 0.40; }

		//用于保存解析结果的临时vector 
		std::vector<cv::Rect> origin_rect_temp;
		std::vector<float> origin_rect_cof_temp;
		//解析blobs
		parseYolov5(blobs[i], th, origin_rect_temp, origin_rect_cof_temp, classId);
		origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
		origin_rect_cof.insert(origin_rect_cof.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());
	}
	//后处理操作获得最终结果
	std::vector<int> final_id;
	//TODO:此处的阈值需要调整
	//调用opencv dnn模块中的NMSBoxes进行非极大抑制处理
	cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, _conf_threshold, _nms_area_threshold, final_id);
	//根据final_id获取最终结果
	for (int i = 0; i < final_id.size(); ++i) {
		cv::Rect resize_rect = origin_rect[final_id[i]];
		//调用detect2origin方法将框映射到原图
		cv::Rect rawrect = detect2origin(resize_rect, ratio, topPad, leftPad);
		//结果以数据结构Object保存到vector
		//cout << "is: " << final_id[i] << endl;
		cout << "is: " << classId[final_id[i]] << endl;
		detectedObj.push_back(Object{
			origin_rect_cof[final_id[i]],
			className[classId[final_id[i]]],rawrect,
			});
	}
	return true;
}

bool Detector::slice_process_frame(Mat &curFrame, std::vector<Object> &detectedObj)
{
	//获取最后结果,保存到summary_rect,summary_rect_cof，summary_classId
	std::vector<Rect> summary_rect;
	std::vector<float> summary_rect_cof;
	std::vector<int> summary_classId;
	vector<vector<int>> slice_bboxes = getboxs(curFrame.rows,curFrame.cols, _slice_height, _slice_width, _overlap_height_ratio, _overlap_width_ratio);
	cout << "input split number: " << slice_bboxes.size() << endl;
	for (auto slice_bbox : slice_bboxes) {
		Mat image_pil_slice;
		curFrame(Rect(slice_bbox[0],slice_bbox[1],slice_bbox[2]-slice_bbox[0],slice_bbox[3]-slice_bbox[1])).copyTo(image_pil_slice);
		cv::Mat resize_img = letterBox(image_pil_slice);
		//cout << "input image_size: " << resize_img.size() << endl;
		std::size_t img_size = IMAGE_LEN * IMAGE_LEN;

		//建立推理请求
		InferenceEngine::InferRequest infer_request = _network.CreateInferRequest();
		InferenceEngine::Blob::Ptr frameBlob = infer_request.GetBlob(_input_name);
		InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
		float* blob_data = blobMapped.as<float*>();
		//nchw
		//将输入图像关联到Blob
		for (std::size_t row = 0; row < IMAGE_LEN; row++) {
			for (std::size_t col = 0; col < IMAGE_LEN; col++) {
				for (std::size_t ch = 0; ch < 3; ch++) {
					//遍历输入图像的每个像素保存到blob_data，目前不理解blob_data的索引值
					blob_data[img_size*ch + row * IMAGE_LEN + col] = float(resize_img.at<cv::Vec3b>(row, col)[ch] / 255.0f);
				}
			}
		}
		//执行推理
		infer_request.Infer();

		//获取各层结果,保存到origin_rect,origin_rect_cof，classId
		std::vector<Rect> origin_rect;
		std::vector<float> origin_rect_cof;
		std::vector<int> classId;

		//大规模计算之前先收集指针
		std::vector<InferenceEngine::Blob::Ptr> blobs;
		//blobs保存推理结果，用于输入到parseYolov5方法进行解析
		for (auto &output : _outputinfo) {
			auto output_name = output.first;
			InferenceEngine::Blob::Ptr blob = infer_request.GetBlob(output_name);
			blobs.push_back(blob);
		}
		//cout << "blobs.size: " << blobs.size() << endl;
		//blob的大小为3
		for (int i = 0; i < 3; i++) {
			float th = 0.5;
			//TODO:根据网格大小使用不同阈值，可自己调配
			if (i == 0) { th = 0.55; }  //小目标严格要求
			if (i == 1) { th = 0.45; }  //大目标放宽要求
			if (i == 2) { th = 0.40; }

			//用于保存解析结果的临时vector 
			std::vector<cv::Rect> origin_rect_temp;
			std::vector<float> origin_rect_cof_temp;
			std::vector<float> classId_temp;
			//解析blobs
			parseYolov5(blobs[i], th, origin_rect_temp, origin_rect_cof_temp, classId);
			origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
			origin_rect_cof.insert(origin_rect_cof.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());
			classId.insert(classId.end(), classId_temp.begin(), classId_temp.end());
		}
		//后处理操作获得最终结果
		std::vector<int> final_id;
		//TODO:此处的阈值需要调整
		//调用opencv dnn模块中的NMSBoxes进行非极大抑制处理
		cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, _conf_threshold, _nms_area_threshold, final_id);
		//cout << "origin_rect: " << origin_rect.size() << " _conf_threshold: " << _conf_threshold << " _nms_area_threshold: " << _nms_area_threshold << " final_id: " << final_id.size() << endl;
		//根据final_id获取最终结果
		for (int i = 0; i < final_id.size(); ++i) {
			cv::Rect resize_rect = origin_rect[final_id[i]];
			//调用detect2origin方法将框映射到原图
			cv::Rect raw_rect = detect2origin(resize_rect, ratio, topPad, leftPad);
			cv::Rect os_rect = Rect(raw_rect.x + slice_bbox[0], raw_rect.y + slice_bbox[1], raw_rect.width, raw_rect.height);
			summary_rect.push_back(os_rect);
			summary_rect_cof.push_back(origin_rect_cof[final_id[i]]);
			summary_classId.push_back(classId[final_id[i]]);
			//结果以数据结构Object保存到vector
			//cout << "is: " << final_id[i] << endl;
		}
	}
	//最后nms操作
	std::vector<int> end_id;
	cv::dnn::NMSBoxes(summary_rect, summary_rect_cof, _conf_threshold, _nms_area_threshold, end_id);
	//cout << "summary_rect: " << summary_rect.size() << " _conf_threshold: " << _conf_threshold << " _nms_area_threshold: " << _nms_area_threshold << " end_id: " << end_id.size() << endl;
	for (int i = 0; i < end_id.size(); ++i) {
		//结果以数据结构Object保存到vector
		//cout << "is: " << classId[end_id[i]] << endl;
		detectedObj.push_back(Object{
			summary_rect_cof[end_id[i]],
			className[summary_classId[end_id[i]]],summary_rect[end_id[i]],
			});
	}
	return true;
}
vector<vector<int>> Detector::getboxs(int image_height,int image_width,int slice_height,int slice_width,float overlap_height_ratio, float overlap_width_ratio) {
	vector<vector<int>> slice_bboxes;
	int y_max = 0;
	int y_min = 0;
	int y_overlap = std::round(overlap_height_ratio * slice_height);
	int x_overlap = std::round(overlap_width_ratio * slice_width);
	while (y_max < image_height)
	{
		int x_min = 0;
		int x_max = 0;
		y_max = y_min + slice_height;
		while (x_max < image_width)
		{
			x_max = x_min + slice_width;
			if (y_max > image_height or x_max > image_width) {
				int xmax = std::min(image_width, x_max);
				int ymax = std::min(image_height, y_max);
				int xmin = std::max(0, xmax - slice_width);
				int ymin = std::max(0, ymax - slice_height);
				vector<int> temp = { xmin, ymin, xmax, ymax };
				slice_bboxes.push_back(temp);
			}
			else {
				vector<int> temp = { x_min, y_min, x_max, y_max };
				slice_bboxes.push_back(temp);
			}
			x_min = x_max - x_overlap;
		}
		y_min = y_max - y_overlap;
	}
	return slice_bboxes;
}

double Detector::sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

const int anchorBig = IMAGE_LEN / 8;            //8倍下采样
const int anchorMid = IMAGE_LEN / 16;           //16倍下采样
const int anchorSml = IMAGE_LEN / 32;           //32倍下采样
 /*yolov5s.yaml内容
 anchors:
   - [10,13,16,30,33,23]        # P3/8
   - [30,61,62,45,59,119]       # P4/16
   - [116,90,156,198,373,326]   # P5/32
   const int aBig[6] = { 2,2,3,3,3,3 };
const int aMid[6] = { 4,5,5,6,7,9 };
const int aSml[6] = { 13,13,25,26,48,113 };
 */
const int aBig[6] = { 2,2,3,3,3,3 };
const int aMid[6] = { 4,5,8,6,7,9 };
const int aSml[6] = { 10,13,25,26,48,113 };
//获取锚框
std::vector<int> Detector::get_anchors(int net_grid)
{
	std::vector<int> anchors(6);
	if (net_grid == anchorBig) { anchors.insert(anchors.begin(), aBig, aBig + 6); }
	else if (net_grid == anchorMid) { anchors.insert(anchors.begin(), aMid, aMid + 6); }
	else if (net_grid == anchorSml) { anchors.insert(anchors.begin(), aSml, aSml + 6); }
	return anchors;
}
//图像缩放与填充
cv::Mat Detector::letterBox(Mat src)
{
	if (src.empty()) { std::cout << "input image invalid" << endl;  return cv::Mat(); }
	//以下为带边框图像生成
	int in_w = src.cols;
	int in_h = src.rows;
	int tar_w = IMAGE_LEN;
	int tar_h = IMAGE_LEN;
	//哪个缩放比例小选哪个
	ratio = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
	int inside_w = std::round(in_w * ratio);
	int inside_h = std::round(in_h * ratio);
	int pad_w = tar_w - inside_w;
	int pad_h = tar_h - inside_h;
	//内层图像resize
	cv::Mat resize_img;
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));  //最小的Resize
	//cout << "input image_size: " << resize_img.size() << endl;
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	pad_w = pad_w / 2;
	pad_h = pad_h / 2;
	//cout << "padding: " << pad_w << " " << pad_h << endl;
	//外层边框填充灰色
	topPad = int(std::round(pad_h - 0.1));
	btmPad = int(std::round(pad_h + 0.1));
	leftPad = int(std::round(pad_w - 0.1));
	rightPad = int(std::round(pad_w + 0.1));
	//cout << "padding: " << topPad<<" "<<btmPad<< " " << leftPad<< " " << rightPad << endl;
	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	//cout << "input image_size: " << resize_img.size() << endl;
	//if (resize_img.cols!= tar_w) {
		//cv::resize(resize_img, resize_img, cv::Size(IMAGE_LEN, IMAGE_LEN));  //输出标准化
	//}
	//if (resize_img.rows!= tar_h) {
		//cv::resize(resize_img, resize_img, cv::Size(IMAGE_LEN, IMAGE_LEN));  //输出标准化
	//}
	return resize_img;
}
//还原
//从detect得到的xywh转换回到原图xywh
cv::Rect Detector::detect2origin(const Rect &det_rect, float rate_to, int top, int left)
{
	//detect坐标转换到内部纯图坐标
	int inside_x = det_rect.x - left;
	int inside_y = det_rect.y - top;
	int ox = std::round(float(inside_x) / rate_to);
	int oy = std::round(float(inside_y) / rate_to);
	int ow = std::round(float(det_rect.width) / rate_to);
	int oh = std::round(float(det_rect.height) / rate_to);

	cv::Rect origin_rect(ox, oy, ow, oh);
	return origin_rect;
}
