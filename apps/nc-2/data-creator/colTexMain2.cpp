#include <stdio.h>
#include <opencv2/opencv.hpp>

void loadTextures(std::vector<cv::Mat>& textures) {
	char fname[256];
	for (int i = 1; i < 6; ++i) {
		for (int j = 1; j < 15; ++j) {
			snprintf(fname, 255, "textures/1.%d.%02d.tiff", i, j);
			cv::Mat T = cv::imread(fname, cv::IMREAD_GRAYSCALE);
			if (!T.empty()) {
				textures.push_back(T);
			}
		}
	}
}

void splitVert(cv::Mat& I) {
	I.setTo(cv::Scalar(0, 0, 0));
	//cv::rectangle(I, cv::Rect(0, 0, I.size().width / 2, I.size().height), cv::Scalar(255, 255, 255), -1);

	int m = I.size().width - 1;
	int x0 = rand() % 32 + 32;
	int x95 = rand() % 32 + 32;
	int x48 = (x0 + x95)/2 + rand()%16;
	int x24 = (x0 + x48) / 2 + rand() % 8;
	int x72 = (x48 + x95) / 2 + rand() % 8;

	cv::line(I, cv::Point(x0,  0 ), cv::Point(x24, 24), cv::Scalar(255, 255, 255), 1, cv::LINE_8);
	cv::line(I, cv::Point(x24, 24), cv::Point(x48, 48), cv::Scalar(255, 255, 255), 1, cv::LINE_8);
	cv::line(I, cv::Point(x48, 48), cv::Point(x72, 72), cv::Scalar(255, 255, 255), 1, cv::LINE_8);
	cv::line(I, cv::Point(x72, 72), cv::Point(x95, 95), cv::Scalar(255, 255, 255), 1, cv::LINE_8);
 
	cv::floodFill(I, cv::Point(x0+5, 5), cv::Scalar(255, 255, 255));
}

void splitHorz(cv::Mat& I) {
	splitVert(I);
	cv::rotate(I, I, cv::ROTATE_90_CLOCKWISE);
}

void generateMasks(cv::Mat& col, cv::Mat& tex) {

	if (rand() % 2) {
		splitVert(col);
		splitHorz(tex);
	} else {
		splitVert(tex);
		splitHorz(col);
	}

}

int main() {
	std::vector<cv::Mat> textures;
	loadTextures(textures);

	cv::Size imgSize(96, 96);


	cv::Mat final_image(imgSize, CV_8UC3);

	cv::Mat col(imgSize, CV_8UC1);
	cv::Mat tex(imgSize, CV_8UC1);

	// cv::namedWindow("Image");
	// cv::namedWindow("Colour");
	// cv::namedWindow("Texture");

	auto noKey = cv::waitKey(10);
	bool done = false;

	cv::Rect roi(128, 128, 96, 96);

	int count = 0;
	char fname[255];
	cv::Mat noise(imgSize, CV_8UC1);
	// cv::Mat img1(imgSize, CV_8UC1);

	while (!done && count < 1000) {

		cv::Mat img = cv::Mat::zeros(imgSize, CV_8UC1);
		cv::Mat img1 = cv::Mat::zeros(imgSize, CV_8UC1);

		++count;
		generateMasks(col, tex);

		int ix1 = rand() % textures.size();
		int ix2 = ix1;
		while (ix1 == ix2) {
			ix2 = rand() % textures.size();
		}

		roi.x = rand() % 256 + 128;
		roi.y = rand() % 256 + 128;
		textures[ix1](roi).copyTo(img1, 255-tex);

		roi.x = rand() % 256 + 128;
		roi.y = rand() % 256 + 128;
		textures[ix2](roi).copyTo(img, tex);

		std::vector<cv::Mat> images(3);
		cv::randn(noise, 0, 128);
		images.at(0) = noise; //for blue channel
		images.at(1) = img;   //for green channel
		images.at(2) = img1;  //for red channel

		// cv::Mat img;
		cv::merge(images, final_image);
		cv::imshow("final_image", final_image);
		cv::imshow("noise", noise);

		// cv::imshow("Image", img);
		// cv::imshow("Image1", img1);


		// cv::imshow("Colour", col);
		// cv::imshow("Texture", tex);
		// cv::imshow("tex2", 255-tex);
		snprintf(fname, 255, "samples/img2/img%04d.png", count);
		cv::imwrite(fname, final_image);
		snprintf(fname, 255, "samples/tex2/tex%04d.png", count);
		cv::imwrite(fname, tex);

		done = cv::waitKey(50) != noKey;


	}


	return 0;

}
