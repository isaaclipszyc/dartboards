/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

int image = 0;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	String input = argv[1];
	if(input == "dart0.jpg") image = 1;
	if(input == "dart1.jpg") image = 2;
	if(input == "dart2.jpg") image = 3;
	if(input == "dart3.jpg") image = 4;
	if(input == "dart4.jpg") image = 5;
	if(input == "dart5.jpg") image = 6;
	if(input == "dart6.jpg") image = 7;
	if(input == "dart7.jpg") image = 8;
	if(input == "dart8.jpg") image = 9;
	if(input == "dart9.jpg") image = 10;
	if(input == "dart10.jpg") image = 11;
	if(input == "dart11.jpg") image = 12;
	if(input == "dart12.jpg") image = 13;
	if(input == "dart13.jpg") image = 14;
	if(input == "dart14.jpg") image = 15;
	if(input == "dart15.jpg") image = 16;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "Detected.jpg", frame );

	return 0;
}

int IoU(float threshold, Point gtStart, Point gtEnd, Point predStart, Point predEnd){
	 	if(gtStart.x > predEnd.x || predStart.x > gtEnd.x){
			return 0;
		}
		if(gtStart.y > predEnd.y || predStart.y > gtEnd.y){
			return 0;
		}

		int xA = max(gtStart.x, predStart.x);
		int yA = max(gtStart.y, predStart.y);
		int xB = min(gtEnd.x, predEnd.x);
		int yB = min(gtEnd.y, predEnd.y);

		float intersectionArea =  ((xB - xA) + 1) * ((yB - yA) + 1);

		float gtArea = (gtEnd.x - gtStart.x + 1) * (gtEnd.y - gtStart.y + 1);
		float predArea = (predEnd.x - predStart.x + 1) * (predEnd.y - predStart.y + 1);

		float area = gtArea + predArea - intersectionArea;

		float iou = intersectionArea / area;

		if(iou > threshold){
			return 1;
		} else {
			return 0;
		}

}

float f1(int tp, int fp, int fn ){

		float f1 = (float)(2 * tp) / (float)((2*tp) + fp + fn);

		return f1;
}

void drawGroundTruth(int x, int y, int width, int height, Mat frame){
	rectangle(frame, Point(x,y), Point(x+width, y+width), Scalar(0,0,255),2);
	// fn++;
}

int SobelEdgeDetection(cv::Mat &input, cv::Mat &edges, float sobel[]){

  edges.create(input.size(), (float) input.type());

  cv::Mat kernel(3,3,CV_32FC1,sobel);

  float kernelRadiusX = (float)  ( kernel.size[0] - 1 ) / 2;
	float  kernelRadiusY = (float)  ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < (int)input.rows; i++ )
	{
		for( int j = 0; j < (int)input.cols; j++ )
		{
			float sum = 0.0;
			for( int m = (int)-kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = (int)-kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + (int)kernelRadiusX;
					int imagey = j + n + (int)kernelRadiusY;
					int kernelx = m + (int)kernelRadiusX;
					int kernely = n + (int)kernelRadiusY;

					// get the values from the padded image and the kernel
					float imageval = ( float ) paddedInput.at<float>( imagex, imagey );
					float kernalval = (float)  kernel.at<float>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			edges.at<float>(i, j) = (float) sum;
		}
	}

  return 0;
}

void magnitude(cv::Mat &inputX, cv::Mat &inputY, cv::Mat &mag){
  mag.create(inputX.size(), (float) inputX.type());
  for ( int i = 0; i < inputX.rows; i++ )
  {
    for( int j = 0; j < inputX.cols; j++ )
    {
      float xsquared = inputX.at<float>(i,j) * inputX.at<float>(i,j);
      float ysquared = inputY.at<float>(i,j) * inputY.at<float>(i,j);
      float root = (float) sqrt(xsquared + ysquared);
      if(root > 255){
        root = 255;
      }
      if(root < 0){
        root = 0;
      }
      mag.at<float>(i,j) = root;
    }
  }
}

void direction(cv::Mat &inputX, cv::Mat &inputY, cv::Mat &dir){
	dir.create(inputX.size(), (float) inputX.type());
	for ( int i = 0; i < inputX.rows; i++ ){
    for( int j = 0; j < inputX.cols; j++ ){
			if(inputX.at<float>(i,j) != 0){
	      float theta = (float) atan2( inputY.at<float>(i,j), inputX.at<float>(i,j));
	      dir.at<float>(i,j) = theta;
			}
    }
  }
}

void ThresholdMagnitude(cv::Mat &Magnitude, cv::Mat &thresholdMagnitude) {
	thresholdMagnitude.create(Magnitude.size(), (float) Magnitude.type());
	for ( int i = 0; i < Magnitude.rows; i++) {
		for ( int j = 0; j < Magnitude.cols; j++) {
			int thresholdVal = 150;
			if (Magnitude.at<float>(i, j) > thresholdVal) {
			thresholdMagnitude.at<float>(i, j) = 255;
			}
			else {
			thresholdMagnitude.at<float>(i, j) = 0;
			}
		}
	}
}

void HoughTransform(cv::Mat &Frame,cv::Mat &Magnitude, cv::Mat &Orientation, cv::Mat &Hough, cv::Mat &Hough2,  std::vector<Rect> dartboards){

	int min_radius  = 30;
	int max_radius = 125;
	int sizes[] = {Magnitude.rows + max_radius, Magnitude.cols + max_radius, max_radius};
	Hough.create(3, sizes, (float) Magnitude.type());
	Hough2.create(Magnitude.rows, Magnitude.cols, (float) Magnitude.type());

	float thetaThreshold = 0.12;

	float thetaIncrement = (2 * PI) / 360.0;

	for (int i = 0; i < Magnitude.rows; i ++) {
		for (int j = 0; j < Magnitude.cols; j ++) {
			if (Magnitude.at<float>(i,j) > 0) {
				for (int r = min_radius; r < max_radius; r++) {
					float direction = Orientation.at<float>(i,j);
					for (float t = direction - thetaThreshold; t < direction + thetaThreshold; t += thetaIncrement) {
						int x0 = j + r * cosf(t);
						int y0 = i + r * sinf(t);
						int nx0 = j - r * cosf(t);
						int ny0 = i - r * sinf(t);
						if(x0 > 0 && x0 < Magnitude.cols && y0 > 0 && y0 < Magnitude.rows){
							Hough.at<float>(y0 , x0, r)++;
						}
						if(nx0 > 0 && nx0 < Magnitude.cols && ny0 > 0 && ny0 < Magnitude.rows){
							Hough.at<float>(ny0 , nx0, r)++;
						}
						if(nx0 > 0 && nx0 < Magnitude.cols && y0 > 0 && y0 < Magnitude.rows){
							Hough.at<float>(y0 , nx0, r)++;
						}
						if(x0 > 0 && x0 < Magnitude.cols && ny0 > 0 && ny0 < Magnitude.rows){
							Hough.at<float>(ny0 , x0, r)++;
						}
					}
				}
			}
		}
	}

	for ( int i = 0; i < Magnitude.rows; i++) {
	 	for ( int j = 0; j < Magnitude.cols; j++) {
			for( int r = min_radius; r < max_radius; r++){
				Hough2.at<float>(i,j) += Hough.at<float>(i,j,r);
			}
		}
	}

normalize(Hough2, Hough2, 0, 255, CV_MINMAX);

int tp = 0;
int fp = 0;
int gt = 0;


// printf("image: %d\n", image);
switch(image){
	case 1: drawGroundTruth(453,30,135,154, Frame); gt = 1;  break;//dart0
  case 2: drawGroundTruth(205,140,176,171, Frame); gt = 1; break;//dart1
	case 3: drawGroundTruth(108,103,77,76, Frame); gt = 1; break;// dart2
	case 4: drawGroundTruth(329,154,57,60, Frame); gt = 1; break;// dart3
	case 5: drawGroundTruth(191,110,194,179, Frame); gt = 1; break;//dart4
	case 6: drawGroundTruth(440,148,93,93, Frame); gt = 1; break;//dart5
	case 7: drawGroundTruth(216,122,54,54, Frame); gt = 1; break;//dart6
	case 8: drawGroundTruth(262, 184, 110, 122, Frame); gt = 1; break;//dart7
	case 9: {
		drawGroundTruth(851, 231, 98, 97, Frame); //dart8
		drawGroundTruth(71, 258, 50, 78, Frame); //dart8
		gt = 2;
		break;
	}
	case 10: drawGroundTruth(219, 63, 199, 198, Frame); gt = 1; break;//dart9
	case 11: {
		drawGroundTruth(99, 112, 81, 94, Frame); //dart10
		drawGroundTruth(588, 135, 48, 72, Frame); //dart10
		drawGroundTruth(920, 156, 30, 57, Frame); //dart10
		gt = 3;
		break;
	}
	case 12: drawGroundTruth(177, 107, 55, 50, Frame); gt = 1; break; //dart11
	case 13: drawGroundTruth(162, 84, 50, 121, Frame); gt = 1; break;//dart12
	case 14: drawGroundTruth(281, 126, 114, 117, Frame); gt = 1; break;//dart13
	case 15: {
		drawGroundTruth(128, 108, 110, 111, Frame); //dart14
		drawGroundTruth(996, 100, 107, 109, Frame); //dart14
		gt = 2;
		break;
	}
	case 16: drawGroundTruth(161, 62, 116, 121, Frame); gt = 1; break;//dart15
}


for(int d = 0; d < dartboards.size(); d++){
	int dartCorrect = 0;
  int min = 999;
	int max = 0;
  for ( int i = dartboards[d].x; i < (dartboards[d].x + dartboards[d].width); i++) {
    for ( int j = dartboards[d].y; j < (dartboards[d].y + dartboards[d].height); j++) {
      int dartboard  = 0;
      int iou = 0;
      for( int r = min_radius; r < max_radius; r++){
        if(Hough.at<float>(j,i,r) > 120){
          dartboard++;
          if(r < min){
            min = r;
          }
					if(r > max){
						max = r;
					}
        }
      }
      if(dartboard > 0){
        iou = IoU(0.50, Point(i - (max), j - (max)),Point(i + (max), j + (max)), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
        if(iou == 1){
          rectangle(Frame, Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height), Scalar( 0, 255, 0 ), 2);
					dartCorrect = 1;
				}
        for( int r = min_radius; r < max_radius; r++){
          if(Hough.at<float>(j,i,r) > 120){
            circle(Frame, Point(i,j), r, Scalar( 255, 0, 0 ), 1);
          }
        }
      }
    }
  }
	int iouR = 0;
	if (dartCorrect == 1){
		switch(image){
			case 1: iouR += IoU(0.55, Point(453,30),Point(453 + 135, 30 + 154), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));  break;
		  case 2: iouR += IoU(0.55, Point(205,140),Point(205 + 176, 140 + 171), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 3: iouR += IoU(0.55, Point(108,103),Point(108 + 77, 103 + 76), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 4: iouR += IoU(0.55, Point(329,154),Point(329 + 57, 154 + 60), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));  break;
			case 5: iouR += IoU(0.55, Point(191,110),Point(191 + 194, 110 + 179), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));  break;
			case 6: iouR += IoU(0.55, Point(440,148),Point(440 + 93, 148 + 93), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 7: iouR += IoU(0.55, Point(216,122),Point(216 + 54, 122 + 54), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));  break;
			case 8: iouR += IoU(0.55, Point(262,184),Point(262 + 110, 184 + 112), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 9: {
				iouR += IoU(0.55, Point(851,231),Point(851 + 98, 231 + 97), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				iouR += IoU(0.55, Point(71,258),Point(71 + 50, 258 + 78), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				break;
			}
			case 10: iouR += IoU(0.55, Point(219,63),Point(219 + 199, 63 + 198), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 11: {
				iouR += IoU(0.55, Point(99,112),Point(99 + 81, 112 + 94), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				iouR += IoU(0.55, Point(588,135),Point(588 + 48, 135 + 72), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				iouR += IoU(0.55, Point(920,156),Point(920 + 30, 156 + 57), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				break;
			}
			case 12: iouR += IoU(0.55, Point(177,107),Point(177 + 55, 107 + 50), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));break;
			case 13: iouR += IoU(0.55, Point(162,84),Point(162 + 50, 84 + 121), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 14: iouR += IoU(0.55, Point(281,126),Point(281 + 114, 126 + 117), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height)); break;
			case 15: {
				iouR += IoU(0.55, Point(128,108),Point(128 + 110, 108 + 111), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				iouR += IoU(0.55, Point(996,100),Point(996 + 107, 100 + 109), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));
				break;
			}
			case 16: iouR += IoU(0.55, Point(161,62),Point(161 + 116, 62 + 121), Point(dartboards[d].x, dartboards[d].y), Point(dartboards[d].x + dartboards[d].width, dartboards[d].y + dartboards[d].height));  break;
		}
		if(iouR > 0){
			tp++;
		} else {
			fp++;
		}
	}
}

int fn = gt - tp;
float f1_score = f1(tp,fp,fn);
printf("tp: %d \n", tp);
printf("fp: %d \n", fp);
printf("fn: %d \n", fn);
printf("gt: %d \n", gt);
printf("f1: %f \n", f1_score);


}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> dartboards;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

   // 3. Print number of Faces found
  frame_gray.convertTo(frame_gray, CV_32FC1);

	// imwrite( "frame0.jpg", frame);

	Mat edgesY;
	edgesY.convertTo(edgesY, CV_32FC1);

  Mat edgesX;
	edgesX.convertTo(edgesX, CV_32FC1);


  float sobelX[] = {-1,0,1,-2,0,2,-1,0,1};
  float sobelY[] = {1,2,1,0,0,0,-1,-2,-1};
  int x = SobelEdgeDetection(frame_gray,edgesX,sobelX);
  int y = SobelEdgeDetection(frame_gray,edgesY,sobelY);

  Mat mag;
  magnitude(edgesX, edgesY, mag);

	Mat thresholdMagnitude;
 	ThresholdMagnitude(mag, thresholdMagnitude);

	Mat dir;
	direction(edgesX, edgesY, dir);

	Mat hough;
	Mat hough2;
	HoughTransform(frame, thresholdMagnitude, dir, hough, hough2, dartboards);

	// imwrite( "hough0.jpg", hough2);
	// imwrite( "direction0.jpg", dir);
	// imwrite( "magnitude0.jpg", thresholdMagnitude);

}
