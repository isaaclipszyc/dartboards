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
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

float fn = 0;

// struct box {
// 	int x;
// 	int y;
// 	int width;
// 	int height;
// } boxes [11];


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "Annotated5.jpg", frame );

	return 0;
}

int IoU(Point gtStart, Point gtEnd, Point predStart, Point predEnd){
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

		// printf("%f, ", gtArea);
		// printf("%f, ", predArea);
		// printf("%f, ", intersectionArea);
		// printf("%f \n", area);

		float iou = intersectionArea / area;

		//printf("%f \n", iou);

		if(iou > 0.55){
			return 1;
		} else {
			return 0;
		}

}

float f1(int tp, int fp, float fn ){

		float f1 = (float)(2* tp) / (float)((2*tp) + fp + fn);

		printf("%d\n", tp );
		printf("%d\n", fp );
		printf("%d\n", fn );
		printf("%f\n", f1 );


		return f1;
}

void drawGroundTruth(int x, int y, int width, int height, Mat frame){
	rectangle(frame, Point(x,y), Point(x+width, y+width), Scalar(0,0,255),2);
	fn++;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found

	int tp = 0;
       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		drawGroundTruth(68,143,52,63,frame);
		drawGroundTruth(58,258,59,63,frame);
		drawGroundTruth(198,224,52,63,frame);
		drawGroundTruth(255,170,49,65,frame);
		drawGroundTruth(296,248,51,67,frame);
		drawGroundTruth(382,191,56,62,frame);
		drawGroundTruth(434,243,54,64,frame);
		drawGroundTruth(514,184,51,61,frame);
		drawGroundTruth(562,255,55,63,frame);
		drawGroundTruth(651,193,47,64,frame);
		drawGroundTruth(681,252,51,63,frame);
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		tp += IoU(Point(68,143),Point(68+52,143+63), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(58,258),Point(58+59,258+63), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(198,224),Point(198+52,224+63), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(255,170),Point(255+49,170+65), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(296,248),Point(296+51,248+67), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(382,191),Point(382+56,191+62), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(434,243),Point(434+54,243+64), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(514,184),Point(514+51,184+61), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(562,255),Point(562+55,255+63), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(651,193),Point(651+47,193+64), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
		tp += IoU(Point(681,252),Point(681+51,252+63), Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
	}
	printf("%d \n", tp);

	int fp = faces.size() - tp;
	fn = fn / faces.size();
	float f1_score = f1(tp, fp, fn);

	printf("%f\n", f1_score );
}
