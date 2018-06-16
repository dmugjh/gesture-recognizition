// OpenCV.cpp : 定义控制台应用程序的入口点。
//
//BIG5 TRANS ALLOWED
#include "stdafx.h"
#include "windows.h"
//#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#include <process.h>
#include "atltypes.h"
#ifdef _WIN64
#pragma comment(lib, "..\\MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "MVCAMSDK.lib")
#endif
#include "..//include//CameraApi.h"

#define NO_CAMERA_BUG 1
//#define USE_CALLBACK_GRAB_IMAGE 

using namespace std;
using namespace cv;

UINT            m_threadID;		//图像抓取线程的ID
HANDLE          m_hDispThread;	//图像抓取线程的句柄
BOOL            m_bExit = FALSE;		//用来通知图像抓取线程结束
CameraHandle    m_hCamera;		//相机句柄，多个相机同时使用时，可以用数组代替	
BYTE*           m_pFrameBuffer; //用于将原始图像数据转换为RGB的缓冲区
tSdkFrameHead   m_sFrInfo;		//用于保存当前图像帧的帧头信息

int	            m_iDispFrameNum;	//用于记录当前已经显示的图像帧的数量
float           m_fDispFps;			//显示帧率
float           m_fCapFps;			//捕获帧率
tSdkFrameStatistic  m_sFrameCount;
tSdkFrameStatistic  m_sFrameLast;
int					m_iTimeLast;
char		    g_CameraName[64];

INT* P_Width;
INT* P_Height;
IplImage *img = NULL;
double scale = 0.8;
int c = 0;

using namespace std;

int pmsf_value = 1;//均值漂移分割平滑系数
int MopEx_value = 1;//开运算
int Hmatch_value = 27;//模板匹配系数	

//亮度
int V_low = 24;		//11	//26
int V_high = 256;	//252	//256
//饱和度
int S_low = 11;		//15	9
int S_high = 187;	//164	203
//色相
int H_low_max = 15;//色相红黄区		//14	14
int H_high_min = 119;//色相蓝紫区	//100	110
int if_high_light = 0; //是否高光补偿	//0	
int Match_Mode = 0;	//识别模式	//0

IplImage *src = 0;
IplImage *srcResize = 0;

IplImage *img_YCrCb = 0;
CvSize newSize;
CvSize sz;

IplImage *tmp1 = NULL;
IplImage *tmp2 = NULL;
IplImage *tmp3 = NULL;
IplImage *tmp4 = NULL;
IplImage *src2 = NULL;
IplImage *src1 = NULL;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IplImage *YCrCb = NULL;
IplImage *YCrCb_mask = NULL;
IplImage *Y_channel, *Cr_channel, *Cb_channel;
IplImage *Y_cmp, *Cr_cmp, *Cb_cmp;

CvScalar Y_lower;
CvScalar Y_upper;

CvScalar Cr_lower;
CvScalar Cr_upper;

CvScalar Cb_lower;
CvScalar Cb_upper;

CvScalar YCrCb_lower;
CvScalar YCrCb_upper;



/////////////////////////
void init_hand_YCrCb()
{
	//
	img_YCrCb = cvCreateImage(sz, 8, 3);
	YCrCb_mask = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	//最终的图片
	YCrCb = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	//三通道
	Y_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cr_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cb_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	//按范围截取后
	Y_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cr_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cb_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	//Y,Cr,Cb的颜色范围
	Y_lower = CV_RGB(0, 0, 130);
	Y_upper = CV_RGB(0, 0, 130);

	Cr_lower = CV_RGB(0, 0, 125);
	Cr_upper = CV_RGB(0, 0, 125);

	Cb_lower = CV_RGB(0, 0, 132);
	Cb_upper = CV_RGB(0, 0, 147);

	YCrCb_lower = cvScalar(0, 0, 132, 0);
	YCrCb_upper = cvScalar(130, 125, 147, 0);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void hand_YCrCb()
{
	//转换到YCrBr
	cvCvtColor(src2, img_YCrCb, CV_RGB2YCrCb);


	//分割到Y,Cr,Cb
	cvSplit(img_YCrCb, Y_channel, Cr_channel, Cb_channel, 0);

	//将Y_channel的位于 Y_lower 和 Y_upper 之间的元素复制到 Y_tmp中
	cvInRangeS(Y_channel, Y_lower, Y_upper, Y_cmp);
	cvInRangeS(Cr_channel, Cr_lower, Cr_upper, Cr_cmp);
	cvInRangeS(Cb_channel, Cb_lower, Cb_upper, Cb_cmp);

	//合并Y,Cr,Cb通道到YCrCb中
	cvMerge(Y_cmp, Cr_cmp, Cb_cmp, 0, YCrCb);

	//显示结果
	//cvShowImage("YCrCb_mask-基于OpenCV对于浮空手势识别技术的探究", YCrCb);


	//cvInRangeS (img_YCrCb, YCrCb_lower, YCrCb_upper, YCrCb_mask);
	//cvShowImage( "YCrCb_mask", YCrCb_mask);


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

IplImage* hsv_image;
IplImage* hsv_mask;
CvScalar  hsv_min;
CvScalar  hsv_max;

IplImage *H_img, *S_img, *V_img;
IplImage *H_mask, *H_mask1, *S_mask, *S_mask1, *V_mask, *V_mask1, *V_mask2;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void init_hand_HSV()
{

	hsv_image = cvCreateImage(sz, 8, 3);
	hsv_mask = cvCreateImage(sz, 8, 1);
	hsv_min = cvScalar(0, 20, 20, 0);
	hsv_max = cvScalar(20, 250, 255, 0);
	//hsv_mask->origin = 1;

	//方法2: 单独处理各个通道
	H_img = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	S_img = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	V_img = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	H_mask = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	H_mask1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	S_mask = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	S_mask1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	V_mask = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	V_mask2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	V_mask1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);



}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void color_blance()
{
	CvScalar avg = cvAvg(H_img, 0);
	printf("%f, %f, %f, %f\n", avg.val[0], avg.val[1], avg.val[2], avg.val[3]);
	double d = 128 - avg.val[0];
	avg.val[0] = d;
	cvAddS(H_img, avg, H_img, 0);


}





///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void hand_HSV()
{

	cvCvtColor(src2, hsv_image, CV_BGR2HSV);
	//cvInRangeS (hsv_image, hsv_min, hsv_max, hsv_mask);
	//cvShowImage( "hsv_msk", hsv_mask);




	//方法2: 单独处理各个通道
	cvSplit(hsv_image, H_img, S_img, V_img, 0);

	//color_blance();
	//cvMerge(H_img,S_img,V_img,0,hsv_image);
	//cvShowImage( "色彩平衡后", hsv_image);
	//cvShowImage( "H通道", H_img);





	//直方图均衡化(效果更差)
	//cvEqualizeHist(H_img, H_img);

	//cvShowImage( "H通道_均衡化", H_img);

	//自适应
	//cvAdaptiveThreshold(H_img, H_mask, 30, 0, 0, 3, 5);

	//cvShowImage( "H通道", H_img);
	//cvShowImage( "S通道", S_img);
	//cvShowImage( "V通道", V_img);

	//色相
	cvInRangeS(H_img, cvScalar(0, 0, 0, 0), cvScalar(H_low_max, 0, 0, 0), H_mask);//红色区
	cvInRangeS(H_img, cvScalar(256 - H_high_min, 0, 0, 0), cvScalar(256, 0, 0, 0), H_mask1);//紫色区

	//饱和度
	cvInRangeS(S_img, cvScalar(S_low, 0, 0, 0), cvScalar(S_high, 0, 0, 0), S_mask); //中间区
	//cvInRangeS(S_img,cvScalar(20,0,0,0),cvScalar(100,0,0,0),S_mask1); //低饱和度



	//亮度
	cvInRangeS(V_img, cvScalar(V_high, 0, 0, 0), cvScalar(256, 0, 0, 0), V_mask);//高亮区
	cvInRangeS(V_img, cvScalar(V_low, 0, 0, 0), cvScalar(V_high, 0, 0, 0), V_mask1); //中间区
	//cvInRangeS(V_img,cvScalar(150,0,0,0),cvScalar(250,0,0,0),V_mask2); //较亮区


	//红黄, 和蓝紫的混合
	cvOr(H_mask1, H_mask, H_mask, 0);//对两个数组进行按位或操作；

	//消除饱和度过低区域
	cvAnd(H_mask, S_mask, H_mask, 0);//对两个数组进行按位与操作

	//cvShowImage( "饱和度过滤", H_mask);

	//消去过亮过暗区域
	cvAnd(H_mask, V_mask1, H_mask, 0);//对两个数组进行按位与操作

	//cvShowImage( "亮度过滤", H_mask);

	//cvShowImage( "hsv_msk", H_mask);



	//补偿过亮区域
	if (if_high_light){ cvOr(H_mask, V_mask, H_mask, 0); }

	//cvShowImage( "补偿高光", H_mask);

	//cvShowImage( "曝光过度 V", V_mask);
	//cvShowImage( "曝光补偿", S_mask);


	//是否补偿曝光过度
	hsv_mask = H_mask;

	//cvShowImage( "hsv_msk", H_mask);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////


//阀值化
IplImage* thd_src;
IplImage* thd_dst1;
IplImage* thd_dst2;
int thd_max = 255;
int thd_val = 100;

void inti_threshold()
{
	thd_src = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	thd_dst1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	thd_dst2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
}

void threshold()
{
	cvCvtColor(src1, thd_src, CV_RGB2GRAY);
	cvAdaptiveThreshold(thd_src, thd_dst1, thd_max, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
	cvThreshold(thd_src, thd_dst2, thd_val, thd_max, CV_THRESH_BINARY);


	cvShowImage("阀值前", thd_src);

	cvShowImage("阀值化1", thd_dst1);
	cvCreateTrackbar("thd_max", "阀值化1", &thd_max, 256, 0);
	cvShowImage("阀值化2", thd_dst2);
	cvCreateTrackbar("thd_val", "阀值化2", &thd_val, 256, 0);

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resizeSrc()
{
	double scale = 0.5;


	//获得图像大小
	sz = cvGetSize(src);
	newSize.height = (int)(sz.height * scale);
	newSize.width = (int)(sz.width * scale);

	src = cvCreateImage(newSize, IPL_DEPTH_8U, 3);

	cvResize(src, src, CV_INTER_LINEAR);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

IplImage* smooth1;
IplImage* smooth2;
IplImage* smooth3;
IplImage* smooth4;
IplImage* smooth5;

void reduce_noise()
{
	//cvSmooth(hsv_mask, smooth1, CV_GAUSSIAN, 3, 0, 0, 0);
	//cvSmooth(hsv_mask, smooth1, CV_MEDIAN, 3, 0, 0, 0);

	//cvSmooth(smooth1, smooth1, CV_BILATERAL, 3, 0, 0, 0);

	//cvDilate(hsv_mask, smooth1, 0 ,2);
	//cvErode(smooth1 ,smooth2,   0, 2);

	cvMorphologyEx(hsv_mask, smooth1, 0, NULL, CV_MOP_CLOSE, MopEx_value);//改为NULL Defualt 3*3 结构元素
	//cvMorphologyEx(smooth1, smooth2, 0, CV_SHAPE_RECT, CV_MOP_OPEN, 1);

	//cvShowImage("扩张腐蚀-基于OpenCV对于浮空手势识别技术的探究", smooth1);

	//cvSmooth(smooth2, smooth3, CV_MEDIAN, 9, 0, 0, 0);

	//cvShowImage( "平滑", smooth3);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//轮廓匹配

IplImage*    g_image = NULL;
IplImage*    g_gray = NULL;
int        g_thresh = 100;
CvMemStorage*  g_storage = NULL;
CvMemStorage*  g_storage1 = NULL;
CvMemStorage*  g_storage2 = NULL;

CvSeq* seqMidObj = 0;//塞选后的轮廓集合
int handNum = 0;

int HandArea = 0;

///////////////////////////////////////////////////////////////////////////

void hand_contours(IplImage* dst) {

	if (g_storage == NULL) {
		g_storage = cvCreateMemStorage(0);
		g_storage1 = cvCreateMemStorage(0);
		g_storage2 = cvCreateMemStorage(0);
	}
	else {
		cvClearMemStorage(g_storage);
		cvClearMemStorage(g_storage1);
		cvClearMemStorage(g_storage2);
	}


	int i = 0, j = 0;
	CvSeq* contours = 0;
	CvSeq* contoursHead = 0;
	CvSeq* p = NULL;
	CvSeq* q = NULL;


	seqMidObj = 0;
	handNum = 0;

	//cvThreshold( g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY );

	cvFindContours(dst, g_storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);//只查找外轮廓
	contoursHead = contours;//contours的头
	//cvZero( g_gray );

	/*CvSeq* seqApprox;

	CvSeq* head;
	int j=0;*/
	/*for(p=contours;p != NULL; p = p->h_next){

	j++;

	seqApprox = cvApproxPoly(p, sizeof(CvContour),g_storage1, CV_POLY_APPROX_DP, 2, 0);
	seqApprox = seqApprox->h_next;
	if(p==contours) head  = seqApprox;

	}
	printf("total = %d\n ",j );
	seqApprox = head;*/


	cvZero(tmp3);
	//cvZero( tmp2 );

	if (contours)cvDrawContours(tmp3, contours, cvScalarAll(255), cvScalar(255, 0, 0, 0), 1);//绘制轮廓
	//if( contours )cvDrawContours( tmp2, contours, cvScalar(255,0,0,0),cvScalar(255,100,0,0),1);//绘制轮廓

	cvShowImage( "查找轮廓-基于OpenCV对于浮空手势识别技术的探究", tmp3);

	//CvSeq* seqMidObj = 0;// = cvCreateSeq(CV_SEQ_ELTYPE_CODE, sizeof(CvSeq), sizeof(int),g_storage2);


	//去除与窗口邻接的轮廓
	contours = contoursHead; i = 0;
	CvRect bound;
	int dat = 2;

	//去除小面积区域
	int contArea = 0;
	int imgArea = newSize.height * newSize.width;

	for (; contours != 0; contours = contours->h_next){

		i++;

		//如果面积过小, 则排除
		contArea = fabs(cvContourArea(contours, CV_WHOLE_SEQ));

		if ((double)contArea / imgArea < 0.015){ continue; }

		//如果边界与窗口相连, 则排除
		bound = cvBoundingRect(contours, 0);

		if (bound.x < dat
			|| bound.y < dat
			|| bound.x + bound.width + dat > newSize.width
			|| bound.y + bound.height + dat > newSize.height)
		{
			//printf(" %d, %d, %d, %d\n",bound.x, bound.y, bound.width, bound.height );
			//cvRectangle(tmp3, cvPoint(bound.x, bound.y),cvPoint(bound.x + bound.width, bound.y + bound.height),cvScalar(255,255,255,255),1,8,0);
			continue;
		}

		//建立轮廓链表
		q = p;
		//p = cvCloneSeq(contours, g_storage2);
		p = contours;

		if (q == NULL){
			seqMidObj = p;
			//p->h_next = NULL;
			//p->h_prev = NULL;
			//printf("第1个!");
		}
		else{
			q->h_next = p;
			p->h_prev = q;
			//printf("1个!");
		}
		//j++;
		handNum++;

	}

	//printf("找到轮廓: %d 个   塞选: %d 个\n", i,j);

	
	if (seqMidObj){
		seqMidObj->h_prev = NULL;
		p->h_next = NULL;
	}
	if (handNum > 0)
	{

		//printf("找到手: %d  ", handNum);
		HandArea = fabs(cvContourArea(seqMidObj, CV_WHOLE_SEQ));
		printf("手面积: %d  ", HandArea);
	}
	//CvSeq* seqMidObj_head = seqMidObj;

	cvZero(tmp3);
	if (seqMidObj)cvDrawContours(tmp3, seqMidObj, cvScalarAll(255), cvScalar(255, 0, 0, 0), 1);//绘制轮廓

	//cvShowImage( "轮廓筛选-基于OpenCV对于浮空手势识别技术的探究", tmp3);


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

IplImage*    tmp_img = 0;
CvMemStorage*  storage_tmp = 0;

CvSeq* handT = 0;
CvSeq* handT1 = 0;
CvSeq* handT2 = 0;

int handTNum = 10;//10个模板
//int handTNum = 2;//10个模板

char *tmp_names[] = { "1.bmp", "2.bmp", "3.bmp", "4.bmp", "5.bmp", "6.bmp", "7.bmp", "8.bmp", "9.bmp", "10.bmp" };
char *num_c[] = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" };


/////////////////////////////////////////////////////

//载入模板的轮廓
void init_hand_template()
{

	storage_tmp = cvCreateMemStorage(0);

	int i = 0;
	for (i = 0; i<handTNum; i++){

		tmp_img = cvLoadImage(tmp_names[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (!tmp_img){
			printf("未找到文件: %s\n", tmp_names[i]);
			continue;
		}
		//cvShowImage("载入模板", tmp_img);
		handT1 = handT2;
		cvFindContours(tmp_img, storage_tmp, &handT2, sizeof(CvContour), CV_RETR_EXTERNAL);

		if (handT2){
			printf("载入模板: %s 成功!\n", tmp_names[i]);
			if (handT1 == NULL){
				printf("载入第一个模板!\n");
				handT = handT2;
			}
			else{
				handT2->h_prev = handT1;
				handT1->h_next = handT2;
			}

		}

	}


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool if_match_num = false;
int match_num = -1;

//模板匹配手
void hand_template_match(CvSeq* handT, CvSeq* hand){
	int i = 0;
	int kind = -1;
	double hu = 1;

	double hutmp;
	CvSeq* handp = handT;
	int method = CV_CONTOURS_MATCH_I1;

	match_num = 0;

	if (handT == NULL){ return; printf("handT==NULL!\n"); }
	if (hand == NULL){ return; printf("hand==NULL!\n"); }

	for (i = 0; i<handTNum; i++){
		hutmp = cvMatchShapes(handp, hand, method, 0);
		handp = handp->h_next;

		//找到hu矩最小的模板
		if (hu > hutmp){
			hu = hutmp;
			kind = i + 1;
		}

		//printf("%f ", hu);
	}

	//显示匹配结果
	if (hu<((double)Hmatch_value) / 100){
		printf("匹配模板: %d (%f)", kind, hu);
		match_num = kind;
		if_match_num = true;
	}
	else{
		if_match_num = false;
	}
}


//模板匹配手
void hand_template_match2(CvSeq* handT, CvSeq* hand){
	int i = 0;
	int kind = -1;
	double hu = 1;

	double hutmp;
	CvSeq* handp = handT;
	int method = CV_CONTOURS_MATCH_I1;

	match_num = 0;

	if (handT == NULL){ return; printf("handT==NULL!\n"); }
	if (hand == NULL){ return; printf("hand==NULL!\n"); }

	for (i = 0; i<handTNum; i++){
		
		if(i==4||i==9)hutmp = cvMatchShapes(handp, hand, method, 0);
		handp = handp->h_next;

		//找到hu矩最小的模板
		if (i == 4 || i == 9)
		if (hu > hutmp){
			hu = hutmp;
			kind = i+1;
		}

		//printf("%f ", hu);
	}

	//显示匹配结果
	if (hu<((double)Hmatch_value) / 100){
		printf("匹配模板: %d (%f)", kind, hu);
		match_num = kind;
		if_match_num = true;
	}
	else{
		if_match_num = false;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//拉普拉斯变换
int sigma = 2;
int smoothType = CV_GAUSSIAN;

IplImage* laplace = 0;
IplImage* colorlaplace = 0;
IplImage* planes[3] = { 0, 0, 0 };

int ls_low = 247;


void init_laplace()
{
	int i;
	for (i = 0; i < 3; i++)
		planes[i] = cvCreateImage(newSize, 8, 1);
	laplace = cvCreateImage(newSize, IPL_DEPTH_16S, 1);
	colorlaplace = cvCreateImage(newSize, 8, 3);

}


void toLaplace(IplImage* dst)
{
	int i, c, ksize;

	ksize = (sigma * 5) | 1;
	//cvSmooth( dst, colorlaplace, smoothType, ksize, ksize, sigma, sigma );

	cvSplit(dst, planes[0], planes[1], planes[2], 0);
	for (i = 0; i < 3; i++)
	{
		cvLaplace(planes[i], laplace, 5);
		cvConvertScaleAbs(laplace, planes[i], (sigma + 1)*0.25, 0);
	}
	cvMerge(planes[0], planes[1], planes[2], 0, colorlaplace);
	colorlaplace->origin = dst->origin;

	//前期平滑
	//cvMorphologyEx(planes[2], planes[2], 0, CV_SHAPE_RECT, CV_MOP_CLOSE, 1);

	cvShowImage("拉普拉斯变换-基于OpenCV对于浮空手势识别技术的探究", colorlaplace);

	//smoothType = smoothType == CV_GAUSSIAN ? CV_BLUR : smoothType == CV_BLUR ? CV_MEDIAN : CV_GAUSSIAN;

	cvCvtColor(colorlaplace, colorlaplace, CV_BGR2HSV);

	cvSplit(colorlaplace, planes[0], planes[1], planes[2], 0);

	cvInRangeS(planes[2], cvScalar(ls_low, 0, 0, 0), cvScalar(256, 0, 0, 0), planes[2]);

	//后期平滑
	//cvMorphologyEx(planes[2], planes[2], 0, CV_SHAPE_RECT, CV_MOP_CLOSE, 1);
	//cvSmooth(planes[2], planes[2], CV_MEDIAN, 3, 0, 0, 0);
	//cvErode(planes[2] ,planes[2], 0, 1);
	cvShowImage("拉普拉斯_边缘化-基于OpenCV对于浮空手势识别技术的探究", planes[2]);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define UP		-1
#define DOWN	1
#define LEFT	1
#define RIGHT	-1


CvPoint hand_center = cvPoint(0, 0); //按外接矩形求中心点
CvPoint hand_center_last = cvPoint(0, 0); //
CvPoint hand_direct_now = cvPoint(0, 0);  //本次监测到的移动方向
CvPoint hand_direct_last = cvPoint(0, 0); //上次监测到的移动方向
CvPoint hand_direct = cvPoint(0, 0);      //合成的方向

//求手的中点和移动方向
void hand_direction(CvSeq* hand){

	//如果没有手, 则清除结果
	if (!hand){
		hand_center = cvPoint(0, 0);
		hand_center_last = cvPoint(0, 0);
		hand_direct_now = cvPoint(0, 0);
		hand_direct_last = cvPoint(0, 0);
		hand_direct = cvPoint(0, 0);
		return;
	}

	hand_center_last = hand_center;

	//获得中心点
	CvRect bound = cvBoundingRect(hand, 0);
	hand_center.x = bound.x + bound.width / 2;
	hand_center.y = bound.y + bound.height / 2;

	if (hand_center_last.x != 0){
		hand_direct_now.x = hand_center.x - hand_center_last.x;
		hand_direct_now.y = hand_center.y - hand_center_last.y;

		if (hand_direct_now.x != 0){

			hand_direct.x = (hand_direct_now.x + hand_direct_last.x) / 2;
			if (hand_direct.x != 0){
				if (Match_Mode != 2) printf("  X 移动: %d ", hand_direct.x);
			}

		}
		if (hand_direct_now.y != 0){

			hand_direct.y = (hand_direct_now.y + hand_direct_last.y) / 2;
			if (hand_direct.y != 0){
				if (Match_Mode != 2) printf("  Y 移动: %d ", hand_direct.y);
			}
		}

	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

//绘制识别结果
void hand_draw(IplImage* dst, CvSeq* hands)
{

	if (!hands) return;

	CvRect bound;

	//凸包
	int i, hullcount;
	CvPoint pt0;
	CvSeq* hull;

	CvSeq* handp = hands;

	//凸包缺陷
	CvConvexityDefect* defect;
	CvSeq* hullDefect;
	//CvSeq* hullDefectSelect;
	int hullDefectNum = 0;
	//cvPoint** points = (cvPoint*)malloc(sizeof(cvPoint)*3);
	//cvPoint points[3];


	//绘制轮廓
	cvDrawContours(dst, handp, cvScalar(255, 150, 100, 0), cvScalar(255, 0, 0, 0), 1, 1, 8, cvPoint(0, 0));

	CvFont font;


	//绘制检测到的数字
	if (if_match_num){

		//外阴影
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0f, 1.0f, 0, 5, 8);
		cvPutText(dst, num_c[match_num - 1], cvPoint(5, 30), &font, CV_RGB(255, 255, 255));
		//内颜色
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0f, 1.0f, 0, 2, 8);
		cvPutText(dst, num_c[match_num - 1], cvPoint(5, 30), &font, CV_RGB(255, 0, 0));


	}

	//绘制移动方向
	if (1){

		//cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,2,8);
		//cvPutText(dst, "X: ", cvPoint(5, 30), &font, CV_RGB(255,0,0));



	}


	//对提取出的轮廓遍历
	for (; handp != 0; handp = handp->h_next){

		bound = cvBoundingRect(handp, 0);


		//求并绘制中心点

		//内颜色
		int det = 2;
		cvRectangle(dst,
			cvPoint(hand_center.x - det, hand_center.y - det),
			cvPoint(hand_center.x + det, hand_center.y + det),
			CV_RGB(0, 0, 0), 3, 8, 0);

		//外轮廓
		det = 3;
		cvRectangle(dst,
			cvPoint(hand_center.x - det, hand_center.y - det),
			cvPoint(hand_center.x + det, hand_center.y + det),
			CV_RGB(255, 255, 255), 1, 8, 0);



		//绘出外包络方框
		cvRectangle(dst,
			cvPoint(bound.x, bound.y),
			cvPoint(bound.x + bound.width, bound.y + bound.height),
			cvScalar(0, 0, 255, 0), 2, 8, 0);

		//寻找凸包
		hull = cvConvexHull2(handp, 0, CV_CLOCKWISE, 0);
		hullcount = hull->total;
		//printf("凸包点数: %d  ",hullcount);

		pt0 = **CV_GET_SEQ_ELEM(CvPoint*, hull, hullcount - 1);

		//画凸包
		for (i = 0; i < hullcount; i++){

			//得到凸包的点
			CvPoint pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);
			cvLine(dst, pt0, pt, CV_RGB(0, 255, 0), 1, CV_AA, 0);
			pt0 = pt;
		}

		//检查缺陷
		/*if(!cvCheckContourConvexity(hands)){

		hullDefect = cvConvexityDefects(hands, hull, 0);
		hullDefectNum = hullDefect->total;
		printf("缺陷个数: %d  ",hullDefectNum);
		for( i = 0; i < hullDefectNum; i++ ){

		defect = (CvConvexityDefect*)cvGetSeqElem(hullDefect, i);

		cvLine( dst, cvPoint(defect->end->x,defect->end->y),    cvPoint(defect->depth_point->x,defect->depth_point->y), CV_RGB(150,150,150),1,8,0);
		cvLine( dst, cvPoint(defect->start->x,defect->start->y),cvPoint(defect->depth_point->x,defect->depth_point->y), CV_RGB(150,150,150),1,8,0);


		}

		}*/

	}

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resizeAllWindow()
{
	cvNamedWindow("原图像-基于OpenCV对于浮空手势识别技术的探究", 0);//src

	//cvNamedWindow("扩张腐蚀-基于OpenCV对于浮空手势识别技术的探究", 0);

	//cvNamedWindow("最终识别-基于OpenCV对于浮空手势识别技术的探究", 0);

	cvResizeWindow("原图像-基于OpenCV对于浮空手势识别技术的探究", newSize.width, newSize.height);
	//cvResizeWindow("扩张腐蚀-基于OpenCV对于浮空手势识别技术的探究", newSize.width, newSize.height);
	//cvResizeWindow("最终识别-基于OpenCV对于浮空手势识别技术的探究", newSize.width, newSize.height);

	cvNamedWindow("参数调试-基于OpenCV对于浮空手势识别技术的探究", CV_WINDOW_AUTOSIZE);
	cvResizeWindow("参数调试-基于OpenCV对于浮空手势识别技术的探究", newSize.width*1.5, 60 * 10);


	cvCreateTrackbar("均值漂移滤波", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &pmsf_value, 20, 0);
	cvCreateTrackbar("开运算降噪", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &MopEx_value, 5, 0);

	cvCreateTrackbar("色相红黄区", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &H_low_max, 150, 0);
	cvCreateTrackbar("色相蓝紫区", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &H_high_min, 150, 0);

	cvCreateTrackbar("亮度下限", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &V_low, 100, 0);
	cvCreateTrackbar("亮度上限", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &V_high, 256, 0);

	cvCreateTrackbar("饱和度下限", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &S_low, 100, 0);
	cvCreateTrackbar("饱和度上限", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &S_high, 255, 0);

	cvCreateTrackbar("高光补偿", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &if_high_light, 1, 0);

	cvCreateTrackbar("match系数", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &Hmatch_value, 50, 0);

	cvCreateTrackbar("识别模式", "参数调试-基于OpenCV对于浮空手势识别技术的探究", &Match_Mode, 5, 0);


}

//控制鼠标
void control_mouse(int dx, int dy)
{
	CPoint point;
	
	GetCursorPos(&point);

	int scale = 10;

	point.x -= dx * scale;
	point.y += dy * scale;

	SetCursorPos(point.x, point.y);

}

//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
///*
//USE_CALLBACK_GRAB_IMAGE
//如果需要使用回调函数的方式获得图像数据，则反注释宏定义USE_CALLBACK_GRAB_IMAGE.
//我们的SDK同时支持回调函数和主动调用接口抓取图像的方式。两种方式都采用了"零拷贝"机制，以最大的程度的降低系统负荷，提高程序执行效率。
//但是主动抓取方式比回调函数的方式更加灵活，可以设置超时等待时间等，我们建议您使用 uiDisplayThread 中的方式
//*/
////#define USE_CALLBACK_GRAB_IMAGE 
//
//#ifdef USE_CALLBACK_GRAB_IMAGE
///*图像抓取回调函数*/
//IplImage *g_iplImage = NULL;
//
//void _stdcall GrabImageCallback(CameraHandle hCamera, BYTE *pFrameBuffer, tSdkFrameHead* pFrameHead,PVOID pContext)
//{
//
//	CameraSdkStatus status;
//	//tSdkFrameHead 	sFrameInfo;
//	//CameraHandle    hCamera = (CameraHandle)lpParam;
//	//BYTE*			pbyBuffer;
//	//CameraSdkStatus status;
//	//IplImage *iplImage = NULL;
//	
//
//	//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
//	//我公司大部分型号的相机，原始数据都是Bayer格式的
//	status = CameraImageProcess(hCamera, pFrameBuffer, m_pFrameBuffer,pFrameHead);
//
//	//分辨率改变了，则刷新背景
//	if (m_sFrInfo.iWidth != pFrameHead->iWidth || m_sFrInfo.iHeight != pFrameHead->iHeight)
//	{
//		m_sFrInfo.iWidth = pFrameHead->iWidth;
//		m_sFrInfo.iHeight = pFrameHead->iHeight;
//	}
//
//	if(status == CAMERA_STATUS_SUCCESS )
//	{
//		//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
//		CameraImageOverlay(hCamera, m_pFrameBuffer,pFrameHead);
//		if (g_iplImage)
//		{
//			cvReleaseImageHeader(&g_iplImage);
//		}
//		g_iplImage = cvCreateImageHeader(cvSize(pFrameHead->iWidth,pFrameHead->iHeight),IPL_DEPTH_8U,sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8?1:3);
//		cvSetData(g_iplImage,m_pFrameBuffer,pFrameHead->iWidth*(sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8?1:3));
//		cvShowImage(g_CameraName,g_iplImage);		
//		m_iDispFrameNum++;
//	    waitKey(30);
//	}    
//
//	memcpy(&m_sFrInfo,pFrameHead,sizeof(tSdkFrameHead));
//
//}
//
//#else 
///*图像抓取线程，主动调用SDK接口函数获取图像*/
//UINT WINAPI uiDisplayThread(LPVOID lpParam)
//{
//	tSdkFrameHead 	sFrameInfo;
//	CameraHandle    hCamera = (CameraHandle)lpParam;
//	BYTE*			pbyBuffer;
//	CameraSdkStatus status;
//	IplImage *iplImage = NULL;
//
//	CameraPlay(m_hCamera);
//	if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
//	{
//		//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
//		//我公司大部分型号的相机，原始数据都是Bayer格式的
//		status = CameraImageProcess(hCamera, pbyBuffer, m_pFrameBuffer, &sFrameInfo);//连续模式
//
//		//分辨率改变了，则刷新背景
//		if (m_sFrInfo.iWidth != sFrameInfo.iWidth || m_sFrInfo.iHeight != sFrameInfo.iHeight)
//		{
//			m_sFrInfo.iWidth = sFrameInfo.iWidth;
//			m_sFrInfo.iHeight = sFrameInfo.iHeight;
//			//图像大小改变，通知重绘
//		}
//
//		if (status == CAMERA_STATUS_SUCCESS)
//		{
//			//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
//			CameraImageOverlay(hCamera, m_pFrameBuffer, &sFrameInfo);
//			if (iplImage)
//			{
//				cvReleaseImageHeader(&iplImage);
//			}
//			iplImage = cvCreateImageHeader(cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight), IPL_DEPTH_8U, sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 : 3);
//			cvSetData(iplImage, m_pFrameBuffer, sFrameInfo.iWidth*(sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? 1 : 3));
//			//cvShowImage(g_CameraName, iplImage);
//			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//			src = iplImage;//cvQueryFrame(iplImage);
//			//cvPyrMeanShiftFiltering(src, src2, pmsf_value, 40, 2);//分割 – 均值漂移滤波
//			//打开摄像头失败
//			//if (!src){ return NO_CAMERA_BUG; }
//			sz = cvGetSize(src);
//			newSize.height = (int)(sz.height * scale);
//			newSize.width = (int)(sz.width * scale);
//
//			sz = newSize;
//
//			//建立所有窗体
//			resizeAllWindow();
//
//			tmp1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//
//			tmp2 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			tmp3 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			tmp4 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			src2 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			smooth1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//			smooth2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//			smooth3 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//			smooth4 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//			smooth5 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
//
//			init_hand_YCrCb();
//			init_hand_HSV();
//			init_laplace();
//			inti_threshold();
//
//			//载入匹配的模板
//			init_hand_template();
//
//			src1 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			m_iDispFrameNum++;
//		}
//
//		//在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
//		//否则再次调用CameraGetImageBuffer时，程序将被挂起，知道其他线程中调用CameraReleaseImageBuffer来释放了buffer
//		CameraReleaseImageBuffer(hCamera, pbyBuffer);
//
//		memcpy(&m_sFrInfo, &sFrameInfo, sizeof(tSdkFrameHead));
//	}
//
//
//	while (!m_bExit)
//	{   
//
//		if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
//		{	
//			//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
//			//我公司大部分型号的相机，原始数据都是Bayer格式的
//			status = CameraImageProcess(hCamera, pbyBuffer, m_pFrameBuffer,&sFrameInfo);//连续模式
//
//			//分辨率改变了，则刷新背景
//			if (m_sFrInfo.iWidth != sFrameInfo.iWidth || m_sFrInfo.iHeight != sFrameInfo.iHeight)
//			{
//				m_sFrInfo.iWidth = sFrameInfo.iWidth;
//				m_sFrInfo.iHeight = sFrameInfo.iHeight;
//				//图像大小改变，通知重绘
//			}
//
//			if(status == CAMERA_STATUS_SUCCESS)
//			{
//				//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
//				CameraImageOverlay(hCamera, m_pFrameBuffer, &sFrameInfo);
//				if (iplImage)
//				{
//					cvReleaseImageHeader(&iplImage);
//				}
//				iplImage = cvCreateImageHeader(cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight),IPL_DEPTH_8U,sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8?1:3);
//				cvSetData(iplImage,m_pFrameBuffer,sFrameInfo.iWidth*(sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8?1:3));
//				//cvShowImage(g_CameraName,iplImage);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//				src = iplImage;//cvQueryFrame(iplImage);
//				//cvShowImage("src",src);
//				cvResize(src, src1, CV_INTER_LINEAR);		//尺寸变换
//				//cvShowImage("src", src1);
//				cvPyrMeanShiftFiltering(src1, src2, pmsf_value, 40, 2);//分割 – 均值漂移滤波
//				//cvShowImage("均值漂移滤波-基于OpenCV对于浮空手势识别技术的探究", src2);
//				hand_YCrCb();//src
//				hand_HSV();		//分离HSV通道
//				reduce_noise();		//降噪
//				hand_contours(smooth1);//寻找手轮廓
//				hand_template_match(handT, seqMidObj);//寻找匹配
//
//				hand_direction(seqMidObj);//求手的移动方向
//				//绘制额外信息
//				cvZero(tmp2);
//				hand_draw(tmp2, seqMidObj); //绘制在检测窗口
//				hand_draw(src1, seqMidObj); //绘制在原窗口
//
//
//				//cvShowImage( "扩张腐蚀", smooth1);
//				cvShowImage("最终识别-基于OpenCV对于浮空手势识别技术的探究", tmp2);
//				cvShowImage("原图像-基于OpenCV对于浮空手势识别技术的探究", src1);
//				//
//				//控制鼠标(当拳头状, 则控制鼠标移动)
//				if (match_num == 10)// cvNamedWindow("xxx", 0);
//			control_mouse(hand_direct.x, hand_direct.y);
//		
//				m_iDispFrameNum++;
//			}    
//
//			//在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
//			//否则再次调用CameraGetImageBuffer时，程序将被挂起，知道其他线程中调用CameraReleaseImageBuffer来释放了buffer
//			CameraReleaseImageBuffer(hCamera,pbyBuffer);
//
//			memcpy(&m_sFrInfo,&sFrameInfo,sizeof(tSdkFrameHead));
//		}
//		
//
//
//
//		int c = waitKey(10);
//
//		if (c == 'q' || c == 'Q' || (c & 255) == 27)
//		{
//			m_bExit = TRUE;
//			break;
//		}
//	}
//
//	if (iplImage)
//	{
//		cvReleaseImageHeader(&iplImage);
//	}
//
//	_endthreadex(0);
//	return 0;
//}
//#endif




//
//int _tmain(int argc, _TCHAR* argv[])
//{
//
//
//	tSdkCameraDevInfo sCameraList[10];
//	INT iCameraNums;
//	CameraSdkStatus status;
//	tSdkCameraCapbility sCameraInfo;
//
//
//	//枚举设备，获得设备列表
//	iCameraNums = 10;//调用CameraEnumerateDevice前，先设置iCameraNums = 10，表示最多只读取10个设备，如果需要枚举更多的设备，请更改sCameraList数组的大小和iCameraNums的值
//
//	if (CameraEnumerateDevice(sCameraList,&iCameraNums) != CAMERA_STATUS_SUCCESS || iCameraNums == 0)
//	{
//		printf("No camera was found!");
//		return FALSE;
//	}
//
//	//该示例中，我们只假设连接了一个相机。因此，只初始化第一个相机。(-1,-1)表示加载上次退出前保存的参数，如果是第一次使用该相机，则加载默认参数.
//	//In this demo ,we just init the first camera.
//	if ((status = CameraInit(&sCameraList[0],-1,-1,&m_hCamera)) != CAMERA_STATUS_SUCCESS)
//	{
//		char msg[128];
//		sprintf_s(msg,"Failed to init the camera! Error code is %d",status);
//		printf(msg);
//		return FALSE;
//	}
//
//
//	//Get properties description for this camera.
//	CameraGetCapability(m_hCamera,&sCameraInfo);//"获得该相机的特性描述"
//
//	m_pFrameBuffer = (BYTE *)CameraAlignMalloc(sCameraInfo.sResolutionRange.iWidthMax*sCameraInfo.sResolutionRange.iWidthMax*3,16);	
//
//	if (sCameraInfo.sIspCapacity.bMonoSensor)
//	{
//		CameraSetIspOutFormat(m_hCamera,CAMERA_MEDIA_TYPE_MONO8);
//	} 
//	
//	strcpy_s(g_CameraName,sCameraList[0].acFriendlyName);
//
//	CameraCreateSettingPage(m_hCamera,NULL,
//		g_CameraName, NULL, NULL, 0);//"通知SDK内部建该相机的属性页面";
//
//
///////////////////////////////////////////////////////////////////////////
//	
//
//	////重新调整图像大小
//	////resizeSrc();
//
//	////建立所有窗体
//	//resizeAllWindow();
//
//	//init_hand_YCrCb();
//	//init_hand_HSV();
//	//init_laplace();
//	//inti_threshold();
//
//	////载入匹配的模板
//	//init_hand_template();
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//#ifdef USE_CALLBACK_GRAB_IMAGE //如果要使用回调函数方式，定义USE_CALLBACK_GRAB_IMAGE这个宏
//	//Set the callback for image capture
//	CameraSetCallbackFunction(m_hCamera,GrabImageCallback,NULL,NULL);//"设置图像抓取的回调函数";
//#else
//	m_hDispThread = (HANDLE)_beginthreadex(NULL, 0, &uiDisplayThread, (PVOID)m_hCamera, 0,  &m_threadID);
//#endif
//
//	//CameraPlay(m_hCamera);
//	
//	CameraShowSettingPage(m_hCamera,TRUE);//TRUE显示相机配置界面。FALSE则隐藏。
//
//	//unsigned char* capture=CameraGetImageBufferEx(m_hCamera, P_Width, P_Height, 10);
//	//IplImage* img = 
//	//cvNamedWindow("Example1", CV_WINDOW_AUTOSIZE);
//	//cvShowImage("Example1", img);
//
//	//src = cvQueryFrame(capture);
//
//	//cvPyrMeanShiftFiltering(img, src2, pmsf_value, 40, 2);//分割 – 均值漂移滤波
//
//
//
//	while(m_bExit != TRUE)
//	{
//		
//		waitKey(10);
//	}
//	
//	CameraUnInit(m_hCamera);
//
//	CameraAlignFree(m_pFrameBuffer);
//
//	destroyWindow(g_CameraName);
//
//#ifdef USE_CALLBACK_GRAB_IMAGE
//	if (g_iplImage)
//	{
//		cvReleaseImageHeader(&g_iplImage);
//	}
//#endif
//	return 0;
//}
INT32 Right_count = 0;
INT32 Left_count = 0;


int Set_HandArea = 0;
int HandArea_err = 0;
int SIZE_flag = 0;
int SF_flag = 0;
int Mouse_flag = 0;
int DIR_flag = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
int _tmain(int argc, _TCHAR* argv[])

{
	INT32 dontnow = 0;
	int c = 0;

	double scale = 0.5;

	//打开摄像头
	CvCapture* capture = cvCaptureFromCAM(0);
	src = cvQueryFrame(capture);

	//打开摄像头失败
	if (!src){ return NO_CAMERA_BUG; }

	//获得图像大小
	sz = cvGetSize(src);
	newSize.height = (int)(sz.height * scale);
	newSize.width = (int)(sz.width * scale);

	sz = newSize;


	//重新调整图像大小
	//resizeSrc();

	//建立所有窗体
	resizeAllWindow();

	tmp1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	tmp2 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	tmp3 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	tmp4 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	src2 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	smooth1 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	smooth2 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	smooth3 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	smooth4 = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	smooth5 = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	init_hand_YCrCb();
	init_hand_HSV();
	init_laplace();
	inti_threshold();

	//载入匹配的模板
	init_hand_template();

	src1 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	/////////////////////开始循环///////////////////////////////
	while (c != 27)

	{

		//当前图像
		src = cvQueryFrame(capture);
		if (!src) return NO_CAMERA_BUG;

		//缩小要处理的图像(减小运算量)
		cvResize(src, src1, CV_INTER_LINEAR);		//尺寸变换

		//cvShowImage("src", src1);

		cvPyrMeanShiftFiltering(src1, src2, pmsf_value, 40, 2);//分割 – 均值漂移滤波
		
		//cvSmooth(src1, tmp4, CV_MEDIAN, 9, 0, 0, 0);//src 输入图像  tmp4输出

		//cvShowImage("均值漂移滤波-基于OpenCV对于浮空手势识别技术的探究", src2);

		//拉普拉斯变换
		//toLaplace(src2);

		hand_YCrCb();//src  //手的区域: H(0-30),S(30-170),V(0-200)

		hand_HSV();		//分离HSV通道

		reduce_noise();		//降噪

		hand_contours(smooth1);//寻找手轮廓
		if (Match_Mode == 0)		
			hand_template_match(handT, seqMidObj);//寻找匹配
		else if (Match_Mode == 1 || Match_Mode == 2|| Match_Mode==3 )
			hand_template_match2(handT, seqMidObj);//寻找匹配

		hand_direction(seqMidObj);//求手的移动方向
		//绘制额外信息
		cvZero(tmp2);
		hand_draw(tmp2, seqMidObj); //绘制在检测窗口
		hand_draw(src1, seqMidObj); //绘制在原窗口


		//cvShowImage( "扩张腐蚀", smooth1);
		//cvShowImage("最终识别-基于OpenCV对于浮空手势识别技术的探究", tmp2);
		cvShowImage("原图像-基于OpenCV对于浮空手势识别技术的探究", src1);

		//控制鼠标(当拳头状, 则控制鼠标移动)
		if (Match_Mode == 1)
		{

			if (match_num == 10)// cvNamedWindow("xxx", 0);
			{
				if (hand_direct.x >= 10 && hand_direct.y <= abs(hand_direct.x / 2) && hand_direct.y >= -abs(hand_direct.x/2) && DIR_flag == 0)	Left_count++;			//合成的方向
				else if (hand_direct.x <= -10 && hand_direct.y <= 5 && hand_direct.y >= -5 && DIR_flag==0)	Right_count++;
				else
				{
					Right_count = 0;
					Left_count = 0;
				}
				if (Right_count >= 2)
				{
					DIR_flag = 1;
					Right_count = 0;
					keybd_event(39, 0, 0, 0);
					keybd_event(39, 0, KEYEVENTF_KEYUP, 0);	//方向右
					printf("往右");
					Beep(1800, 50);
				}
				if (Left_count >= 2)
				{
					DIR_flag = 1;
					Left_count = 0;
					keybd_event(37, 0, 0, 0);
					keybd_event(37, 0, KEYEVENTF_KEYUP, 0);	//方向左
					printf("往左");
					Beep(1800, 50);
				}
			}
			else
			{
				Right_count = 0;
				Left_count = 0;
				DIR_flag = 0;
			}
		}
		else if (Match_Mode == 2)
		{
			if (match_num == 10)
			{
				if (SF_flag != 1)Beep(1200, 80);
				SF_flag = 1;
				
			}
			else if (SF_flag == 1 && match_num == 5)
			{
				Set_HandArea = HandArea;
				SF_flag = 2;
				SIZE_flag = 5;
				printf("\n***************缩放模式******************\n");
				Beep(1600,60);
			}
			else if (SF_flag == 2 && match_num == 5)
			{
				HandArea_err = HandArea - Set_HandArea;
				if (HandArea_err >= 1000)
				{
					SIZE_flag++;
					Set_HandArea = HandArea;
					keybd_event(107, 0, 0, 0);
					keybd_event(107, 0, KEYEVENTF_KEYUP, 0);	//+
					printf("放大");
				}
				else if (HandArea_err <= -1000)
				{
					SIZE_flag--;
					Set_HandArea = HandArea;
					keybd_event(109, 0, 0, 0);
					keybd_event(109, 0, KEYEVENTF_KEYUP, 0);	//—
					printf("缩小");
				}				
			}
			else SF_flag = 0;

		}
		else if (Match_Mode == 3)
		{
			if (match_num == 10)
			{
				control_mouse(hand_direct.x, hand_direct.y);
				Mouse_flag = 0;
			}
			else if (match_num == 5 && Mouse_flag==0)
			{
				mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
				mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
				Mouse_flag = 1;
			}
			else Mouse_flag = 1;
		}


			//mouse_event(MOUSEEVENTF_WHEEL,0,0,1,dontnow);		
			//control_mouse(hand_direct.x, hand_direct.y);

		printf("\n");
		c = cvWaitKey(10);

	}
	////////////////////////////////////////////////////////////

	//cvReleaseCapture( &capture);
	cvDestroyAllWindows();

	return 0;

}