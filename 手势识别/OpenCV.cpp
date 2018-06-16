// OpenCV.cpp : �������̨Ӧ�ó������ڵ㡣
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

UINT            m_threadID;		//ͼ��ץȡ�̵߳�ID
HANDLE          m_hDispThread;	//ͼ��ץȡ�̵߳ľ��
BOOL            m_bExit = FALSE;		//����֪ͨͼ��ץȡ�߳̽���
CameraHandle    m_hCamera;		//��������������ͬʱʹ��ʱ���������������	
BYTE*           m_pFrameBuffer; //���ڽ�ԭʼͼ������ת��ΪRGB�Ļ�����
tSdkFrameHead   m_sFrInfo;		//���ڱ��浱ǰͼ��֡��֡ͷ��Ϣ

int	            m_iDispFrameNum;	//���ڼ�¼��ǰ�Ѿ���ʾ��ͼ��֡������
float           m_fDispFps;			//��ʾ֡��
float           m_fCapFps;			//����֡��
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

int pmsf_value = 1;//��ֵƯ�Ʒָ�ƽ��ϵ��
int MopEx_value = 1;//������
int Hmatch_value = 27;//ģ��ƥ��ϵ��	

//����
int V_low = 24;		//11	//26
int V_high = 256;	//252	//256
//���Ͷ�
int S_low = 11;		//15	9
int S_high = 187;	//164	203
//ɫ��
int H_low_max = 15;//ɫ������		//14	14
int H_high_min = 119;//ɫ��������	//100	110
int if_high_light = 0; //�Ƿ�߹ⲹ��	//0	
int Match_Mode = 0;	//ʶ��ģʽ	//0

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

	//���յ�ͼƬ
	YCrCb = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	//��ͨ��
	Y_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cr_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cb_channel = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	//����Χ��ȡ��
	Y_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cr_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	Cb_cmp = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	//Y,Cr,Cb����ɫ��Χ
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
	//ת����YCrBr
	cvCvtColor(src2, img_YCrCb, CV_RGB2YCrCb);


	//�ָY,Cr,Cb
	cvSplit(img_YCrCb, Y_channel, Cr_channel, Cb_channel, 0);

	//��Y_channel��λ�� Y_lower �� Y_upper ֮���Ԫ�ظ��Ƶ� Y_tmp��
	cvInRangeS(Y_channel, Y_lower, Y_upper, Y_cmp);
	cvInRangeS(Cr_channel, Cr_lower, Cr_upper, Cr_cmp);
	cvInRangeS(Cb_channel, Cb_lower, Cb_upper, Cb_cmp);

	//�ϲ�Y,Cr,Cbͨ����YCrCb��
	cvMerge(Y_cmp, Cr_cmp, Cb_cmp, 0, YCrCb);

	//��ʾ���
	//cvShowImage("YCrCb_mask-����OpenCV���ڸ�������ʶ������̽��", YCrCb);


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

	//����2: �����������ͨ��
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




	//����2: �����������ͨ��
	cvSplit(hsv_image, H_img, S_img, V_img, 0);

	//color_blance();
	//cvMerge(H_img,S_img,V_img,0,hsv_image);
	//cvShowImage( "ɫ��ƽ���", hsv_image);
	//cvShowImage( "Hͨ��", H_img);





	//ֱ��ͼ���⻯(Ч������)
	//cvEqualizeHist(H_img, H_img);

	//cvShowImage( "Hͨ��_���⻯", H_img);

	//����Ӧ
	//cvAdaptiveThreshold(H_img, H_mask, 30, 0, 0, 3, 5);

	//cvShowImage( "Hͨ��", H_img);
	//cvShowImage( "Sͨ��", S_img);
	//cvShowImage( "Vͨ��", V_img);

	//ɫ��
	cvInRangeS(H_img, cvScalar(0, 0, 0, 0), cvScalar(H_low_max, 0, 0, 0), H_mask);//��ɫ��
	cvInRangeS(H_img, cvScalar(256 - H_high_min, 0, 0, 0), cvScalar(256, 0, 0, 0), H_mask1);//��ɫ��

	//���Ͷ�
	cvInRangeS(S_img, cvScalar(S_low, 0, 0, 0), cvScalar(S_high, 0, 0, 0), S_mask); //�м���
	//cvInRangeS(S_img,cvScalar(20,0,0,0),cvScalar(100,0,0,0),S_mask1); //�ͱ��Ͷ�



	//����
	cvInRangeS(V_img, cvScalar(V_high, 0, 0, 0), cvScalar(256, 0, 0, 0), V_mask);//������
	cvInRangeS(V_img, cvScalar(V_low, 0, 0, 0), cvScalar(V_high, 0, 0, 0), V_mask1); //�м���
	//cvInRangeS(V_img,cvScalar(150,0,0,0),cvScalar(250,0,0,0),V_mask2); //������


	//���, �����ϵĻ��
	cvOr(H_mask1, H_mask, H_mask, 0);//������������а�λ�������

	//�������Ͷȹ�������
	cvAnd(H_mask, S_mask, H_mask, 0);//������������а�λ�����

	//cvShowImage( "���Ͷȹ���", H_mask);

	//��ȥ������������
	cvAnd(H_mask, V_mask1, H_mask, 0);//������������а�λ�����

	//cvShowImage( "���ȹ���", H_mask);

	//cvShowImage( "hsv_msk", H_mask);



	//������������
	if (if_high_light){ cvOr(H_mask, V_mask, H_mask, 0); }

	//cvShowImage( "�����߹�", H_mask);

	//cvShowImage( "�ع���� V", V_mask);
	//cvShowImage( "�عⲹ��", S_mask);


	//�Ƿ񲹳��ع����
	hsv_mask = H_mask;

	//cvShowImage( "hsv_msk", H_mask);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////


//��ֵ��
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


	cvShowImage("��ֵǰ", thd_src);

	cvShowImage("��ֵ��1", thd_dst1);
	cvCreateTrackbar("thd_max", "��ֵ��1", &thd_max, 256, 0);
	cvShowImage("��ֵ��2", thd_dst2);
	cvCreateTrackbar("thd_val", "��ֵ��2", &thd_val, 256, 0);

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resizeSrc()
{
	double scale = 0.5;


	//���ͼ���С
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

	cvMorphologyEx(hsv_mask, smooth1, 0, NULL, CV_MOP_CLOSE, MopEx_value);//��ΪNULL Defualt 3*3 �ṹԪ��
	//cvMorphologyEx(smooth1, smooth2, 0, CV_SHAPE_RECT, CV_MOP_OPEN, 1);

	//cvShowImage("���Ÿ�ʴ-����OpenCV���ڸ�������ʶ������̽��", smooth1);

	//cvSmooth(smooth2, smooth3, CV_MEDIAN, 9, 0, 0, 0);

	//cvShowImage( "ƽ��", smooth3);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//����ƥ��

IplImage*    g_image = NULL;
IplImage*    g_gray = NULL;
int        g_thresh = 100;
CvMemStorage*  g_storage = NULL;
CvMemStorage*  g_storage1 = NULL;
CvMemStorage*  g_storage2 = NULL;

CvSeq* seqMidObj = 0;//��ѡ�����������
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

	cvFindContours(dst, g_storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);//ֻ����������
	contoursHead = contours;//contours��ͷ
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

	if (contours)cvDrawContours(tmp3, contours, cvScalarAll(255), cvScalar(255, 0, 0, 0), 1);//��������
	//if( contours )cvDrawContours( tmp2, contours, cvScalar(255,0,0,0),cvScalar(255,100,0,0),1);//��������

	cvShowImage( "��������-����OpenCV���ڸ�������ʶ������̽��", tmp3);

	//CvSeq* seqMidObj = 0;// = cvCreateSeq(CV_SEQ_ELTYPE_CODE, sizeof(CvSeq), sizeof(int),g_storage2);


	//ȥ���봰���ڽӵ�����
	contours = contoursHead; i = 0;
	CvRect bound;
	int dat = 2;

	//ȥ��С�������
	int contArea = 0;
	int imgArea = newSize.height * newSize.width;

	for (; contours != 0; contours = contours->h_next){

		i++;

		//��������С, ���ų�
		contArea = fabs(cvContourArea(contours, CV_WHOLE_SEQ));

		if ((double)contArea / imgArea < 0.015){ continue; }

		//����߽��봰������, ���ų�
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

		//������������
		q = p;
		//p = cvCloneSeq(contours, g_storage2);
		p = contours;

		if (q == NULL){
			seqMidObj = p;
			//p->h_next = NULL;
			//p->h_prev = NULL;
			//printf("��1��!");
		}
		else{
			q->h_next = p;
			p->h_prev = q;
			//printf("1��!");
		}
		//j++;
		handNum++;

	}

	//printf("�ҵ�����: %d ��   ��ѡ: %d ��\n", i,j);

	
	if (seqMidObj){
		seqMidObj->h_prev = NULL;
		p->h_next = NULL;
	}
	if (handNum > 0)
	{

		//printf("�ҵ���: %d  ", handNum);
		HandArea = fabs(cvContourArea(seqMidObj, CV_WHOLE_SEQ));
		printf("�����: %d  ", HandArea);
	}
	//CvSeq* seqMidObj_head = seqMidObj;

	cvZero(tmp3);
	if (seqMidObj)cvDrawContours(tmp3, seqMidObj, cvScalarAll(255), cvScalar(255, 0, 0, 0), 1);//��������

	//cvShowImage( "����ɸѡ-����OpenCV���ڸ�������ʶ������̽��", tmp3);


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

IplImage*    tmp_img = 0;
CvMemStorage*  storage_tmp = 0;

CvSeq* handT = 0;
CvSeq* handT1 = 0;
CvSeq* handT2 = 0;

int handTNum = 10;//10��ģ��
//int handTNum = 2;//10��ģ��

char *tmp_names[] = { "1.bmp", "2.bmp", "3.bmp", "4.bmp", "5.bmp", "6.bmp", "7.bmp", "8.bmp", "9.bmp", "10.bmp" };
char *num_c[] = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" };


/////////////////////////////////////////////////////

//����ģ�������
void init_hand_template()
{

	storage_tmp = cvCreateMemStorage(0);

	int i = 0;
	for (i = 0; i<handTNum; i++){

		tmp_img = cvLoadImage(tmp_names[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (!tmp_img){
			printf("δ�ҵ��ļ�: %s\n", tmp_names[i]);
			continue;
		}
		//cvShowImage("����ģ��", tmp_img);
		handT1 = handT2;
		cvFindContours(tmp_img, storage_tmp, &handT2, sizeof(CvContour), CV_RETR_EXTERNAL);

		if (handT2){
			printf("����ģ��: %s �ɹ�!\n", tmp_names[i]);
			if (handT1 == NULL){
				printf("�����һ��ģ��!\n");
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

//ģ��ƥ����
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

		//�ҵ�hu����С��ģ��
		if (hu > hutmp){
			hu = hutmp;
			kind = i + 1;
		}

		//printf("%f ", hu);
	}

	//��ʾƥ����
	if (hu<((double)Hmatch_value) / 100){
		printf("ƥ��ģ��: %d (%f)", kind, hu);
		match_num = kind;
		if_match_num = true;
	}
	else{
		if_match_num = false;
	}
}


//ģ��ƥ����
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

		//�ҵ�hu����С��ģ��
		if (i == 4 || i == 9)
		if (hu > hutmp){
			hu = hutmp;
			kind = i+1;
		}

		//printf("%f ", hu);
	}

	//��ʾƥ����
	if (hu<((double)Hmatch_value) / 100){
		printf("ƥ��ģ��: %d (%f)", kind, hu);
		match_num = kind;
		if_match_num = true;
	}
	else{
		if_match_num = false;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//������˹�任
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

	//ǰ��ƽ��
	//cvMorphologyEx(planes[2], planes[2], 0, CV_SHAPE_RECT, CV_MOP_CLOSE, 1);

	cvShowImage("������˹�任-����OpenCV���ڸ�������ʶ������̽��", colorlaplace);

	//smoothType = smoothType == CV_GAUSSIAN ? CV_BLUR : smoothType == CV_BLUR ? CV_MEDIAN : CV_GAUSSIAN;

	cvCvtColor(colorlaplace, colorlaplace, CV_BGR2HSV);

	cvSplit(colorlaplace, planes[0], planes[1], planes[2], 0);

	cvInRangeS(planes[2], cvScalar(ls_low, 0, 0, 0), cvScalar(256, 0, 0, 0), planes[2]);

	//����ƽ��
	//cvMorphologyEx(planes[2], planes[2], 0, CV_SHAPE_RECT, CV_MOP_CLOSE, 1);
	//cvSmooth(planes[2], planes[2], CV_MEDIAN, 3, 0, 0, 0);
	//cvErode(planes[2] ,planes[2], 0, 1);
	cvShowImage("������˹_��Ե��-����OpenCV���ڸ�������ʶ������̽��", planes[2]);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define UP		-1
#define DOWN	1
#define LEFT	1
#define RIGHT	-1


CvPoint hand_center = cvPoint(0, 0); //����Ӿ��������ĵ�
CvPoint hand_center_last = cvPoint(0, 0); //
CvPoint hand_direct_now = cvPoint(0, 0);  //���μ�⵽���ƶ�����
CvPoint hand_direct_last = cvPoint(0, 0); //�ϴμ�⵽���ƶ�����
CvPoint hand_direct = cvPoint(0, 0);      //�ϳɵķ���

//���ֵ��е���ƶ�����
void hand_direction(CvSeq* hand){

	//���û����, ��������
	if (!hand){
		hand_center = cvPoint(0, 0);
		hand_center_last = cvPoint(0, 0);
		hand_direct_now = cvPoint(0, 0);
		hand_direct_last = cvPoint(0, 0);
		hand_direct = cvPoint(0, 0);
		return;
	}

	hand_center_last = hand_center;

	//������ĵ�
	CvRect bound = cvBoundingRect(hand, 0);
	hand_center.x = bound.x + bound.width / 2;
	hand_center.y = bound.y + bound.height / 2;

	if (hand_center_last.x != 0){
		hand_direct_now.x = hand_center.x - hand_center_last.x;
		hand_direct_now.y = hand_center.y - hand_center_last.y;

		if (hand_direct_now.x != 0){

			hand_direct.x = (hand_direct_now.x + hand_direct_last.x) / 2;
			if (hand_direct.x != 0){
				if (Match_Mode != 2) printf("  X �ƶ�: %d ", hand_direct.x);
			}

		}
		if (hand_direct_now.y != 0){

			hand_direct.y = (hand_direct_now.y + hand_direct_last.y) / 2;
			if (hand_direct.y != 0){
				if (Match_Mode != 2) printf("  Y �ƶ�: %d ", hand_direct.y);
			}
		}

	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

//����ʶ����
void hand_draw(IplImage* dst, CvSeq* hands)
{

	if (!hands) return;

	CvRect bound;

	//͹��
	int i, hullcount;
	CvPoint pt0;
	CvSeq* hull;

	CvSeq* handp = hands;

	//͹��ȱ��
	CvConvexityDefect* defect;
	CvSeq* hullDefect;
	//CvSeq* hullDefectSelect;
	int hullDefectNum = 0;
	//cvPoint** points = (cvPoint*)malloc(sizeof(cvPoint)*3);
	//cvPoint points[3];


	//��������
	cvDrawContours(dst, handp, cvScalar(255, 150, 100, 0), cvScalar(255, 0, 0, 0), 1, 1, 8, cvPoint(0, 0));

	CvFont font;


	//���Ƽ�⵽������
	if (if_match_num){

		//����Ӱ
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0f, 1.0f, 0, 5, 8);
		cvPutText(dst, num_c[match_num - 1], cvPoint(5, 30), &font, CV_RGB(255, 255, 255));
		//����ɫ
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0f, 1.0f, 0, 2, 8);
		cvPutText(dst, num_c[match_num - 1], cvPoint(5, 30), &font, CV_RGB(255, 0, 0));


	}

	//�����ƶ�����
	if (1){

		//cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,2,8);
		//cvPutText(dst, "X: ", cvPoint(5, 30), &font, CV_RGB(255,0,0));



	}


	//����ȡ������������
	for (; handp != 0; handp = handp->h_next){

		bound = cvBoundingRect(handp, 0);


		//�󲢻������ĵ�

		//����ɫ
		int det = 2;
		cvRectangle(dst,
			cvPoint(hand_center.x - det, hand_center.y - det),
			cvPoint(hand_center.x + det, hand_center.y + det),
			CV_RGB(0, 0, 0), 3, 8, 0);

		//������
		det = 3;
		cvRectangle(dst,
			cvPoint(hand_center.x - det, hand_center.y - det),
			cvPoint(hand_center.x + det, hand_center.y + det),
			CV_RGB(255, 255, 255), 1, 8, 0);



		//�������緽��
		cvRectangle(dst,
			cvPoint(bound.x, bound.y),
			cvPoint(bound.x + bound.width, bound.y + bound.height),
			cvScalar(0, 0, 255, 0), 2, 8, 0);

		//Ѱ��͹��
		hull = cvConvexHull2(handp, 0, CV_CLOCKWISE, 0);
		hullcount = hull->total;
		//printf("͹������: %d  ",hullcount);

		pt0 = **CV_GET_SEQ_ELEM(CvPoint*, hull, hullcount - 1);

		//��͹��
		for (i = 0; i < hullcount; i++){

			//�õ�͹���ĵ�
			CvPoint pt = **CV_GET_SEQ_ELEM(CvPoint*, hull, i);
			cvLine(dst, pt0, pt, CV_RGB(0, 255, 0), 1, CV_AA, 0);
			pt0 = pt;
		}

		//���ȱ��
		/*if(!cvCheckContourConvexity(hands)){

		hullDefect = cvConvexityDefects(hands, hull, 0);
		hullDefectNum = hullDefect->total;
		printf("ȱ�ݸ���: %d  ",hullDefectNum);
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
	cvNamedWindow("ԭͼ��-����OpenCV���ڸ�������ʶ������̽��", 0);//src

	//cvNamedWindow("���Ÿ�ʴ-����OpenCV���ڸ�������ʶ������̽��", 0);

	//cvNamedWindow("����ʶ��-����OpenCV���ڸ�������ʶ������̽��", 0);

	cvResizeWindow("ԭͼ��-����OpenCV���ڸ�������ʶ������̽��", newSize.width, newSize.height);
	//cvResizeWindow("���Ÿ�ʴ-����OpenCV���ڸ�������ʶ������̽��", newSize.width, newSize.height);
	//cvResizeWindow("����ʶ��-����OpenCV���ڸ�������ʶ������̽��", newSize.width, newSize.height);

	cvNamedWindow("��������-����OpenCV���ڸ�������ʶ������̽��", CV_WINDOW_AUTOSIZE);
	cvResizeWindow("��������-����OpenCV���ڸ�������ʶ������̽��", newSize.width*1.5, 60 * 10);


	cvCreateTrackbar("��ֵƯ���˲�", "��������-����OpenCV���ڸ�������ʶ������̽��", &pmsf_value, 20, 0);
	cvCreateTrackbar("�����㽵��", "��������-����OpenCV���ڸ�������ʶ������̽��", &MopEx_value, 5, 0);

	cvCreateTrackbar("ɫ������", "��������-����OpenCV���ڸ�������ʶ������̽��", &H_low_max, 150, 0);
	cvCreateTrackbar("ɫ��������", "��������-����OpenCV���ڸ�������ʶ������̽��", &H_high_min, 150, 0);

	cvCreateTrackbar("��������", "��������-����OpenCV���ڸ�������ʶ������̽��", &V_low, 100, 0);
	cvCreateTrackbar("��������", "��������-����OpenCV���ڸ�������ʶ������̽��", &V_high, 256, 0);

	cvCreateTrackbar("���Ͷ�����", "��������-����OpenCV���ڸ�������ʶ������̽��", &S_low, 100, 0);
	cvCreateTrackbar("���Ͷ�����", "��������-����OpenCV���ڸ�������ʶ������̽��", &S_high, 255, 0);

	cvCreateTrackbar("�߹ⲹ��", "��������-����OpenCV���ڸ�������ʶ������̽��", &if_high_light, 1, 0);

	cvCreateTrackbar("matchϵ��", "��������-����OpenCV���ڸ�������ʶ������̽��", &Hmatch_value, 50, 0);

	cvCreateTrackbar("ʶ��ģʽ", "��������-����OpenCV���ڸ�������ʶ������̽��", &Match_Mode, 5, 0);


}

//�������
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
//�����Ҫʹ�ûص������ķ�ʽ���ͼ�����ݣ���ע�ͺ궨��USE_CALLBACK_GRAB_IMAGE.
//���ǵ�SDKͬʱ֧�ֻص��������������ýӿ�ץȡͼ��ķ�ʽ�����ַ�ʽ��������"�㿽��"���ƣ������ĳ̶ȵĽ���ϵͳ���ɣ���߳���ִ��Ч�ʡ�
//��������ץȡ��ʽ�Ȼص������ķ�ʽ�������������ó�ʱ�ȴ�ʱ��ȣ����ǽ�����ʹ�� uiDisplayThread �еķ�ʽ
//*/
////#define USE_CALLBACK_GRAB_IMAGE 
//
//#ifdef USE_CALLBACK_GRAB_IMAGE
///*ͼ��ץȡ�ص�����*/
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
//	//����õ�ԭʼ����ת����RGB��ʽ�����ݣ�ͬʱ����ISPģ�飬��ͼ����н��룬������������ɫУ���ȴ���
//	//�ҹ�˾�󲿷��ͺŵ������ԭʼ���ݶ���Bayer��ʽ��
//	status = CameraImageProcess(hCamera, pFrameBuffer, m_pFrameBuffer,pFrameHead);
//
//	//�ֱ��ʸı��ˣ���ˢ�±���
//	if (m_sFrInfo.iWidth != pFrameHead->iWidth || m_sFrInfo.iHeight != pFrameHead->iHeight)
//	{
//		m_sFrInfo.iWidth = pFrameHead->iWidth;
//		m_sFrInfo.iHeight = pFrameHead->iHeight;
//	}
//
//	if(status == CAMERA_STATUS_SUCCESS )
//	{
//		//����SDK��װ�õ���ʾ�ӿ�����ʾͼ��,��Ҳ���Խ�m_pFrameBuffer�е�RGB����ͨ��������ʽ��ʾ������directX,OpengGL,�ȷ�ʽ��
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
///*ͼ��ץȡ�̣߳���������SDK�ӿں�����ȡͼ��*/
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
//		//����õ�ԭʼ����ת����RGB��ʽ�����ݣ�ͬʱ����ISPģ�飬��ͼ����н��룬������������ɫУ���ȴ���
//		//�ҹ�˾�󲿷��ͺŵ������ԭʼ���ݶ���Bayer��ʽ��
//		status = CameraImageProcess(hCamera, pbyBuffer, m_pFrameBuffer, &sFrameInfo);//����ģʽ
//
//		//�ֱ��ʸı��ˣ���ˢ�±���
//		if (m_sFrInfo.iWidth != sFrameInfo.iWidth || m_sFrInfo.iHeight != sFrameInfo.iHeight)
//		{
//			m_sFrInfo.iWidth = sFrameInfo.iWidth;
//			m_sFrInfo.iHeight = sFrameInfo.iHeight;
//			//ͼ���С�ı䣬֪ͨ�ػ�
//		}
//
//		if (status == CAMERA_STATUS_SUCCESS)
//		{
//			//����SDK��װ�õ���ʾ�ӿ�����ʾͼ��,��Ҳ���Խ�m_pFrameBuffer�е�RGB����ͨ��������ʽ��ʾ������directX,OpengGL,�ȷ�ʽ��
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
//			//cvPyrMeanShiftFiltering(src, src2, pmsf_value, 40, 2);//�ָ� �C ��ֵƯ���˲�
//			//������ͷʧ��
//			//if (!src){ return NO_CAMERA_BUG; }
//			sz = cvGetSize(src);
//			newSize.height = (int)(sz.height * scale);
//			newSize.width = (int)(sz.width * scale);
//
//			sz = newSize;
//
//			//�������д���
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
//			//����ƥ���ģ��
//			init_hand_template();
//
//			src1 = cvCreateImage(sz, IPL_DEPTH_8U, 3);
//
//			m_iDispFrameNum++;
//		}
//
//		//�ڳɹ�����CameraGetImageBuffer�󣬱������CameraReleaseImageBuffer���ͷŻ�õ�buffer��
//		//�����ٴε���CameraGetImageBufferʱ�����򽫱�����֪�������߳��е���CameraReleaseImageBuffer���ͷ���buffer
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
//			//����õ�ԭʼ����ת����RGB��ʽ�����ݣ�ͬʱ����ISPģ�飬��ͼ����н��룬������������ɫУ���ȴ���
//			//�ҹ�˾�󲿷��ͺŵ������ԭʼ���ݶ���Bayer��ʽ��
//			status = CameraImageProcess(hCamera, pbyBuffer, m_pFrameBuffer,&sFrameInfo);//����ģʽ
//
//			//�ֱ��ʸı��ˣ���ˢ�±���
//			if (m_sFrInfo.iWidth != sFrameInfo.iWidth || m_sFrInfo.iHeight != sFrameInfo.iHeight)
//			{
//				m_sFrInfo.iWidth = sFrameInfo.iWidth;
//				m_sFrInfo.iHeight = sFrameInfo.iHeight;
//				//ͼ���С�ı䣬֪ͨ�ػ�
//			}
//
//			if(status == CAMERA_STATUS_SUCCESS)
//			{
//				//����SDK��װ�õ���ʾ�ӿ�����ʾͼ��,��Ҳ���Խ�m_pFrameBuffer�е�RGB����ͨ��������ʽ��ʾ������directX,OpengGL,�ȷ�ʽ��
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
//				cvResize(src, src1, CV_INTER_LINEAR);		//�ߴ�任
//				//cvShowImage("src", src1);
//				cvPyrMeanShiftFiltering(src1, src2, pmsf_value, 40, 2);//�ָ� �C ��ֵƯ���˲�
//				//cvShowImage("��ֵƯ���˲�-����OpenCV���ڸ�������ʶ������̽��", src2);
//				hand_YCrCb();//src
//				hand_HSV();		//����HSVͨ��
//				reduce_noise();		//����
//				hand_contours(smooth1);//Ѱ��������
//				hand_template_match(handT, seqMidObj);//Ѱ��ƥ��
//
//				hand_direction(seqMidObj);//���ֵ��ƶ�����
//				//���ƶ�����Ϣ
//				cvZero(tmp2);
//				hand_draw(tmp2, seqMidObj); //�����ڼ�ⴰ��
//				hand_draw(src1, seqMidObj); //������ԭ����
//
//
//				//cvShowImage( "���Ÿ�ʴ", smooth1);
//				cvShowImage("����ʶ��-����OpenCV���ڸ�������ʶ������̽��", tmp2);
//				cvShowImage("ԭͼ��-����OpenCV���ڸ�������ʶ������̽��", src1);
//				//
//				//�������(��ȭͷ״, ���������ƶ�)
//				if (match_num == 10)// cvNamedWindow("xxx", 0);
//			control_mouse(hand_direct.x, hand_direct.y);
//		
//				m_iDispFrameNum++;
//			}    
//
//			//�ڳɹ�����CameraGetImageBuffer�󣬱������CameraReleaseImageBuffer���ͷŻ�õ�buffer��
//			//�����ٴε���CameraGetImageBufferʱ�����򽫱�����֪�������߳��е���CameraReleaseImageBuffer���ͷ���buffer
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
//	//ö���豸������豸�б�
//	iCameraNums = 10;//����CameraEnumerateDeviceǰ��������iCameraNums = 10����ʾ���ֻ��ȡ10���豸�������Ҫö�ٸ�����豸�������sCameraList����Ĵ�С��iCameraNums��ֵ
//
//	if (CameraEnumerateDevice(sCameraList,&iCameraNums) != CAMERA_STATUS_SUCCESS || iCameraNums == 0)
//	{
//		printf("No camera was found!");
//		return FALSE;
//	}
//
//	//��ʾ���У�����ֻ����������һ���������ˣ�ֻ��ʼ����һ�������(-1,-1)��ʾ�����ϴ��˳�ǰ����Ĳ���������ǵ�һ��ʹ�ø�����������Ĭ�ϲ���.
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
//	CameraGetCapability(m_hCamera,&sCameraInfo);//"��ø��������������"
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
//		g_CameraName, NULL, NULL, 0);//"֪ͨSDK�ڲ��������������ҳ��";
//
//
///////////////////////////////////////////////////////////////////////////
//	
//
//	////���µ���ͼ���С
//	////resizeSrc();
//
//	////�������д���
//	//resizeAllWindow();
//
//	//init_hand_YCrCb();
//	//init_hand_HSV();
//	//init_laplace();
//	//inti_threshold();
//
//	////����ƥ���ģ��
//	//init_hand_template();
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//#ifdef USE_CALLBACK_GRAB_IMAGE //���Ҫʹ�ûص�������ʽ������USE_CALLBACK_GRAB_IMAGE�����
//	//Set the callback for image capture
//	CameraSetCallbackFunction(m_hCamera,GrabImageCallback,NULL,NULL);//"����ͼ��ץȡ�Ļص�����";
//#else
//	m_hDispThread = (HANDLE)_beginthreadex(NULL, 0, &uiDisplayThread, (PVOID)m_hCamera, 0,  &m_threadID);
//#endif
//
//	//CameraPlay(m_hCamera);
//	
//	CameraShowSettingPage(m_hCamera,TRUE);//TRUE��ʾ������ý��档FALSE�����ء�
//
//	//unsigned char* capture=CameraGetImageBufferEx(m_hCamera, P_Width, P_Height, 10);
//	//IplImage* img = 
//	//cvNamedWindow("Example1", CV_WINDOW_AUTOSIZE);
//	//cvShowImage("Example1", img);
//
//	//src = cvQueryFrame(capture);
//
//	//cvPyrMeanShiftFiltering(img, src2, pmsf_value, 40, 2);//�ָ� �C ��ֵƯ���˲�
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

	//������ͷ
	CvCapture* capture = cvCaptureFromCAM(0);
	src = cvQueryFrame(capture);

	//������ͷʧ��
	if (!src){ return NO_CAMERA_BUG; }

	//���ͼ���С
	sz = cvGetSize(src);
	newSize.height = (int)(sz.height * scale);
	newSize.width = (int)(sz.width * scale);

	sz = newSize;


	//���µ���ͼ���С
	//resizeSrc();

	//�������д���
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

	//����ƥ���ģ��
	init_hand_template();

	src1 = cvCreateImage(sz, IPL_DEPTH_8U, 3);

	/////////////////////��ʼѭ��///////////////////////////////
	while (c != 27)

	{

		//��ǰͼ��
		src = cvQueryFrame(capture);
		if (!src) return NO_CAMERA_BUG;

		//��СҪ�����ͼ��(��С������)
		cvResize(src, src1, CV_INTER_LINEAR);		//�ߴ�任

		//cvShowImage("src", src1);

		cvPyrMeanShiftFiltering(src1, src2, pmsf_value, 40, 2);//�ָ� �C ��ֵƯ���˲�
		
		//cvSmooth(src1, tmp4, CV_MEDIAN, 9, 0, 0, 0);//src ����ͼ��  tmp4���

		//cvShowImage("��ֵƯ���˲�-����OpenCV���ڸ�������ʶ������̽��", src2);

		//������˹�任
		//toLaplace(src2);

		hand_YCrCb();//src  //�ֵ�����: H(0-30),S(30-170),V(0-200)

		hand_HSV();		//����HSVͨ��

		reduce_noise();		//����

		hand_contours(smooth1);//Ѱ��������
		if (Match_Mode == 0)		
			hand_template_match(handT, seqMidObj);//Ѱ��ƥ��
		else if (Match_Mode == 1 || Match_Mode == 2|| Match_Mode==3 )
			hand_template_match2(handT, seqMidObj);//Ѱ��ƥ��

		hand_direction(seqMidObj);//���ֵ��ƶ�����
		//���ƶ�����Ϣ
		cvZero(tmp2);
		hand_draw(tmp2, seqMidObj); //�����ڼ�ⴰ��
		hand_draw(src1, seqMidObj); //������ԭ����


		//cvShowImage( "���Ÿ�ʴ", smooth1);
		//cvShowImage("����ʶ��-����OpenCV���ڸ�������ʶ������̽��", tmp2);
		cvShowImage("ԭͼ��-����OpenCV���ڸ�������ʶ������̽��", src1);

		//�������(��ȭͷ״, ���������ƶ�)
		if (Match_Mode == 1)
		{

			if (match_num == 10)// cvNamedWindow("xxx", 0);
			{
				if (hand_direct.x >= 10 && hand_direct.y <= abs(hand_direct.x / 2) && hand_direct.y >= -abs(hand_direct.x/2) && DIR_flag == 0)	Left_count++;			//�ϳɵķ���
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
					keybd_event(39, 0, KEYEVENTF_KEYUP, 0);	//������
					printf("����");
					Beep(1800, 50);
				}
				if (Left_count >= 2)
				{
					DIR_flag = 1;
					Left_count = 0;
					keybd_event(37, 0, 0, 0);
					keybd_event(37, 0, KEYEVENTF_KEYUP, 0);	//������
					printf("����");
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
				printf("\n***************����ģʽ******************\n");
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
					printf("�Ŵ�");
				}
				else if (HandArea_err <= -1000)
				{
					SIZE_flag--;
					Set_HandArea = HandArea;
					keybd_event(109, 0, 0, 0);
					keybd_event(109, 0, KEYEVENTF_KEYUP, 0);	//��
					printf("��С");
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