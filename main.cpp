// particle_tracking.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Rect select;
bool select_flag = false;
bool tracking = false;//���ٱ�־λ
bool select_show = false;
Point origin;
Mat frame, hsv;
int after_select_frames = 0;//ѡ�������������֡����

							/****rgb�ռ��õ��ı���****/
							//int hist_size[]={16,16,16};//rgb�ռ��ά�ȵ�bin����
							//float rrange[]={0,255.0};
							//float grange[]={0,255.0};
							//float brange[]={0,255.0};
							//const float *ranges[] ={rrange,grange,brange};//range�൱��һ����ά����ָ��

							/****hsv�ռ��õ��ı���****/
int hist_size[] = { 16,16,16 };
float hrange[] = { 0,180.0 };
float srange[] = { 0,256.0 };
float vrange[] = { 0,256.0 };

//int hist_size[]={32,32,32};
//float hrange[]={0,359.0.0};
//float srange[]={0,1.0};
//float vrange[]={0,1.0};
const float *ranges[] = { hrange,srange,vrange };

int channels[] = { 0,1,2 };

/****�й����Ӵ��ڱ仯�õ�����ر���****/
int A1 = 2;
int A2 = -1;
int B0 = 1;
double sigmax = 1.0;
double sigmay = 0.5;
double sigmas = 0.001;

/****����ʹ��������Ŀ��****/
#define PARTICLE_NUMBER 100 //���������趨̫�󣬾�����������ֳ���25�ͻᱨ����������ʱ������ִ���

/****�������ӽṹ��****/
typedef struct particle
{
	int orix, oriy;//ԭʼ��������
	int x, y;//��ǰ���ӵ�����
	double scale;//��ǰ���Ӵ��ڵĳߴ�
	int prex, prey;//��һ֡���ӵ�����
	double prescale;//��һ֡���Ӵ��ڵĳߴ�
	Rect rect;//��ǰ���Ӿ��δ���
	Mat hist;//��ǰ���Ӵ���ֱ��ͼ����
	double weight;//��ǰ����Ȩֵ
}PARTICLE;

PARTICLE particles[PARTICLE_NUMBER];

/************************************************************************************************************************/
/**** ����������onMouse()�����Ļ�������Ի�������϶����ο��4������ ****/
/************************************************************************************************************************/
void onMouse(int event, int x, int y, int, void*)
{
	//Point origin;//����������ط����ж��壬��Ϊ���ǻ�����Ϣ��Ӧ�ĺ�����ִ�����origin���ͷ��ˣ����Դﲻ��Ч����
	if (select_flag)
	{
		select.x = MIN(origin.x, x);//��һ��Ҫ����굯��ż�����ο򣬶�Ӧ������갴�¿�ʼ���������ʱ��ʵʱ������ѡ���ο�
		select.y = MIN(origin.y, y);
		select.width = abs(x - origin.x);//����ο�Ⱥ͸߶�
		select.height = abs(y - origin.y);
		select &= Rect(0, 0, frame.cols, frame.rows);//��֤��ѡ���ο�����Ƶ��ʾ����֮��

													 // rectangle(frame,select,Scalar(0,0,255),3,8,0);//��ʾ�ֶ�ѡ��ľ��ο�
	}
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		select_flag = true;//��갴�µı�־����ֵ
		tracking = false;
		select_show = true;
		after_select_frames = 0;//��û��ʼѡ�񣬻������¿�ʼѡ�񣬼���Ϊ0
		origin = Point(x, y);//�������������ǲ�׽���ĵ�
		select = Rect(x, y, 0, 0);//����һ��Ҫ��ʼ������Ϊ��opencv��Rect���ο����ڵĵ��ǰ������Ͻ��Ǹ���ģ����ǲ������½��Ǹ��㡣
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		select_flag = false;
		tracking = true;
		select_show = false;
		after_select_frames = 1;//ѡ��������һ֡������1֡
	}
}

/****����Ȩֵ�������к���****/
int particle_decrease(const void *p1, const void *p2)
{
	PARTICLE* _p1 = (PARTICLE*)p1;
	PARTICLE* _p2 = (PARTICLE*)p2;
	if (_p1->weight<_p2->weight)
		return 1;
	else if (_p1->weight>_p2->weight)
		return -1;
	return 0;//��ȵ�����·���0
}

int main()
{
	char c;
	Mat target_img, track_img;
	Mat target_hist, track_hist;
	PARTICLE *pParticle;

	/***������ͷ****/
	VideoCapture cam(0);
	if (!cam.isOpened())
		return -1;

	/****��ȡһ֡ͼ��****/
	cam >> frame;
	if (frame.empty())
		return -1;

	VideoWriter output_dst("demo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, frame.size(), 1);

	/****��������****/
	namedWindow("camera", 1);//��ʾ��Ƶԭͼ��Ĵ���

							 /****��׽���****/
	setMouseCallback("camera", onMouse, 0);

	while (1)
	{
		/****��ȡһ֡ͼ��****/
		cam >> frame;
		if (frame.empty())
			return -1;
		cv::flip(frame, frame, 1);
		/****��rgb�ռ�ת��Ϊhsv�ռ�****/
		cvtColor(frame, hsv, CV_BGR2HSV);

		if (tracking)
		{

			if (1 == after_select_frames)//ѡ����Ŀ�������
			{
				/****����Ŀ��ģ���ֱ��ͼ����****/
				target_img = Mat(hsv, select);//�ڴ�֮ǰ�ȶ����target_img,Ȼ��������ֵҲ�У�Ҫѧ��Mat���������
				calcHist(&target_img, 1, channels, Mat(), target_hist, 3, hist_size, ranges);
				normalize(target_hist, target_hist);

				/****��ʼ��Ŀ������****/
				pParticle = particles;//ָ���ʼ��ָ��particles����
				for (int x = 0; x<PARTICLE_NUMBER; x++)
				{
					pParticle->x = cvRound(select.x + 0.5*select.width);//ѡ��Ŀ����ο�����Ϊ��ʼ���Ӵ�������
					pParticle->y = cvRound(select.y + 0.5*select.height);
					pParticle->orix = pParticle->x;//���ӵ�ԭʼ����Ϊѡ�����ο�(��Ŀ��)������
					pParticle->oriy = pParticle->y;
					pParticle->prex = pParticle->x;//������һ�ε�����λ��
					pParticle->prey = pParticle->y;
					pParticle->rect = select;
					pParticle->prescale = 1;
					pParticle->scale = 1;
					pParticle->hist = target_hist;
					pParticle->weight = 0;
					pParticle++;
				}
			}
			else if (2 == after_select_frames)//�ӵڶ�֡��ʼ�Ϳ��Կ�ʼ������
			{
				double sum = 0.0;
				pParticle = particles;
				RNG rng;//�����������

						/****�������ӽṹ��Ĵ󲿷ֲ���****/
				for (int i = 0; i<PARTICLE_NUMBER; i++)
				{
					int x, y;
					int xpre, ypre;
					double s, pres;

					xpre = pParticle->x;
					ypre = pParticle->y;
					pres = pParticle->scale;

					/****�������ӵľ���������������****/
					x = cvRound(A1*(pParticle->x - pParticle->orix) + A2*(pParticle->prex - pParticle->orix) +
						B0*rng.gaussian(sigmax) + pParticle->orix);
					pParticle->x = max(0, min(x, frame.cols - 1));

					y = cvRound(A1*(pParticle->y - pParticle->oriy) + A2*(pParticle->prey - pParticle->oriy) +
						B0*rng.gaussian(sigmay) + pParticle->oriy);
					pParticle->y = max(0, min(y, frame.rows - 1));

					s = A1*(pParticle->scale - 1) + A2*(pParticle->prescale - 1) + B0*(rng.gaussian(sigmas)) + 1.0;
					pParticle->scale = max(1.0, min(s, 3.0));

					pParticle->prex = xpre;
					pParticle->prey = ypre;
					pParticle->prescale = pres;
					// pParticle->orix=pParticle->orix;
					// pParticle->oriy=pParticle->oriy;

					//ע����c�����У�x-1.0�����x��int�ͣ�������﷨�д���,�����ǰ�����cvRound(x-0.5)������ȷ��
					pParticle->rect.x = max(0, min(cvRound(pParticle->x - 0.5*pParticle->scale*pParticle->rect.width), frame.cols));
					pParticle->rect.y = max(0, min(cvRound(pParticle->y - 0.5*pParticle->scale*pParticle->rect.height), frame.rows));
					pParticle->rect.width = min(cvRound(pParticle->rect.width), frame.cols - pParticle->rect.x);
					pParticle->rect.height = min(cvRound(pParticle->rect.height), frame.rows - pParticle->rect.y);
					// pParticle->rect.width=min(cvRound(pParticle->scale*pParticle->rect.width),frame.cols-pParticle->rect.x);
					// pParticle->rect.height=min(cvRound(pParticle->scale*pParticle->rect.height),frame.rows-pParticle->rect.y);

					/****��������������µ�ֱ��ͼ����****/
					track_img = Mat(hsv, pParticle->rect);
					calcHist(&track_img, 1, channels, Mat(), track_hist, 3, hist_size, ranges);
					normalize(track_hist, track_hist);

					/****�������ӵ�Ȩֵ****/
					// pParticle->weight=compareHist(target_hist,track_hist,CV_COMP_INTERSECT);
					//���ð���ϵ���������ƶ�,��Զ���ʼ����һĿ��֡��Ƚ�
					pParticle->weight = 1.0 - compareHist(target_hist, track_hist, CV_COMP_BHATTACHARYYA);
					/****�ۼ�����Ȩֵ****/
					sum += pParticle->weight;
					pParticle++;
				}

				/****��һ������Ȩ��****/
				pParticle = particles;
				for (int i = 0; i<PARTICLE_NUMBER; i++)
				{
					pParticle->weight /= sum;
					pParticle++;
				}

				/****�������ӵ�Ȩֵ��������****/
				pParticle = particles;
				qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);

				/****��������Ȩ���ز�������****/
				PARTICLE newParticle[PARTICLE_NUMBER];
				int np = 0, k = 0;
				for (int i = 0; i<PARTICLE_NUMBER; i++)
				{
					np = cvRound(pParticle->weight*PARTICLE_NUMBER);
					for (int j = 0; j<np; j++)
					{
						newParticle[k++] = particles[i];
						if (k == PARTICLE_NUMBER)
							goto EXITOUT;
					}
				}
				while (k<PARTICLE_NUMBER)
					newParticle[k++] = particles[0];
			EXITOUT:
				for (int i = 0; i<PARTICLE_NUMBER; i++)
					particles[i] = newParticle[i];
			}//end else

			 //????????������������������һ��Ϳ�
			 // qsort(pParticle,PARTICLE_NUMBER,sizeof(PARTICLE),&particle_decrease);

			 /****��������������������������λ�õ�����ֵ��Ϊ���ٽ��****/
			 /*Rect_<double> rectTrackingTemp(0.0,0.0,0.0,0.0);
			 pParticle=particles;
			 for(int i=0;i<PARTICLE_NUMBER;i++)
			 {
			 rectTrackingTemp.x+=pParticle->rect.x*pParticle->weight;
			 rectTrackingTemp.y+=pParticle->rect.y*pParticle->weight;
			 rectTrackingTemp.width+=pParticle->rect.width*pParticle->weight;
			 rectTrackingTemp.height+=pParticle->rect.height*pParticle->weight;
			 pParticle++;
			 }*/


			 /****�������Ȩ��Ŀ�������λ�ã���Ϊ���ٽ��****/
			Rect rectTrackingTemp(0, 0, 0, 0);
			pParticle = particles;
			rectTrackingTemp.x = pParticle->x - 0.5*pParticle->rect.width;
			rectTrackingTemp.y = pParticle->y - 0.5*pParticle->rect.height;
			rectTrackingTemp.width = pParticle->rect.width;
			rectTrackingTemp.height = pParticle->rect.height;



			/****�������Ȩ��Ŀ�������λ�ã�����Ȩֵ����1/4����������Ϊ���ٽ��****/
			/*Rect rectTrackingTemp(0,0,0,0);
			double weight_temp=0.0;
			pParticle=particles;
			for(int i=0;i<PARTICLE_NUMBER/4;i++)
			{
			weight_temp+=pParticle->weight;
			pParticle++;
			}
			pParticle=particles;
			for(int i=0;i<PARTICLE_NUMBER/4;i++)
			{
			pParticle->weight/=weight_temp;
			pParticle++;
			}
			pParticle=particles;
			for(int i=0;i<PARTICLE_NUMBER/4;i++)
			{
			rectTrackingTemp.x+=pParticle->rect.x*pParticle->weight;
			rectTrackingTemp.y+=pParticle->rect.y*pParticle->weight;
			rectTrackingTemp.width+=pParticle->rect.width*pParticle->weight;
			rectTrackingTemp.height+=pParticle->rect.height*pParticle->weight;
			pParticle++;
			}*/


			/****�������Ȩ��Ŀ�������λ�ã�����������������Ϊ���ٽ��****/
			/*Rect rectTrackingTemp(0,0,0,0);
			pParticle=particles;
			for(int i=0;i<PARTICLE_NUMBER;i++)
			{
			rectTrackingTemp.x+=cvRound(pParticle->rect.x*pParticle->weight);
			rectTrackingTemp.y+=cvRound(pParticle->rect.y*pParticle->weight);
			pParticle++;
			}
			pParticle=particles;
			rectTrackingTemp.width = pParticle->rect.width;
			rectTrackingTemp.height = pParticle->rect.height;*/


			//����Ŀ���������
			Rect tracking_rect(rectTrackingTemp);

			pParticle = particles;

			/****��ʾ�������˶����****/
			for (int m = 0; m<PARTICLE_NUMBER; m++)
			{
				rectangle(frame, pParticle->rect, Scalar(255, 0, 0), 1, 8, 0);
				pParticle++;
			}

			/****��ʾ���ٽ��****/
			rectangle(frame, tracking_rect, Scalar(0, 0, 255), 3, 8, 0);

			after_select_frames++;//��ѭ��ÿѭ��һ�Σ�������1
			if (after_select_frames>2)//��ֹ����̫����after_select_frames�������
				after_select_frames = 2;
		}

		if (select_show)
			rectangle(frame, select, Scalar(0, 0, 255), 3, 8, 0);//��ʾ�ֶ�ѡ��ľ��ο�
		output_dst << frame;
		//��ʾ��ƵͼƬ������
		imshow("camera", frame);

		// select.zeros();
		//������Ӧ
		c = (char)waitKey(20);
		if (27 == c)//ESC��
			return -1;
	}

	return 0;
}