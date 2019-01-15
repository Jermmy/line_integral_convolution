//
//  lic.hpp
//  LineIntegralConvolution
//
//  Created by xyz on 2017/2/2.
//  Copyright © 2017年 xyz. All rights reserved.
//

#ifndef lic_hpp
#define lic_hpp

#include <stdio.h>
#include <cmath>
#include <cstdlib>

#include <opencv2/opencv.hpp>
using namespace cv;

class Lic {
private:
    const int SQUARE_FLOW_FIELD_SZ = 400;
    const int DISCRETE_FILTER_SIZE = 2048;
    const float LOWPASS_FILTER_LENGTH = 32.000f;
    const float LINE_SQUARE_CLIP_MAX = 100000.0f;
    const float VECTOR_COMPONENT_MIN = 0.0500f;
    // 白噪音
    Mat pNoise;
    // 卷积核
    float *pLUT0, *pLUT1;
    
private:
    void makeWhiteNoise(int row, int col);
    void genBoxFilterLUT(int size);
    Mat normalizeVect(const Mat &pVect);
    Mat flowImagingLIC(const Mat &pVectr, float krnlen);
public:
    Lic();
    ~Lic();
    Mat showLIC(const Mat &pVectr);
};

Lic::Lic() {
    genBoxFilterLUT(DISCRETE_FILTER_SIZE);
}

Lic::~Lic() {
    pNoise.release();
    free(pLUT0);  pLUT0 = NULL;
    free(pLUT1);  pLUT1 = NULL;
}

Mat Lic::showLIC(const cv::Mat &pVectr) {
    assert(pVectr.channels() == 2);
    int row = pVectr.rows, col = pVectr.cols;
    makeWhiteNoise(row, col);
    
    // imwrite("white_noise.jpg", pNoise);
    
    Mat normVect = normalizeVect(pVectr);
    
    return flowImagingLIC(normVect, LOWPASS_FILTER_LENGTH);
}

Mat Lic::normalizeVect(const cv::Mat &pVect) {
    assert(pVect.type() == CV_32FC2);
    int row = pVect.rows, col = pVect.cols;
    Mat ret(row, col, CV_32FC2);
    float vcMag = 0.0f, scale = 0.0f, vecX = 0.0f, vecY = 0.0f;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            vecX = pVect.at<Vec2f>(i, j)[0];
            vecY = pVect.at<Vec2f>(i, j)[1];
            vcMag = float(sqrt(vecX*vecX + vecY*vecY));
            scale = (vcMag < 0.001f) ? 0.0f : 1.0f / vcMag;
            vecX *= scale;
            vecY *= scale;
            ret.at<Vec2f>(i, j)[0] = vecX;
            ret.at<Vec2f>(i, j)[1] = vecY;
        }
    }
    return ret;
}


void Lic::genBoxFilterLUT(int size) {
    pLUT0 = (float*)malloc(sizeof(float) * DISCRETE_FILTER_SIZE);
    pLUT1 = (float*)malloc(sizeof(float) * DISCRETE_FILTER_SIZE);
    for (int i = 0; i < DISCRETE_FILTER_SIZE; i++) {
        pLUT0[i] = pLUT1[i] = (float)i;
    }
}


void Lic::makeWhiteNoise(int row, int col) {
    pNoise = Mat::zeros(row, col, CV_8U);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int r = rand();
            r = ( (r & 0xff) + ( (r & 0xff00) >> 8 ) ) & 0xff;
            pNoise.at<uchar>(i, j) = (unsigned char)r;
        }
    }
}

Mat Lic::flowImagingLIC(const cv::Mat &pVectr, float krnlen) {
    assert(pVectr.type() == CV_32FC2);
    
    int row = pVectr.rows, col = pVectr.cols;
    
    Mat pImage(row, col, CV_8U, Scalar(0));
    
    int advcts = 0;   // 追踪的步数
    int ADVCTS = int(krnlen * 3);
    
    float vctrX = 0;   // 速度x分量
    float vctrY = 0;   // 速度y分量
    float clp0X = 0;   // 当前点x坐标
    float clp0Y = 0;   // 当前点y坐标
    float clp1X = 0;   //下一点x坐标
    float clp1Y = 0;   //下一点y坐标
    float sampX = 0;   //采样点x坐标
    float sampY = 0;   //采样点y坐标
    float tmpLen = 0;  //临时长度
    float segLen = 0;  //每段长度
    float curLen = 0;  //当前的流线长度
    float prvLen = 0;  //上一条流线长度
    float WACUM = 0;   //计算权重之和
    float texVal = 0;  //纹理的灰度值
    float smpWgt = 0;  //当前采样点的权重值
    float tacum[2];    //输入纹理对应流线上的灰度值之和
    float wacum[2];    //权重之和
    float *wgtLUT = NULL;  //权重查找表
    float len2ID = (DISCRETE_FILTER_SIZE - 1) / krnlen;  //将曲线长度映射到卷积核的某一项
    
    for (int y = 0; y < row; y++) {
        for (int x = 0; x < col; x++) {
            tacum[0] = tacum[1] = wacum[0] = wacum[1] = 0.0f;
            
            // 追踪方向，正向流线与反向流线
            for (int advDir = 0; advDir < 2; advDir++) {
                
                // 初始化追踪步数、追踪长度、种子点位置
                advcts = 0;
                curLen = 0.0f;
                clp0X = x + 0.5f;
                clp0Y = y + 0.5f;
                
                // 获取相应卷积核
                wgtLUT = (advDir == 0) ? pLUT0 : pLUT1;
                
                // 循环终止条件：流线追踪足够长或者到达涡流中心
                while (curLen < krnlen && advcts < ADVCTS) {
                    // 获得采样点的矢量数据
                    vctrX = pVectr.at<Vec2f>(clp0Y, clp0X)[0];
                    vctrY = pVectr.at<Vec2f>(clp0Y, clp0X)[1];
                    
                    // 若为关键点即一般情况下为涡流的中心时，跳出本次追踪
                    if (fabs(vctrX-0) < 0.0001 && fabs(vctrY-0) < 0.0001) {
                        tacum[advDir] = (advcts == 0) ? 0.0f : tacum[advDir];
                        wacum[advDir] = (advcts == 0) ? 1.0f : wacum[advDir];
                        break;
                    }
                    
                    // 正向追踪或反向追踪
                    vctrX = (advDir == 0) ? vctrX : -vctrX;
                    vctrY = (advDir == 0) ? vctrY : -vctrY;
                    
                    segLen = LINE_SQUARE_CLIP_MAX;
                    segLen = (vctrX < -VECTOR_COMPONENT_MIN) ?
                             (int(clp0X)-clp0X) / vctrX : segLen;
                    segLen = (vctrX > VECTOR_COMPONENT_MIN) ?
                             (int(int(clp0X) + 1.5f)-clp0X) / vctrX : segLen;
                    
                    segLen = (vctrY < -VECTOR_COMPONENT_MIN) ?
                             (((tmpLen = (int(clp0Y)-clp0Y) / vctrY) < segLen) ? tmpLen : segLen) : segLen;
                    segLen = (vctrY > VECTOR_COMPONENT_MIN) ?
                             (((tmpLen = (int(int(clp0Y)+1.5f)-clp0Y) / vctrY) < segLen) ? tmpLen : segLen) : segLen;
                    
                    prvLen = curLen;
                    curLen += segLen;
                    segLen += 0.0004f;
                    
                    // 判断长度
                    segLen = (curLen > krnlen) ? ((curLen = krnlen) - prvLen) : segLen;
                    
                    // 获取下一个追踪点位置
                    clp1X = clp0X + vctrX * segLen;
                    clp1Y = clp0Y + vctrY * segLen;
                    
                    // 计算采样点位置
                    sampX = (clp0X + clp1X) * 0.5f;
                    sampY = (clp0Y + clp1Y) * 0.5f;
                    
                    // 获取纹理采样点的灰度值
                    texVal = pNoise.at<uchar>(int(sampY), int(sampX));
                    
                    WACUM = wgtLUT[int(curLen*len2ID)];
                    smpWgt = WACUM - wacum[advDir];
                    wacum[advDir] = WACUM;
                    tacum[advDir] += texVal * smpWgt;
                    
                    advcts++;
                    clp0X = clp1X;
                    clp0Y = clp1Y;
                    
                    if (clp0X < 0.0f || clp0X >= col || clp0Y < 0.0f || clp0Y >= row) {
                        break;
                    }
                    
                } // end while
            } // end for
            
            texVal = (tacum[0] + tacum[1]) / (wacum[0] + wacum[1]);
            texVal = (texVal < 0.0f) ? 0.0f : texVal;
            texVal = (texVal > 255.0f) ? 255.0f : texVal;
            
            pImage.at<uchar>(y, x) = (uchar)texVal;
            
        }
    }
    
    return pImage;
    
}


#endif /* lic_hpp */
