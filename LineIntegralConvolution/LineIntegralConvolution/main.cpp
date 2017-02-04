//
//  main.cpp
//  LineIntegralConvolution
//
//  Created by xyz on 2017/2/2.
//  Copyright © 2017年 xyz. All rights reserved.
//

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include "lic.hpp"

Mat syntheszSaddle(int row, int col);

int main(int argc, const char * argv[]) {
    // insert code here...
    Lic lic;
    Mat pVect = syntheszSaddle(400, 300);
    Mat result = lic.showLIC(pVect);
    imwrite("res.jpg", result);
    return 0;
}

Mat syntheszSaddle(int row, int col) {
    Mat pVect(row, col, CV_32FC2);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            vecX = -(float)i / row + 0.5f;
            vecY = (float)j / col - 0.5f;
            pVect.at<Vec2f>(i, j)[0] = vecX;
            pVect.at<Vec2f>(i, j)[1] = vecY;
        }
    }
    return pVect;
}
