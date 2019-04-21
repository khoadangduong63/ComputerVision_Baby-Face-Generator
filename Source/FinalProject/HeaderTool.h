#pragma once
#define DLIB_JPEG_SUPPORT
#include <opencv\cv.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <string>
#include <vector>

using namespace dlib;
using namespace std;
using namespace cv;
// Lấy đăc trưng của các bộ phần khuôn mặt
int getFeaturePoints(std::vector<Point2i> &PointList, string ImgName);
// Phát sinh thêm các điểm đặc trưng của khuôn mặt
void getMorePoints(std::vector<Point2i> &PointList);
// Phát sinh ảnh mang đặc trưng của khuôn mặc bố và mẹ
void morphBabyFromParents(Mat &Father, Mat &Mother, Mat &MorphImage, std::vector<Point2f> &TriangleFather, std::vector<Point2f> &TriangleMother, std::vector<Point2f> &TriangleMorph, double Alpha);
