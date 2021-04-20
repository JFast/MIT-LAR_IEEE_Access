// Contributors: Adrian K. Rüppel/Hardik R. Dava/Jacob F. Fast
//
// 2018-2021
//
// Stereo_Reconstruction_and_Prediction.cpp
//
// Read raw stereo laryngoscopic single frame (or frame sequence) and corresponding camera settings and calibration parameters and perform spatial stereo reconstruction.
// If desired, compute droplet impact site prediction, based on spatial stereo reconstruction of target and provided droplet trajectory parameter file,
// visualize it in the left rectified laryngoscopic image and estimate prediction error in mm based on user-defined distance in rectified image (STANDARD METHOD)
// and/or spatial measurement in reconstructed target (NOT USED FOR ERROR QUANTIFICATION PURPOSES AS IT IS AFFECTED BY STEREO RECONSTRUCTION ERROR).

#include <QApplication>
#include <QtCore>

#include <stdio.h>
#include <string.h>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2/core/cvdef.h>
#include "opencv2/stereo.hpp"

#include<opencv2/stereo.hpp>
#include<opencv2/stereo/matching.hpp>
#include<opencv2/stereo/descriptor.hpp>
#include<opencv2/ximgproc/disparity_filter.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/core/cvdef.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/calib3d/calib3d.hpp"

#include <stdint.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <QtXml>
#include "time.h"
#include <ctime>
#include <cmath>

#define NanoSec 1000000000L;

using namespace cv;
using namespace cv::ximgproc;
using namespace cv::stereo;

// GLOBAL VARIABLES

// project path
std::string sProjectPath;

// flag for file over-writing
bool bAlreadyAsked = false;

// flag for stereo laryngoscope system
bool bRodLens = true;

// flag for BM algorithm (DO NOT CHANGE)
bool bBM = false;

// flag for binary BM (DO NOT CHANGE)
bool bBinaryBM = false;

// flag for binary SGBM (DO NOT CHANGE)
bool bBinarySGBM = true;

// flag for WLS filter (DO NOT CHANGE)
bool bWLS = true;

// flag for bilateral filter (DO NOT CHANGE)
bool bBilateralFilter = false;

// flag for impact site prediction after stereo reconstruction
bool bImpactSitePrediction = true;

// flag for single frame or frame sequence stereo reconstruction
bool bSingleFrameReconstruction = false;

// flag for impact site error estimation
bool bPredErrorEstimation = true;

// stereo reconstruction parameters
int nSADWinSize;            // must be odd
int nPreFilterCap;
int nUniquenessRatio;       // must be >0
int nSpeckleWinSize;
int nSpeckleRange;
int nDisp12MaxDiff;         // negative = disabled
int nMinDisparity;          // typically 0 for parallel stereo setup
int nNumOfDisparities;      // maximum expected disparity value (must be multiple of 16)
int nP1;
int nP2;

// WLS filter parameters
int nLambda;
float fSigmaColor;

// tolerances for impact site prediction process
double tol_pi, tol_lin, tol_para;

// size of morphological closing structuring element
int nClosingRadius;

// variables for mouse callback handling
cv::Point2i pClickedPoint = cv::Point2i(0,0);
std::vector<cv::Point2i> vClickedPointCoords;
int nClickCounter=0;

// scale factor for fiberoptical system image display
float fDisplayScale = 3.5;

// vector for images with marked impact sites
std::vector<Mat> vMarkedImages;

// vector for images with marked impact sites and manually indicated image scale
std::vector<Mat> vMarkedImagesScaleIndication;

// vector for images with marked impact sites using fiberoptical system
std::vector<Mat> vMarkedImagesResized;

// vector for images with marked impact sites and manually indicated image scale using fiberoptical system
std::vector<Mat> vMarkedImagesResizedScaleIndication;

// variables for distances between predicted and observed impact site in image in pixels
float fDistancePxLin, fDistancePxPara;

// computation time calculation
struct timespec start, stop;
double duration;

int nKey = 0;

int nOverride = -1;

// define ROIs
int nLeftPicLeftUpperCorner[2] = {0, 0};        // in Px
int nRightPicLeftUpperCorner[2] = {0, 0};       // in Px

int nImageSize[2] = {0 ,0};                     // in Px

// calibration pattern properties
float fDistanceBetweenCircles = 0.0;              // in mm
Size circlesGridSize;

// vectors for point cloud mark
std::vector<int> nRowMark, nColMark;

// FUNCTIONS

// save point cloud (with texture)
static void savePointCloud(std::string sFolderName, std::string sFileName,
                           std::vector<Mat> &v3DClouds, std::vector<int> nRowToMark,
                           std::vector<int> nColToMark, std::vector<Mat> &vImages)
{
    // path of new directory
    std::string sDirPath = sProjectPath + "/" + "OutputDataCollection" + "/" + sFolderName;

    // conversion into char
    const char* cDirPath = sDirPath.c_str();

    // try to access directory
    DIR* pDir = opendir(cDirPath);

    // if directory not found
    if (pDir == NULL)
    {
        std::cout << "No directory \"" << cDirPath << "\" found. \n"
                                                      "Folder being created... \n" << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);
    }

    for (unsigned int j = 0; j < v3DClouds.size(); j++)
    {
        // check if current point cloud contains valid data
        if(!v3DClouds[j].empty())
        {
            const float fMaxZ = 200.0; // in mm
            std::size_t pointCount = 0;

            std::vector<cv::Vec3f> points;
            std::vector<cv::Vec3b> colors;

            // read points and point texture from "v3DClouds" vector of Mats

            // for all rows of current point cloud
            for(int r = 0; r < v3DClouds[j].rows; r++)
            {
                // for all columns of current point cloud
                for(int c = 0; c < v3DClouds[j].cols; c++)
                {
                    cv::Vec3f currentPoint = v3DClouds[j].at<cv::Vec3f>(r, c);
                    cv::Vec3b currentColor = vImages[j].at<cv::Vec3b>(r, c);

                    // if absolute value of current z coordinate greater than "fMaxZ": skip point
                    if(fabs(currentPoint[2]) > fMaxZ)
                    {
                        continue;
                    }

                    // highlight desired point (see input parameters of function)
                    if (nRowToMark[j] == r && nColToMark[j] == c)
                    {
                        currentColor[0] = 255;
                        currentColor[1] = 0;
                        currentColor[2] = 0;
                    }

                    points.push_back(currentPoint);
                    colors.push_back(currentColor);
                    pointCount++;
                }
            }

            char cNumber[255];
            sprintf(cNumber, "%i", j);

            std::string sCountName = sFileName + cNumber;

            // create full path
            std::string sCompletePath = sDirPath + "/" + sCountName + ".ply";
            const char* cCompletePath = sCompletePath.c_str();

            std::ofstream file;
            file.open(cCompletePath, std::ios_base::binary);

            // write PLY file header
            file << "ply" << "\n"
                 << "format ascii 1.0" << "\n"
                 << "element vertex " << pointCount << "\n"
                 << "property float x" << "\n"
                 << "property float y" << "\n"
                 << "property float z" << "\n"
                 << "property uchar red" << "\n"
                 << "property uchar green" << "\n"
                 << "property uchar blue" << "\n"
                 << "end_header" << "\n";

            // for all points in point cloud
            for(std::size_t y = 0; y<pointCount; y++)
            {
                Vec3f point = points[y];
                Vec3b color = colors[y];

                file << point[0] << " " << point[1] << " " << point[2] << " "
                                 << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";

                //                    if(y == pointCount/2)
                //                    {
                //                        std::cout << point[0] << "  " << point[1] << "  " << point[2] << std::endl;
                //                    }
            }

            points.clear();
            colors.clear();

            file.close();
        }
    }
}

// save point cloud (without texture)
static void savePointCloud(std::string sFolderName, std::string sFileName,
                           std::vector<Mat> &p3DClouds)
{

    // path of new directory
    std::string sDirPath = sProjectPath + "/" + "OutputDataCollection" + "/" + sFolderName;

    // conversion into char
    const char* cDirPath = sDirPath.c_str();

    // try to access directory
    DIR* pDir = opendir(cDirPath);

    // if directory not found
    if (pDir == NULL)
    {
        std::cout << "No directory \"" << cDirPath << "\" found. \n"
                                                      "Folder being created... \n" << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);
    }

    for (unsigned int j = 0; j < p3DClouds.size(); j++)
    {

        // check if current point cloud contains valid data
        if(!p3DClouds[j].empty())
        {

            // read point coordinates from "p3DClouds" vector of Mats
            const float fMaxZ = 200.0; // in mm
            std::size_t pointCount = 0;

            std::vector<cv::Vec3d> points;

            // for all rows of current point cloud
            for(int y = 0; y < p3DClouds[j].rows; y++)
            {
                // for all columns of current point cloud
                for(int x = 0; x < p3DClouds[j].cols; x++)
                {
                    cv::Vec3d currentPoint = p3DClouds[j].at<cv::Vec3d>(y, x);

                    // if absolute value of current z coordinate greater than "fMaxZ": skip point
                    if(fabs(currentPoint[2] > fMaxZ))
                    {
                        continue;
                    }

                    points.push_back(currentPoint);
                    pointCount++;
                }
            }

            char cNumber[255];
            sprintf(cNumber, "%i", j);

            std::string sCountName = sFileName + cNumber;

            // create full path
            std::string sCompletePath = sDirPath + "/" + sCountName + ".ply";
            const char* cCompletePath = sCompletePath.c_str();

            std::ofstream file;
            file.open(cCompletePath, std::ios_base::binary);

            // write ply header
            file << "ply" << "\n"
                 << "format ascii 1.0" << "\n"
                 << "element vertex " << pointCount << "\n"
                 << "property float x" << "\n"
                 << "property float y" << "\n"
                 << "property float z" << "\n"
                 << "property uchar red" << "\n"
                 << "property uchar green" << "\n"
                 << "property uchar blue" << "\n"
                 << "end_header" << "\n";

            for(std::size_t y = 0; y < pointCount; y++)
            {
                Vec3d point = points[y];

                file << point[0] << " " << point[1] << " " << point[2] << " "
                                 << 0 << " " << 0 << " " << 0 << "\n";
            }

            points.clear();

            file.close();
        }

    }
}

// read all images in directory
bool loadImages(std::string sFolderName, std::vector<std::string> &vNames, std::vector<Mat> &vLoadedImages, bool bGray)
{
    // directory path
    std::string sDirPath = sProjectPath + "/" + sFolderName;
    const char* cDirPath = sDirPath.c_str();

    // loading error variable
    bool bError = false;

    // open directory
    DIR* pDir = opendir(cDirPath);

    std::string sFullPath;

    // get files in directory
    struct dirent* pFilesInDir;

    // current image
    cv::Mat currentImage;

    // if directory not found
    if (pDir == NULL)
    {
        std::cout << "Folder \"" << cDirPath << "\" not found. \n"
                     "Folder being created. \n"
                     "Please add images to directory \"" << cDirPath << "\" "
                     "and restart program. \n" "Hit key to end program execution." << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);

        bError = true;

        // wait for user input
        cv::waitKey(0);

        return bError;
    }

    // read images from directory
    while((pFilesInDir = readdir(pDir)) != NULL)
    {
        // exclude directories "." and ".."
        if (!(strcmp (pFilesInDir->d_name, ".")) || !(strcmp(pFilesInDir->d_name, "..")))
        {
            continue;
        }

        // full path
        sFullPath = sDirPath + "/" + std::string(pFilesInDir->d_name);

        // read current image
        if (bGray)
        {
            currentImage = cv::imread(sFullPath, IMREAD_GRAYSCALE);
        }
        else
        {
            currentImage = cv::imread(sFullPath, IMREAD_COLOR);
        }

        // skip file, if no image file
        if(currentImage.empty())
        {
            continue;
        }

        // save current image
        vLoadedImages.push_back(currentImage);

        // save current file name
        vNames.push_back(pFilesInDir->d_name);
    }

    closedir(pDir);
    currentImage.release();
    sDirPath.clear();
    sFullPath.clear();

    // cDirPath.clear();

    return bError;
}

// retrieve left and right stereoscopic sub-image from sensor of high-speed camera
void splitImages(Mat &pImage, Mat &pLeftImage, Mat &pRightImage)
{
    // define ROIs
    Rect leftRectangle = Rect(nLeftPicLeftUpperCorner[0], nLeftPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);
    Rect rightRectangle = Rect(nRightPicLeftUpperCorner[0], nRightPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);

    // extract left image
    pLeftImage = pImage(leftRectangle);

    // extract right image
    pRightImage = pImage(rightRectangle);
}

// save/show images
void saveOrShowImage(std::vector<Mat> &vImages,std::string sFolderName, std::string sFileName, bool bShow)
{
    // folder path
    std::string sDirPath = sProjectPath + "/" + "OutputDataCollection" + "/" + sFolderName;

    // conversion in char
    const char* cDirPath = sDirPath.c_str();

    // open folder
    DIR* pDir = opendir(cDirPath);

    // create folder if not already existing
    if (pDir == NULL)
    {
        std::cout << "No folder \"" << cDirPath << "\" found. \n"
                     "Folder is being created. \n" << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);

        nOverride = 1;
    }
    else
    {
        if (!bAlreadyAsked)
        {
            while(true)
            {
                std::cout << "One or more folders for image storage already exist.\n"
                        "Override folder contents, if applicable? (1 = Yes, 0 = No)" << std::endl;

                std::cin >> nOverride;

                if (nOverride == 0 || nOverride == 1)
                {
                    break;
                }
            }

            bAlreadyAsked = true;
        }
    }

    for (unsigned int i = 0; i < vImages.size(); i++)
    {
        if (bShow == true)
        {

            // skip file, if no image could be read
            if((vImages[i].empty()))
            {
                continue;
            }

            cv::namedWindow("Display", CV_WINDOW_AUTOSIZE);

            cv::imshow("Display", vImages[i]);

            // wait for user input
            cv::waitKey(0);

            cv::destroyWindow("Display");
        }

        if (bShow == false && nOverride == 1)
        {
            char cNumber[255];
            sprintf(cNumber, "%i", i);

            std::string sCountName = sFileName + cNumber;

            // embed path to image
            std::string sCompletePath = sDirPath + "/" + sCountName + ".png";
            const char* cCompletePath = sCompletePath.c_str();

            // skip file, if no image could be read
            if((vImages[i].empty()))
            {
                continue;
            }

            cv::imwrite(cCompletePath, vImages[i]);
        }
    }
}

// mouse callback handler
void mouse_callback(int event, int x, int y, int flags, void* param)
{
    cv::Mat& image = *(cv::Mat*) param;

    switch(event)
    {
    case cv::EVENT_LBUTTONUP: {
        pClickedPoint.x = x;
        pClickedPoint.y = y;
        vClickedPointCoords.push_back(pClickedPoint);

        if(nClickCounter<2)
        {
            // draw crosshairs to indicate chosen position
            cv::line(image, Point(pClickedPoint.x-5, pClickedPoint.y), Point(pClickedPoint.x+5, pClickedPoint.y), Scalar(255,255,0),0.6,8);
            cv::line(image, Point(pClickedPoint.x, pClickedPoint.y-5), Point(pClickedPoint.x, pClickedPoint.y+5), Scalar(255,255,0),0.6,8);
        }

        if(nClickCounter==1)
        {
            cv::line(image, Point(vClickedPointCoords[0].x, vClickedPointCoords[0].y), Point(vClickedPointCoords[1].x, vClickedPointCoords[1].y), Scalar(150,255,0),0.5,8);
        }

        nClickCounter++;
    }
        break;
    }
}

// calculate average geometric distance between sampling points and parabolic fit function (STANDARD MODEL)
double calculateAverageGeometricDistanceParaSinglePoint(cv::Mat &vecF, cv::Point3f pTriangulatedPoint)
{
    // geometric distance between sampling point and parabolic fit function
    double dist_para = 0.0;

    // variable for squared distance between sampling point and parabolic fit function
    double dist_para_squared = 0.0;

    // vectors for coefficients and roots of cubic polynomial for distance calculation
    std::vector<double> coeffs(4,0.0);
    std::vector<double> roots(3,0.0);

    // variable for t_starred (in ms)
    double t_starred = 0.0;

    // coefficient of t³ (constant)
    coeffs[0] = (double)(4.0*(std::pow(vecF.at<float>(0,0),2)+std::pow(vecF.at<float>(3,0),2)+std::pow(vecF.at<float>(6,0),2)));

    // coefficient of t² (constant)
    coeffs[1] = (double)(6.0*(vecF.at<float>(1,0)*vecF.at<float>(0,0)+vecF.at<float>(4,0)*vecF.at<float>(3,0)+vecF.at<float>(7,0)*vecF.at<float>(6,0)));

    // coefficient of t (depends of current sampling point)
    coeffs[2] = (double)(2.0*(std::pow(vecF.at<float>(1,0),2)+2.0*vecF.at<float>(2,0)*vecF.at<float>(0,0)-2.0*vecF.at<float>(0,0)*pTriangulatedPoint.x+std::pow(vecF.at<float>(4,0),2)+2.0*vecF.at<float>(5,0)*vecF.at<float>(3,0)-2.0*vecF.at<float>(3,0)*pTriangulatedPoint.y+std::pow(vecF.at<float>(7,0),2)+2.0*vecF.at<float>(8,0)*vecF.at<float>(6,0)-2.0*vecF.at<float>(6,0)*pTriangulatedPoint.z));

    // coefficient of 1 (depends of current sampling point)
    coeffs[3] = (double)(2.0*(vecF.at<float>(2,0)*vecF.at<float>(1,0)-vecF.at<float>(1,0)*pTriangulatedPoint.x+vecF.at<float>(5,0)*vecF.at<float>(4,0)-vecF.at<float>(4,0)*pTriangulatedPoint.y+vecF.at<float>(8,0)*vecF.at<float>(7,0)-vecF.at<float>(7,0)*pTriangulatedPoint.z));

    // calculate value of parameter t_starred (in ms) of point on parabolic function with lowest distance to current sampling point
    // highest-order coefficients come first
    cv::solveCubic(coeffs, roots);

    // root must be positive (first sampling point recorded at time stamp t_1 >= 0 ms)
    // only one positive root expected in vector roots[]

    if(roots[0] > 0)
    {
        t_starred = roots[0];
    }
    else if(roots[1] > 0)
    {
        t_starred = roots[1];
    }
    else if(roots[2] > 0)
    {
        t_starred = roots[2];
    }
    else    // no physically plausible root found
    {
        t_starred = 0.0;    // value will yield large distance to target point cloud (approx. corresponds to position of first observation of droplet along its trajectory towards the target)
    }

    // calculate shortest distance between current point and parabolic trajectory model
    dist_para_squared = (double)(std::pow(vecF.at<float>(2,0) + vecF.at<float>(1,0)*t_starred+vecF.at<float>(0,0)*std::pow(t_starred,2) - pTriangulatedPoint.x, 2) + std::pow(vecF.at<float>(5,0) + vecF.at<float>(4,0)*t_starred+vecF.at<float>(3,0)*std::pow(t_starred,2) - pTriangulatedPoint.y,2) + std::pow(vecF.at<float>(8,0) + vecF.at<float>(7,0)*t_starred + vecF.at<float>(6,0)*std::pow(t_starred,2) - pTriangulatedPoint.z,2));

    // calculate  geometric distance
    dist_para = std::sqrt(dist_para_squared);

    // std::cout << "Current geometric distance from sampling point to parabolic fit in mm: " << dist_para << std::endl;

    // output average geometric distance between sampling points and parabolic fit function
    // std::cout << "Distance from sampling point to parabolic fit in mm: " << dist_para << std::endl;

    // return average geometric distance between sampling points and parabolic fit function
    return dist_para;
}

// calculate reduced set of candidate points based on fit plane PI
std::vector<cv::Point3f> ReduceCandidateSetPI(std::vector<cv::Point3f> &vFullSetOfCandidates, cv::Vec3f &vNormalVectorPI, double dTolPI, double dD)
{
    // declare output vector
    std::vector<cv::Point3f> vImpactSiteCandidates;
    vImpactSiteCandidates.resize(0);

    // go through spatially reconstructed target points
    for (unsigned int i=0; i<vFullSetOfCandidates.size(); ++i)
    {
        // only maintain points which are closer than dTolPI to plane PI
        if ((double)std::abs(vFullSetOfCandidates[i].dot(vNormalVectorPI)-dD) < dTolPI)
        {
            vImpactSiteCandidates.push_back(vFullSetOfCandidates[i]);
        }
    }

    return vImpactSiteCandidates;
}

// MAIN PROGRAM

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    cv::ocl::setUseOpenCL(true);

    // std::cout << cv::checkHardwareSupport(CV_CPU_SSE2) << std::endl;

    // image boundary handling
    const Scalar borderValueRemap = (0);

    // configure stereo matching parameters

    int nTextureThreshold = 10;

    // shared parameters for both stereo laryngoscopes
    // (heuristically optimized by Mr Dava in 2020)

    nUniquenessRatio = 5;   // must be >0
    nDisp12MaxDiff = 2;     // negative = disabled
    nSADWinSize = 5;        // must be odd
    nMinDisparity = 0;

    // pre-filter cap value
    nPreFilterCap = 20;

    // adapted matching parameters for each laryngoscope system

    if(bRodLens)
    {
        nSpeckleWinSize = 150;
        nSpeckleRange = 2;
        nNumOfDisparities = 112; // must be >0 and divisible by 16 for BinarySGBM
        nP1 = 200;
        nP2 = 1000;
    }
    else
    {
        nSpeckleWinSize = 40;
        nSpeckleRange = 1;
        nNumOfDisparities = 48;  // must be >0 and divisible by 16 for BinarySGBM
        nP1 = 90;
        nP2 = 500;
    }

    // parameters of OPTIONAL bilateral filter
    int nBilateralDiameter = 10;
    int nSigmaColour = 100;
    int nSigmaSpace = 100;

    // kernel size of optional median and Gaussian filter
    // int nKernelSize = 9;

    // parameters of optional WLS filter

    if(bRodLens)
    {
        nLambda = 90;
        fSigmaColor = 1.4;
    }
    else
    {
        nLambda = 30;
        fSigmaColor = 1.2;
    }

    // configure morphological closing operation

    if(bRodLens)
    {
        nClosingRadius = 20;
    }
    else
    {
        nClosingRadius = 6;
    }

    // read current project path from argv
    std::string pTempPath[256] = {argv[0]};
    std::string sExecPath;
    sExecPath = *pTempPath;

    int iPosOfChar = sExecPath.find_last_of("/");

    // save project path
    sProjectPath = sExecPath.substr(0, iPosOfChar);

    std::cout  <<  sProjectPath  <<  std::endl;

    // declare calibration data matrices
    cv::Mat pCameraMatrixLeft, pCameraMatrixRight, pDistortionCoeffsLeft, pDistortionCoeffsRight;

    // initialize camera matrices
    pCameraMatrixLeft = cv::Mat::eye(3,3, CV_64F);
    pCameraMatrixRight = cv::Mat::eye(3,3, CV_64F);

    // initialize stereo reprojection error in px
    double dRepErrorComplete = 9999.0;

    // declare structure for ROI size
    cv::Size pImageSize;

    // declare Mat structure for rotation between virtual cameras
    cv::Mat pLeftToRightRotationMatrix;

    // declare Mat structure for translation between virtual cameras
    cv::Mat pLeftToRightTranslationVector;

    // read file "settings.xml"
    std::cout << "Loading file \"settings.xml\"..." << std::endl;

    cv::FileStorage set;

    // check if file "settings.xml" with ROI information exists

    // if file not detected: stop program execution
    if(!set.open(sProjectPath + "/" +"settings.xml", cv::FileStorage::READ))
    {
        std::cout << "No file \"settings.xml\" available.\n"
                     "File is being created.\n"
                     "Please fill in required parameters (coordinates in pixels) and re-run program.\n"
                     "Hit key to end program execution..."<< std::endl;

        // create "settings.xml" file
        cv::FileStorage set ("settings.xml", FileStorage::WRITE);

        time_t rawtime;
        time(&rawtime);

        set << "Date" << asctime(localtime(&rawtime));
        set << "picture_sections" << "{";
        set << "upper_left_corner_of_left_picture_section" << "{";
        set << "x_coordinate" << 0;
        set << "y_coordinate" << 0;
        set << "}";
        set << "upper_left_corner_of_right_picture_section" << "{";
        set << "x_coordinate" << 0;
        set << "y_coordinate" << 0;
        set << "}";
        set << "size_each_image_section" << "{";
        set << "x_direction" << 0;
        set << "y_direction" << 0;
        set << "}";
        set << "}";
        set << "circles_grid_settings" << "{";
        set << "distance_between_circles_in_mm" << 0;
        set << "size_circles_grid" << "{";
        set << "rows" << 0;
        set << "columns" << 0;
        set << "}";

        set.release();

        // wait for user input
        cv::waitKey(0);

        // stop program execution
        return 0;
    }

    // if file "settings.xml" available: read stored parameter values
    else
    {
        // read ROI settings from file
        FileNode pictureSections = set["picture_sections"];
        FileNodeIterator sectionIt = pictureSections.begin(), sectionItEnd = pictureSections.end();

        std::vector <int> vSectionInfo;
        std::vector <float> vGridInfo;

        // ROIs
        for ( ; (sectionIt != sectionItEnd); sectionIt++)
        {
            FileNode childNode = *sectionIt;
            FileNodeIterator childIt = childNode.begin(), childItEnd = childNode.end();

            for (; (childIt != childItEnd); childIt++)
            {
                int nTemp;
                *childIt >> nTemp;
                vSectionInfo.push_back(nTemp);
            }

        }

        // read calibration pattern information
        FileNode gridSettings = set ["circles_grid_settings"];
        FileNodeIterator gridIt = gridSettings.begin(), gridItEnd = gridSettings.end();

        for (; (gridIt != gridItEnd); gridIt++)
        {
            FileNode childNode = *gridIt;
            FileNodeIterator childIt = childNode.begin(), childItEnd = childNode.end();

            for (; (childIt != childItEnd); childIt++)
            {
                float nTemp;
                *childIt >> nTemp;
                vGridInfo.push_back(nTemp);
            }
        }

        // save parameter values
        nLeftPicLeftUpperCorner[0] = vSectionInfo[0];
        nLeftPicLeftUpperCorner[1] = vSectionInfo[1];

        nRightPicLeftUpperCorner[0] = vSectionInfo[2];
        nRightPicLeftUpperCorner[1] = vSectionInfo[3];

        nImageSize[0] = vSectionInfo[4];
        nImageSize[1] = vSectionInfo[5];

        fDistanceBetweenCircles = vGridInfo[0];
        circlesGridSize = Size(((int)vGridInfo[1]), ((int) vGridInfo[2]));
    }

    set.release();

    // read calibration data from file "calibration.xml"
    cv::FileStorage fs;

    // stop program execution if file "calibration.xml" not available
    if(!fs.open(sProjectPath + "/" +"calibration.xml", cv::FileStorage::READ))
    {
        std::cout << "No file \"calibration.xml\" with calibration parameters found.\n"
                     "Hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        // stop program execution
        return 0;
    }

    // read calibration parameter values from file
    fs["imageSize"] >> pImageSize;
    fs["CameraMatrixLeft"] >> pCameraMatrixLeft;
    fs["DistortionCoefficientsLeft"] >> pDistortionCoeffsLeft;
    fs["CameraMatrixRight"] >> pCameraMatrixRight;
    fs["DistortionCoefficientsRight"] >> pDistortionCoeffsRight;
    fs["RotationMatrix"] >> pLeftToRightRotationMatrix;
    fs["TranslationVector"] >> pLeftToRightTranslationVector;
    fs["ReprojectionError"] >> dRepErrorComplete;

    fs.release();

    std::cout << "Calibration parameters successfully read. \n"
                 "Hit key to start rectification parameter calculation..." << std::endl;

    // wait for user input
    cv::waitKey(0);

    // declare structures for rectification parameters and look-up-maps
    cv::Mat pR1Left, pR2Right, pP1Left, pP2Right, pQ, pMapLeft1, pMapLeft2, pMapRight1, pMapRight2;

    // ROI declarations
    cv::Rect validROIL, validROIR;

    // calculate stereo image rectification parameters
    cv::stereoRectify(pCameraMatrixLeft,
                  pDistortionCoeffsLeft,
                  pCameraMatrixRight,
                  pDistortionCoeffsRight,
                  pImageSize,
                  pLeftToRightRotationMatrix,
                  pLeftToRightTranslationVector,
                  pR1Left,
                  pR2Right,
                  pP1Left,
                  pP2Right,
                  pQ,
                  CALIB_ZERO_DISPARITY,             // flag for disparity at infinity (0 for parallel setup)
                  -1,                               // flag "alpha" for pixel range to be considered
                  pImageSize,                       // image size after rectification
                  &validROIL,                       // valid ROI left
                  &validROIR);                      // valid ROI right

    std::cout << "Q matrix: " << pQ << std::endl;

    // pre-calculation of look-up maps for fast undistortion and rectification

    // calculate look-up map for left virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixLeft, pDistortionCoeffsLeft, pR1Left,
                            pP1Left, pImageSize, CV_16SC2, pMapLeft1, pMapLeft2);

    // calculate look-up map for right virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixRight, pDistortionCoeffsRight, pR2Right,
                            pP2Right, pImageSize, CV_16SC2, pMapRight1, pMapRight2);


    // STEREO RECONSTRUCTION PROCEDURE

    // declarations (may contain unused vectors)
    std::vector<cv::Mat> vRawSingleFrames, vRawSingleFramesGray, vLeftImages, vLeftImagesGray, vRightImages, vRightImagesGray, vLeftImagesRect, vLeftImagesRectGray,
            vRightImagesRect, vRightImagesRectGray, vCompleteImagesRectGray, vCompleteImagesRectLines,
            vLeftDispMaps, vRightDispMaps, vWLSDispMaps, vDepthMaps, vColorCodedDepthImages, vNormalizedDispImages;

    // if single frame reconstruction desired
    if(bSingleFrameReconstruction)
    {
        std::vector<std::string> vFrameNames;

        // read image files from folder "single_frames" into vector "vRawSingleFrames"
        loadImages("single_frames", vFrameNames, vRawSingleFrames, false);

        cv::namedWindow("First Input Image", WINDOW_AUTOSIZE);
        cv::imshow("First Input Image", vRawSingleFrames[0]);

        cv::waitKey(0);

        cv::destroyWindow("First Input Image");
    }
    // if frame sequence assessment desired (for quantitative comparison of predicted and observed droplet impact site)
    else
    {
        // read frame sequence from folder and store in array
        cv::VideoCapture trajVidObj(sProjectPath + "/" + "Os7-S1 Camera.mp4");

        // end program if no frame sequence found
        if(!trajVidObj.isOpened())
            {
                std::cout << "No frame sequence could be found! \n"
                         "Please add frame sequence to project directory and re-run program. \n"
                         "Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                // stop program execution
                return 0;
            }

        // declaration for current frame
        cv::Mat pCurrentFrame;

        // store single frames from frame sequence in vector
        while(trajVidObj.read(pCurrentFrame))
            {
                vRawSingleFrames.push_back(pCurrentFrame.clone());
            }

        // declaration for first and last frame of frame sequence
        std::vector<Mat> vFirstFrame, vLastFrame;

        // store first and last frame of sequence
        vFirstFrame.push_back(vRawSingleFrames[0]);
        vLastFrame.push_back(vRawSingleFrames[vRawSingleFrames.size()-1]);

        // show last frame if desired
        // saveOrShowImage(vLastFrame, "InputFrames", "LastFrame", false);

        // show first frame of sequence
        cv::namedWindow("First Frame", WINDOW_AUTOSIZE);
        cv::imshow("First Frame", vRawSingleFrames[0]);

        std::cout << "Press key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("First Frame");

        trajVidObj.release();
    }

    // resize vectors
    vRawSingleFramesGray.resize(vRawSingleFrames.size());

    vLeftImages.resize(vRawSingleFrames.size());
    vRightImages.resize(vRawSingleFrames.size());

    vLeftImagesGray.resize(vRawSingleFrames.size());
    vRightImagesGray.resize(vRawSingleFrames.size());

    vLeftImagesRect.resize(vRawSingleFrames.size());
    vRightImagesRect.resize(vRawSingleFrames.size());

    vLeftImagesRectGray.resize(vRawSingleFrames.size());
    vRightImagesRectGray.resize(vRawSingleFrames.size());

    vCompleteImagesRectGray.resize(vRawSingleFrames.size());

    vLeftDispMaps.resize(vRawSingleFrames.size());
    vRightDispMaps.resize(vRawSingleFrames.size());
    vWLSDispMaps.resize(vRawSingleFrames.size());
    vDepthMaps.resize(vRawSingleFrames.size());

    // perform stereo reconstruction with all frames

    // for (int i = 0; i < vRawSingleFrames.size(); i++){}

    // perform stereo reconstruction with first frame only
    for (int i = 0; i < 1; i++)
    {
        // extract ROIs from frame (COLOR IMAGES FOR STEREO RECONSTRUCTION TEXTURE INFORMATION)
        splitImages(vRawSingleFrames[i], vLeftImages[i], vRightImages[i]);

        // rectify left image (COLOR IMAGES FOR STEREO RECONSTRUCTION TEXTURE INFORMATION)
        cv::remap(vLeftImages[i], vLeftImagesRect[i], pMapLeft1, pMapLeft2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);
        // rectify right image (COLOR IMAGES FOR STEREO RECONSTRUCTION TEXTURE INFORMATION)
        cv::remap(vRightImages[i], vRightImagesRect[i], pMapRight1, pMapRight2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

        // convert frame to grayscale
        cv::cvtColor(vRawSingleFrames[i], vRawSingleFramesGray[i], CV_BGR2GRAY);

        // extract ROIs from frame (GRAYSCALE IMAGES FOR BINARY SGBM)
        splitImages(vRawSingleFramesGray[i], vLeftImagesGray[i], vRightImagesGray[i]);

        // rectify left image (GRAYSCALE IMAGES FOR BINARY SGBM)
        cv::remap(vLeftImagesGray[i], vLeftImagesRectGray[i], pMapLeft1, pMapLeft2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);
        // rectify right image (GRAYSCALE IMAGES FOR BINARY SGBM)
        cv::remap(vRightImagesGray[i], vRightImagesRectGray[i], pMapRight1, pMapRight2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

        // concatenate rectified images (GRAYSCALE IMAGES FOR BINARY SGBM)
        cv::hconcat(vLeftImagesRectGray[i], vRightImagesRectGray[i], vCompleteImagesRectGray[i]);

        cv::namedWindow("Rectified Input Image", WINDOW_AUTOSIZE);
        cv::imshow("Rectified Input Image", vCompleteImagesRectGray[0]);

        cv::waitKey(0);

        cv::destroyWindow("Rectified Input Image");

        // SIMPLE BM (NOT USED)
        if(bBM)
        {
            Ptr<cv::StereoBM> pLeftMatcherBM = cv::StereoBM::create(nNumOfDisparities,nSADWinSize);

            pLeftMatcherBM->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
            pLeftMatcherBM->setPreFilterCap(nPreFilterCap);
            pLeftMatcherBM->setUniquenessRatio(nUniquenessRatio);
            pLeftMatcherBM->setSpeckleWindowSize(nSpeckleWinSize);
            pLeftMatcherBM->setSpeckleRange(nSpeckleRange);
            pLeftMatcherBM->setMinDisparity(0);

            pLeftMatcherBM->compute(vLeftImagesRectGray[i], vRightImagesRectGray[i], vLeftDispMaps[i]);

            // WLS FILTERING
            if(bWLS)
            {
                Ptr<cv::ximgproc::DisparityWLSFilter> pWlsFilter;
                Ptr<cv::StereoMatcher> pRightMatcher;

                pRightMatcher = cv::ximgproc::createRightMatcher(pLeftMatcherBM);
                pWlsFilter = cv::ximgproc::createDisparityWLSFilter(pLeftMatcherBM);

                // disparity left to right
                pLeftMatcherBM->compute(vLeftImagesRectGray[i], vRightImagesRectGray[i], vLeftDispMaps[i]);
                // disparity right to left
                pRightMatcher->compute(vRightImagesRectGray[i], vLeftImagesRectGray[i], vRightDispMaps[i]);

                pWlsFilter->setLambda(nLambda);
                pWlsFilter->setSigmaColor(fSigmaColor);

                pWlsFilter->filter(vLeftDispMaps[i], vLeftImagesRectGray[i], vWLSDispMaps[i], vRightDispMaps[i]);

                std::cout << "WLS filtering successful!" << std::endl;

                pWlsFilter->clear();
            }
            else
            {
                vWLSDispMaps[i] = vLeftDispMaps[i].clone();
            }
        }
        // BINARY BM (WITH CENSUS TRANSFORMATION OF INPUT IMAGES) (NOT USED)
        else if(bBinaryBM)
        {
            cv::Mat pBinaryBMTempDispMap = cv::Mat(vLeftImagesRectGray[i].rows, vLeftImagesRectGray[i].cols, CV_16S);

            Ptr<cv::stereo::StereoBinaryBM> pLeftmatcherBMBinary = cv::stereo::StereoBinaryBM::create(nNumOfDisparities, nSADWinSize);

            pLeftmatcherBMBinary->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_ALGORITHM); // speckle removal technique
            pLeftmatcherBMBinary->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
            pLeftmatcherBMBinary->setPreFilterCap(nPreFilterCap);
            pLeftmatcherBMBinary->setUniquenessRatio(nUniquenessRatio);
            // pLeftmatcherBMBinary->setSpeckleWindowSize(nSpeckleWinSize);
            // pLeftmatcherBMBinary->setSpeckleRange(nSpeckleRange);
            // pLeftmatcherBMBinary->setScalleFactor(16);
            pLeftmatcherBMBinary->setDisp12MaxDiff(nDisp12MaxDiff);
            pLeftmatcherBMBinary->setTextureThreshold(nTextureThreshold);
            pLeftmatcherBMBinary->setMinDisparity(0);
            pLeftmatcherBMBinary->setAgregationWindowSize(11);
            pLeftmatcherBMBinary->setUsePrefilter(false);

            // binary descriptor type
            // ALL KERNEL TYPES DO NOT YIELD 100 % REPRODUCIBLE STEREO RECONSTRUCTION RESULTS (POINT CLOUDS)
            // KERNEL IDs: 0 CV_DENSE_CENSUS, 1 CV_SPARSE_CENSUS, 2 CV_CS_CENSUS, 3 CV_MODIFIED_CS_CENSUS, 4 CV_MODIFIED_CENSUS_TRANSFORM, 5 CV_MEAN_VARIATION, 6 CV_STAR_KERNEL
            // source code for calculation of descriptors available here: https://github.com/opencv/opencv_contrib/blob/master/modules/stereo/src/descriptor.cpp
            pLeftmatcherBMBinary->setBinaryKernelType(2);

            pLeftmatcherBMBinary->compute(vLeftImagesRectGray[i], vRightImagesRectGray[i], pBinaryBMTempDispMap);

            vLeftDispMaps[i] = pBinaryBMTempDispMap.clone();

            // WLS FILTERING of raw disparity map (Fast Global Smoother)
            if(bWLS)
            {
                // normal WLS filter method does not work for CENSUS method. Set every parameter manually
                // see https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
                // (calculate right disparity map to enable LR check)

                Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

                // create WLS filter instance without left-right consistency check (thus flag false)
                wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);

                wls_filter->setDepthDiscontinuityRadius((int)std::ceil(0.5*nSADWinSize));

                wls_filter->setLambda(nLambda);
                wls_filter->setSigmaColor(fSigmaColor);

                // perform filtering
                wls_filter->filter(vLeftDispMaps[i],vLeftImagesRectGray[i],vWLSDispMaps[i],cv::Mat());
            }
            else
            {
                vWLSDispMaps[i] = vLeftDispMaps[i].clone();
            }
        }
        // BINARY SGBM (WITH CENSUS TRANSFORMATION OF INPUT IMAGES)
        else if(bBinarySGBM)
        {
            // negative nDisp12MaxDiff: difference check disabled
            // uniqueness ratio must be non-negative
            // num of disparities must be > 0 and divisible by 16
            // full 8-direction algorithm used here (slower, but more accurate)
            // alternatively use cv::stereo::StereoBinarySGBM::MODE_SGBM (5-direction mode)
            Ptr<cv::stereo::StereoBinarySGBM> pLeftMatcherSGBMBinary = cv::stereo::StereoBinarySGBM::create(nMinDisparity,nNumOfDisparities,nSADWinSize, nP1, nP2, nDisp12MaxDiff, nPreFilterCap, nUniquenessRatio, nSpeckleWinSize, nSpeckleRange, cv::stereo::StereoBinarySGBM::MODE_HH);

            pLeftMatcherSGBMBinary->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_ALGORITHM); // speckle removal technique

            pLeftMatcherSGBMBinary->setSubPixelInterpolationMethod(cv::stereo::CV_QUADRATIC_INTERPOLATION); // interpolation technique

            // binary descriptor type
            // MAY NOT YIELD 100 % REPRODUCIBLE STEREO RECONSTRUCTION RESULTS (POINT CLOUDS)
            // KERNEL IDs: 0 CV_DENSE_CENSUS, 1 CV_SPARSE_CENSUS, 2 CV_CS_CENSUS, 3 CV_MODIFIED_CS_CENSUS, 4 CV_MODIFIED_CENSUS_TRANSFORM, 5 CV_MEAN_VARIATION, 6 CV_STAR_KERNEL
            // source code for calculation of descriptors available here: https://github.com/opencv/opencv_contrib/blob/master/modules/stereo/src/descriptor.cpp
            // (test with 3D phantom) 0: not reproducible, 1: not reproducible, 2: reproducible, 3: not reproducible, 4: reproducible, 5: not reproducible, 6: not reproducible
            // (test with 2D target): 0: not reproducible, 1: not reproducible, 2: reproducible, 3: not reproducible, 4: not reproducible, 5: not reproducible, 6: not reproducible
            pLeftMatcherSGBMBinary->setBinaryKernelType(2);

            // measure SGBM computation time in milliseconds
            timespec timeStartSGBM, timeEndSGBM;

            int nCompTimeSGBM = 0;

            clock_gettime(CLOCK_MONOTONIC, &timeStartSGBM);

            pLeftMatcherSGBMBinary->compute(vLeftImagesRectGray[i], vRightImagesRectGray[i], vLeftDispMaps[i]);

            clock_gettime(CLOCK_MONOTONIC, &timeEndSGBM);

            nCompTimeSGBM = ((uint64_t)timeEndSGBM.tv_sec * 1000LL + (uint64_t)timeEndSGBM.tv_nsec / 1000000LL) -
                    ((uint64_t)timeStartSGBM.tv_sec * 1000LL + (uint64_t)timeStartSGBM.tv_nsec / 1000000LL);

            std::cout << "Binary SGBM computation time in ms: " << nCompTimeSGBM << std::endl;

            // WLS FILTERING of raw disparity map (using Fast Global Smoother method, see OpenCV documentation)
            if(bWLS)
            {
                // normal WLS filter method does not work for CENSUS method. Set parameters manually.
                // see: https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html (accessed on 03/02/2021)
                // (calculate right disparity map to enable LR check)

                Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

                // create WLS filter instance without left-right consistency check (thus flag false)
                wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);

                wls_filter->setDepthDiscontinuityRadius((int)std::ceil(0.5*nSADWinSize));

                wls_filter->setLambda(nLambda);
                wls_filter->setSigmaColor(fSigmaColor);

                // measure WLS filtering computation time in milliseconds
                timespec timeStartWLS, timeEndWLS;

                int nCompTimeWLS = 0;

                clock_gettime(CLOCK_MONOTONIC, &timeStartWLS);

                // perform filtering
                wls_filter->filter(vLeftDispMaps[i],vLeftImagesRectGray[i],vWLSDispMaps[i],cv::Mat());

                clock_gettime(CLOCK_MONOTONIC, &timeEndWLS);

                nCompTimeWLS = ((uint64_t)timeEndWLS.tv_sec * 1000LL + (uint64_t)timeEndWLS.tv_nsec / 1000000LL) -
                        ((uint64_t)timeStartWLS.tv_sec * 1000LL + (uint64_t)timeStartWLS.tv_nsec / 1000000LL);

                std::cout << "WLS filtering computation time in ms: " << nCompTimeWLS << std::endl;
            }
            else
            {
                vWLSDispMaps[i] = vLeftDispMaps[i].clone();
            }

            // output vWLSDispMaps[0]
            // std::cout << "vWLSDispMaps[0]: " << vWLSDispMaps[0] << std::endl;
        }
        // STANDARD SGBM (NOT USED)
        else
        {
            // create SGBM object (last flag: 0 = MODE_SGBM (5 directions), 1 = MODE_HH (8 directions, slow), 2 = MODE_SGBM_3WAY (3 directions, fast))
            // MODE_HH is not parallelized and hence, very slow

            Ptr<cv::StereoSGBM> pLeftMatcherSGBM = cv::StereoSGBM::create(nMinDisparity,
                                                                          nNumOfDisparities,
                                                                          nSADWinSize,
                                                                          nP1,
                                                                          nP2,
                                                                          nDisp12MaxDiff,
                                                                          nPreFilterCap,
                                                                          nUniquenessRatio,
                                                                          nSpeckleWinSize,
                                                                          nSpeckleRange,cv::StereoSGBM::MODE_HH);

            // STANDARD SGBM (USING COLOR IMAGES)

            // measure standard SGBM computation time in milliseconds
            timespec timeStartSGBMNonBinary, timeEndSGBMNonBinary;

            int nCompTimeSGBMNonBinary = 0;

            clock_gettime(CLOCK_MONOTONIC, &timeStartSGBMNonBinary);

            pLeftMatcherSGBM->compute(vLeftImagesRect[i], vRightImagesRect[i], vLeftDispMaps[i]);

            clock_gettime(CLOCK_MONOTONIC, &timeEndSGBMNonBinary);

            nCompTimeSGBMNonBinary = ((uint64_t)timeEndSGBMNonBinary.tv_sec * 1000LL + (uint64_t)timeEndSGBMNonBinary.tv_nsec / 1000000LL) -
                    ((uint64_t)timeStartSGBMNonBinary.tv_sec * 1000LL + (uint64_t)timeStartSGBMNonBinary.tv_nsec / 1000000LL);

            std::cout << "Standard SGBM computation time in ms: " << nCompTimeSGBMNonBinary << std::endl;

            // WLS FILTERING
            if(bWLS)
            {
                Ptr<cv::ximgproc::DisparityWLSFilter> pWlsFilter;
                Ptr<cv::StereoMatcher> pRightMatcher;

                pRightMatcher = cv::ximgproc::createRightMatcher(pLeftMatcherSGBM);
                pWlsFilter = cv::ximgproc::createDisparityWLSFilter(pLeftMatcherSGBM);

                // measure WLS filtering computation time in milliseconds
                timespec timeStartWLS, timeEndWLS;

                timeStartWLS.tv_sec = 0;
                timeStartWLS.tv_nsec = 0;
                timeEndWLS.tv_sec = 0;
                timeEndWLS.tv_nsec = 0;

                int nCompTimeWLS = 0;

                clock_gettime(CLOCK_MONOTONIC, &timeStartWLS);

                // disparity left to right
                pLeftMatcherSGBM->compute(vLeftImagesRect[i], vRightImagesRect[i], vLeftDispMaps[i]);
                // disparity right to left
                pRightMatcher->compute(vRightImagesRect[i], vLeftImagesRect[i], vRightDispMaps[i]);

                std::getchar();

                pWlsFilter->setLambda(nLambda);
                pWlsFilter->setSigmaColor(fSigmaColor);

                pWlsFilter->filter(vLeftDispMaps[i], vLeftImagesRect[i], vWLSDispMaps[i], vRightDispMaps[i]);

                clock_gettime(CLOCK_MONOTONIC, &timeEndWLS);

                std::cout << "WLS filtering successful!" << std::endl;

                nCompTimeWLS = ((uint64_t)timeEndWLS.tv_sec * 1000LL + (uint64_t)timeEndWLS.tv_nsec / 1000000LL) -
                        ((uint64_t)timeStartWLS.tv_sec * 1000LL + (uint64_t)timeStartWLS.tv_nsec / 1000000LL);

                std::cout << "WLS filtering computation time in ms: " << nCompTimeWLS << std::endl;

                pWlsFilter->clear();                
            }
            else
            {
                vWLSDispMaps[i] = vLeftDispMaps[i].clone();
            }
        }

        std::cout << "Disparity image calculation finished. Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        // divide disparity map by 16 (required due to the stereo matcher's output data type)
        vWLSDispMaps[i].convertTo(vWLSDispMaps[i], CV_32F, 1./16);

        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;

        cv::minMaxLoc(vWLSDispMaps[i], &minVal, &maxVal, &minLoc, &maxLoc);

        // std::cout << "Maximum found disparity: " << maxVal << std::endl;

        // BILATERAL FILTERING of transformed disparity map (OPTIONAL)
        if(bBilateralFilter)
        {
            std::vector<cv::Mat> vWLSDispMapsBilatFiltered;
            vWLSDispMapsBilatFiltered.resize(vRawSingleFrames.size());

            cv::bilateralFilter(vWLSDispMaps[i], vWLSDispMapsBilatFiltered[i], nBilateralDiameter, nSigmaColour, nSigmaSpace, BORDER_CONSTANT);

            vWLSDispMaps[i] = vWLSDispMapsBilatFiltered[i].clone();
        }

        // std::cout << "Disparity image after division by 16 and WLS filtering: " << vWLSDispMaps[i] << std::endl;

        // optional median filtering of disparity map
        // pNormDispMaps[i].convertTo(pNormDispMaps[i], CV_8U);
        // cv::medianBlur(pNormDispMaps[i], pNormDispMaps[i], nKernelSize);
        // pNormDispMaps[i].convertTo(pNormDispMaps[i], CV_32F);

        // optional Gaussian filtering of disparity map
        // cv::GaussianBlur(pNormDispMaps[i], pNormDispMaps[i], Size(nKernelSize, nKernelSize),
        // (double)0.3*((nKernelSize-1)*.1-1)+.8, (double)0.3*((nKernelSize-1)*.1-1)+.8,
        // 0, 0, BORDER_DEFAULT);

        // optional bilateral filtering of disparity map
        // cv::bilateralFilter(pNormDispMaps[i], pFilteredDispMaps[i], nBilateralDiameter,
        // nSigmaColour, nSigmaSpace, BORDER_CONSTANT);

        // pNormDispMaps[i] = pFilteredDispMaps[i].clone();


        // remove areas with high uncertainty (i.e., dark and bright areas) from disparity image

        // declarations
        cv::Mat blackAreas, whiteAreas, tempLeft;

        // grayscale conversion
        cv::cvtColor(vLeftImagesRect[i], tempLeft, CV_BGR2GRAY);

        // identify dark and bright images in grayscale image and binarize (thresholding operation)

        // dark areas

        // all pixels with intensity < 20 are set to 255
        // all pixels with intensity > 20 are set to 0

        cv::threshold(tempLeft, blackAreas, 20, 255, THRESH_BINARY_INV);

        // bright areas

        // all pixels with intensity > 240 are set to 255
        // all pixels with intensity < 240 are set to 0

        cv::threshold(tempLeft, whiteAreas, 240, 255, THRESH_BINARY);

        // data type conversion
        blackAreas.convertTo(blackAreas, CV_32F);
        whiteAreas.convertTo(whiteAreas, CV_32F);

        // remove dark and bright areas from disparity image
        vWLSDispMaps[i] = vWLSDispMaps[i] - blackAreas;
        vWLSDispMaps[i] = vWLSDispMaps[i] - whiteAreas;

        // remove negative values (resulting from subtraction of a float image)

        // disparities are set to 0 for (-vWLSDispMaps[i] > 0) and remain unchanged otherwise

        // all pixels in vWLSDispMaps[i] with "NEGATIVE intensity" after subtraction of dark and bright areas (value of these areas: 255) are set to 0
        cv::threshold(-vWLSDispMaps[i], vWLSDispMaps[i], 0, 0, THRESH_TRUNC);

        // invert resulting disparity image
        vWLSDispMaps[i] = -vWLSDispMaps[i];

        // std::cout << "Final disparity image after uncertainty reduction by area filtering: " << vWLSDispMaps[i] << std::endl;
        std::cout << "Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        // calculate and show normalized disparity image (color-coded, high disparity: red, low disparity: green)

        // to better visualize the result, apply a color map to the computed disparity image
        // code snippet taken from https://github.com/opencv/opencv_contrib/blob/565bcab2490f2275c0e855c7e5e61b8212806722/modules/structured_light/samples/pointcloud.cpp#L262
        // (accessed on 04/02/2021)
        double min;
        double max;
        cv::minMaxIdx(vWLSDispMaps[i], &min, &max);

        cv::Mat cm_disp, scaledDisparityMap;
        std::cout << "disp min " << min << std::endl << "disp max " << max << std::endl;
        cv::convertScaleAbs(vWLSDispMaps[i], scaledDisparityMap, 255/(max - min));
        cv::applyColorMap(scaledDisparityMap, cm_disp, cv::COLORMAP_JET);

        // show the result
        cv::resize(cm_disp, cm_disp, cv::Size(640, 480));

        cv::namedWindow("Normalized Disparity Image",WINDOW_AUTOSIZE);
        cv::imshow("Normalized Disparity Image", cm_disp);

        std::cout << "Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("Normalized Disparity Image");

        // cv::Mat dispNormalizedNoWLS(vLeftDispMaps[i].rows,vLeftDispMaps[i].cols, CV_32F);

        // cv::normalize(vLeftDispMaps[i], dispNormalizedNoWLS, 0, 1, NORM_MINMAX, CV_32F);

        // cv::namedWindow("Normalized Disparity Image before WLS",WINDOW_AUTOSIZE);
        // cv::imshow("Normalized Disparity Image before WLS",dispNormalizedNoWLS);

        // cv::waitKey(0);
        // cv::destroyWindow("Normalized Disparity Image before WLS");


        // measure reprojection time in milliseconds
        timespec timeStartPointCloud, timeEndPointCloud;

        int nCompTimePointCloud = 0;

        clock_gettime(CLOCK_MONOTONIC, &timeStartPointCloud);

        // compute stereo reconstruction in coordinate system of left virtual camera (point coordinates are returned in calibration pattern units, here: mm)
        cv::reprojectImageTo3D(vWLSDispMaps[i], vDepthMaps[i], pQ, false, CV_32F);

        clock_gettime(CLOCK_MONOTONIC, &timeEndPointCloud);

        nCompTimePointCloud = ((uint64_t)timeEndPointCloud.tv_sec * 1000LL + (uint64_t)timeEndPointCloud.tv_nsec / 1000000LL) -
                ((uint64_t)timeStartPointCloud.tv_sec * 1000LL + (uint64_t)timeStartPointCloud.tv_nsec / 1000000LL);

        std::cout << "Point cloud computation time in ms: " << nCompTimePointCloud << std::endl;

        nRowMark.resize(1);
        nColMark.resize(1);
        nRowMark.at(0)=0;
        nColMark.at(0)=0;

        // save resulting point cloud
        savePointCloud("stereo_reconstructions", "point_cloud", vDepthMaps, nRowMark, nColMark, vLeftImagesRect);

        std::cout << "Reconstruction complete. Please hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);
    }

    // if droplet impact site prediction desired after stereo reconstruction complete
    if(bImpactSitePrediction)
    {
        // set tolerance values

        // std::cout << "Set tol_pi value: ";
        // std::cin >> tol_pi; // get user input from the keyboard

        if(bRodLens)
        {
            // maximum allowed distance to trajectory fit plane PI in mm
            tol_pi = 0.4;

            // maximum allowed distance to linear trajectory approximation (DEPRECATED MODEL) in mm (radius of tolerance cylinder around T_lin)
            tol_lin = 0.2;

            // maximum allowed distance to parabolic trajectory approximation (STANDARD MODEL) in mm (radius of tolerance tube around T_para)
            tol_para = 0.2;
        }
        // double tolerances for fiberoptic system with lower image resolution
        else
        {
            // maximum allowed distance to trajectory fit plane PI in mm
            tol_pi = 0.8;

            // maximum allowed distance to linear trajectory approximation (DEPRECATED MODEL) in mm (radius of tolerance cylinder around T_lin)
            tol_lin = 0.4;

            // maximum allowed distance to parabolic trajectory approximation (STANDARD MODEL) in mm (radius of tolerance tube around T_para)
            tol_para = 0.4;
        }

        // read trajectory model parameters from file "TrajInfos.yml"

        // load defining parameters of plane PI from file "TrajInfos.yml"
        std::string filename = "TrajInfos.yml";
        cv::FileStorage fs_traj;

        fs_traj.open(filename, cv::FileStorage::READ);

        // parameter declarations
        cv::Vec3f n_PI_0;   // normal vector of Hesse normal form of plane PI
        double d;           // distance from plane PI to coordinate system origin

        // load parameters of best-fit plane PI in Hesse normal form from file "TrajInfos.yml"
        fs_traj["n_PI_0"] >> n_PI_0;
        fs_traj["d"] >> d;

        // load parameters of linear trajectory from file "TrajInfos.yml"
        // parameter declarations
        cv::Point3f suppVecLin = cv::Point3f(0.0,0.0,0.0);
        cv::Point3f dirVecLin = cv::Point3f(0.0,0.0,0.0);

        // load parameters from file "TrajInfos.yml"
        fs_traj["dirVecLin"] >> dirVecLin;
        fs_traj["suppVecLin"] >> suppVecLin;

        // load parameter vector of parabolic trajectory from file "TrajInfos.yml"

        // parameter vector declaration
        cv::Mat vF = cv::Mat::zeros(9, 1, CV_32F);

        // load parameter vector from file "TrajInfos.yml"
        fs_traj["F"] >> vF;
        fs_traj.release();

        // go through all points in "vDepthMaps[0]" and calculate impact site prediction with (1) linear (DEPRECATED MODEL) and (2) parabolic (STANDARD MODEL) trajectory approximation method

        // temporary helper point
        cv::Point3f TempSamplingPoint = cv::Point3f(0.0,0.0,0.0);

        // stereo reconstruction points, stored as 3D points in vector
        std::vector<cv::Point3f> vSamplingPoints;

        // for all rows of depth map vDepthMaps[0]
        for(int r = 0; r < vDepthMaps[0].rows; r++)
        {
            // for all columns of depth map vDepthMaps[0]
            for(int c = 0; c < vDepthMaps[0].cols; c++)
            {
                TempSamplingPoint.x = vDepthMaps[0].at<Vec3f>(r, c)[0];
                TempSamplingPoint.y = vDepthMaps[0].at<Vec3f>(r, c)[1];
                TempSamplingPoint.z = vDepthMaps[0].at<Vec3f>(r, c)[2];

                // add current point to vector of all points of stereo reconstruction
                vSamplingPoints.push_back(TempSamplingPoint);
            }
        }

        // pre-processing: remove all points from stereo reconstruction that are too far away from plane PI

        // measure linear impact site computation time in milliseconds
        timespec timeStartPi, timeEndPi;

        int nCompTimePi = 0;

        // declare vector for impact site candidate points
        std::vector<cv::Point3f> ImpactSiteCandidates;

        ImpactSiteCandidates.resize(0);

        clock_gettime(CLOCK_MONOTONIC, &timeStartPi);

        // go through spatially reconstructed target points
        for (unsigned int i=0; i<vSamplingPoints.size(); ++i)
        {
            // if distance to fit plane lower than tol_pi: add point to set of impact site candidates
            if ((double)std::abs(vSamplingPoints[i].dot(n_PI_0)-d) < tol_pi)
            {
                ImpactSiteCandidates.push_back(vSamplingPoints[i]);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &timeEndPi);

        nCompTimePi = ((uint64_t)timeEndPi.tv_sec * 1000LL + (uint64_t)timeEndPi.tv_nsec / 1000000LL) -
                ((uint64_t)timeStartPi.tv_sec * 1000LL + (uint64_t)timeStartPi.tv_nsec / 1000000LL);

        std::cout << "Reduced set computation time in ms: " << nCompTimePi << std::endl;

        std::cout << ImpactSiteCandidates.size() << " points are closer than tol_pi to plane PI!" << std::endl;

        // declaration for vector of reduced candidate sets
        std::vector<cv::Mat> vRedCandidateSet;

        // declaration for reduced candidate set
        cv::Mat pRedCandidateSet;

        pRedCandidateSet = cv::Mat(ImpactSiteCandidates);

        // conversion into double
        pRedCandidateSet.convertTo(pRedCandidateSet, CV_64F);

        vRedCandidateSet.push_back(pRedCandidateSet);

        savePointCloud("reduced_set","candidates",vRedCandidateSet);

        // calculation of impact site prediction (linear trajectory model, DEPRECATED)

        // measure linear impact site computation time in milliseconds
        timespec timeStartLinPrediction, timeEndLinPrediction;

        int nCompTimeLinPrediction = 0;

        // initialize linear impact site prediction (very far away)
        cv::Point3f P_imp_lin = cv::Point3f(0.0,0.0,5000.0);

        // declare vector for linear impact site prediction (DEPRECATED MODEL)
        std::vector<cv::Point3f> vImpactSiteCandidatesLin;

        vImpactSiteCandidatesLin.resize(0);

        clock_gettime(CLOCK_MONOTONIC, &timeStartLinPrediction);

        for (unsigned int i=0; i<ImpactSiteCandidates.size(); ++i)
        {

            // if current candidate in tolerance cylinder around linear trajectory approximation AND closer to left virtual camera in z direction than previous candidate
            if (cv::norm((ImpactSiteCandidates[i] - suppVecLin).cross(dirVecLin))/cv::norm(dirVecLin) < tol_lin && ImpactSiteCandidates[i].z < P_imp_lin.z)
            {
                // current point is impact site candidate
                P_imp_lin = ImpactSiteCandidates[i];

                // std::cout << "Current distance to fit line in mm: " << cv::norm((suppVecLin - ImpactSiteCandidates[i]).cross(dirVecLin))/cv::norm(dirVecLin) << std::endl;
            }
        }

        vImpactSiteCandidatesLin.push_back(P_imp_lin);

        clock_gettime(CLOCK_MONOTONIC, &timeEndLinPrediction);

        nCompTimeLinPrediction = ((uint64_t)timeEndLinPrediction.tv_sec * 1000LL + (uint64_t)timeEndLinPrediction.tv_nsec / 1000000LL) -
                ((uint64_t)timeStartLinPrediction.tv_sec * 1000LL + (uint64_t)timeStartLinPrediction.tv_nsec / 1000000LL);

        std::cout << "Linear impact site prediction computation time in ms: " << nCompTimeLinPrediction << std::endl;

        // save linear impact site prediction

        cv::Mat pImpPredLin;

        std::vector<cv::Mat> vImpPredLin;

        vImpPredLin.resize(0);

        pImpPredLin = cv::Mat(vImpactSiteCandidatesLin);

        // convert into double
        pImpPredLin.convertTo(pImpPredLin, CV_64F);

        vImpPredLin.push_back(pImpPredLin);

        savePointCloud("P_imp_lin","pred_lin",vImpPredLin);

        // _____________________________________________________calculation of impact site prediction with parabolic trajectory model (STANDARD)______________________________________________

        // initialize parabolic impact site prediction
        cv::Point3f P_imp_para = cv::Point3f(0.0,0.0,5000.0);

        // declare vector for parabolic impact site prediction
        std::vector<cv::Point3f> vImpactSiteCandidatesPara;

        vImpactSiteCandidatesPara.resize(0);

        // initialize distance between 3D point and parabolic trajectory model (here in mm)
        double dist_para = 0.0;

        // measure parabolic impact site computation time in milliseconds
        timespec timeStartParaPrediction, timeEndParaPrediction;

        int nCompTimeParaPrediction = 0;

        clock_gettime(CLOCK_MONOTONIC, &timeStartParaPrediction);

        // for all impact site candidate points
        for (unsigned int i=0; i<ImpactSiteCandidates.size(); ++i)
        {    
            // calculate shortest distance between current point and parabolic trajectory model
            dist_para = calculateAverageGeometricDistanceParaSinglePoint(vF, ImpactSiteCandidates[i]);

            // if current candidate located in tolerance tube around parabolic trajectory approximation AND closer to left virtual camera in z direction than previous candidate:
            // update impact site prediction
            if (dist_para < tol_para && ImpactSiteCandidates[i].z < P_imp_para.z)
            {
                P_imp_para = ImpactSiteCandidates[i];

                // std::cout << "Current distance to parabola in mm: " << std::sqrt(dist_para_squared) << std::endl;
            }
        }

        vImpactSiteCandidatesPara.push_back(P_imp_para);

        clock_gettime(CLOCK_MONOTONIC, &timeEndParaPrediction);

        nCompTimeParaPrediction = ((uint64_t)timeEndParaPrediction.tv_sec * 1000LL + (uint64_t)timeEndParaPrediction.tv_nsec / 1000000LL) -
                ((uint64_t)timeStartParaPrediction.tv_sec * 1000LL + (uint64_t)timeStartParaPrediction.tv_nsec / 1000000LL);

        std::cout << "Parabolical impact site prediction computation time in ms: " << nCompTimeParaPrediction << std::endl;

        // save parabolic impact site prediction

        cv::Mat pImpPredPara;

        std::vector<cv::Mat> vImpPredPara;

        vImpPredPara.resize(0);

        pImpPredPara = cv::Mat(vImpactSiteCandidatesPara);

        // convert into double
        pImpPredPara.convertTo(pImpPredPara, CV_64F);

        vImpPredPara.push_back(pImpPredPara);

        savePointCloud("P_imp_para","pred_para",vImpPredPara);

        std::cout << "vImpPredPara[0]: " << vImpPredPara[0] << std::endl;

        // display impact site predictions on rectified left endoscopic image "vLeftImagesRect[0]"

        vMarkedImages.resize(vDepthMaps.size());

        // image coordinates of impact site predictions
        cv::Point2f P_imp_lin_2D = cv::Point2f(0.0,0.0);
        cv::Point2f P_imp_para_2D = cv::Point2f(0.0,0.0);

        vMarkedImages[0] = vLeftImagesRect[0].clone();

        // initialize vectors for zero rotation and translation

        std::vector<float> vRotZero(3, 0.0);
        std::vector<float> vTranslatZero(3, 0.0);

        // initialize vectors for compatibility with cv::projectPoints()
        std::vector<cv::Point2f> vImpPredLin_2D;
        std::vector<cv::Point2f> vImpPredPara_2D;

        vImpPredLin_2D.resize(1);
        vImpPredPara_2D.resize(1);

        std::vector<cv::Vec3f> vP_imp_lin;
        std::vector<cv::Vec3f> vP_imp_para;

        vP_imp_lin.resize(1);
        vP_imp_para.resize(1);

        vP_imp_lin[0][0] = P_imp_lin.x;
        vP_imp_lin[0][1] = P_imp_lin.y;
        vP_imp_lin[0][2] = P_imp_lin.z;

        vP_imp_para[0][0] = P_imp_para.x;
        vP_imp_para[0][1] = P_imp_para.y;
        vP_imp_para[0][2] = P_imp_para.z;

        cv::Rect rectSubMat = cv::Rect(0,0,3,3);

        cv::Mat pNewCamMatrixLeft = pP1Left(rectSubMat);

        // project spatial impact site predictions into left sub-image
        cv::projectPoints(vP_imp_lin, vRotZero, vTranslatZero, pNewCamMatrixLeft, cv::noArray(), vImpPredLin_2D);
        cv::projectPoints(vP_imp_para, vRotZero, vTranslatZero, pNewCamMatrixLeft, cv::noArray(), vImpPredPara_2D);

        P_imp_lin_2D = vImpPredLin_2D[0];
        P_imp_para_2D = vImpPredPara_2D[0];

        std::cout << "P_imp_lin_2D: " << P_imp_lin_2D << std::endl;
        std::cout << "P_imp_para_2D: " << P_imp_para_2D << std::endl;

        if(bRodLens)
        {
            // mark linear impact site prediction (DEPRECATED MODEL) on image in blue

            // horizontal line
            cv::line(vMarkedImages[0], Point(P_imp_lin_2D.x-15, P_imp_lin_2D.y),
                     Point(P_imp_lin_2D.x+15, P_imp_lin_2D.y), Scalar(255,0,0), 1, LINE_AA);
            // vertical line
            cv::line(vMarkedImages[0], Point(P_imp_lin_2D.x, P_imp_lin_2D.y-15),
                     Point(P_imp_lin_2D.x, P_imp_lin_2D.y+15), Scalar(255,0,0), 1, LINE_AA);

            // mark parabolic impact site prediction (STANDARD METHOD) on image in red

            // horizontal line
            cv::line(vMarkedImages[0], Point(P_imp_para_2D.x-15, P_imp_para_2D.y),
                     Point(P_imp_para_2D.x+15, P_imp_para_2D.y), Scalar(0,0,255), 1, LINE_AA);
            // vertical line
            cv::line(vMarkedImages[0], Point(P_imp_para_2D.x, P_imp_para_2D.y-15),
                     Point(P_imp_para_2D.x, P_imp_para_2D.y+15), Scalar(0,0,255), 1, LINE_AA);

            // show marked left rectified image

            cv::namedWindow("Impact Site Predictions");
            cv::imshow("Impact Site Predictions", vMarkedImages[0]);
        }
        else
        {
            // resize image for fiberoptical system (low resolution)

            vMarkedImagesResized.resize(1);

            cv::resize(vMarkedImages[0], vMarkedImagesResized[0], cv::Size(), fDisplayScale, fDisplayScale, cv::INTER_LINEAR);

            // mark linear impact site prediction on image in blue

            // horizontal line
            cv::line(vMarkedImagesResized[0], Point(fDisplayScale*P_imp_lin_2D.x-15, fDisplayScale*P_imp_lin_2D.y),
                     Point(fDisplayScale*P_imp_lin_2D.x+15, fDisplayScale*P_imp_lin_2D.y), Scalar(255,0,0), 1, LINE_AA);
            // vertical line
            cv::line(vMarkedImagesResized[0], Point(fDisplayScale*P_imp_lin_2D.x,fDisplayScale*P_imp_lin_2D.y-15),
                     Point(fDisplayScale*P_imp_lin_2D.x, fDisplayScale*P_imp_lin_2D.y+15), Scalar(255,0,0), 1, LINE_AA);

            // mark parabolic impact site prediction on image in red

            // horizontal line
            cv::line(vMarkedImagesResized[0], Point(fDisplayScale*P_imp_para_2D.x-15, fDisplayScale*P_imp_para_2D.y),
                     Point(fDisplayScale*P_imp_para_2D.x+15, fDisplayScale*P_imp_para_2D.y), Scalar(0,0,255), 1, LINE_AA);
            // vertical line
            cv::line(vMarkedImagesResized[0], Point(fDisplayScale*P_imp_para_2D.x, fDisplayScale*P_imp_para_2D.y-15),
                     Point(fDisplayScale*P_imp_para_2D.x, fDisplayScale*P_imp_para_2D.y+15), Scalar(0,0,255), 1, LINE_AA);

            // show marked left rectified image

            cv::namedWindow("Impact Site Predictions");
            cv::imshow("Impact Site Predictions", vMarkedImagesResized[0]);
        }

        // if impact site prediction error estimation desired
        // (only feasible if frame sequence showing droplet flight is provided!)
        if(bPredErrorEstimation && !bSingleFrameReconstruction)
        {
            // find frame showing instant of droplet impact

            // declarations for single frames
            cv::Mat pBlobTempFrameLeft, pBlobTempFrameRight, pBlobTempFrame, pFirstFrame;

            // store first frame of sequence in color
            pFirstFrame = vRawSingleFrames[0].clone();

            // vectors for single frames
            std::vector<cv::Mat> vTrajLeft, vTrajLeftRect, vTrajRight, vTrajRightRect, vImagesWithCircles, vCompleteTrajRectImages;

            vTrajLeft.resize(vRawSingleFrames.size());
            vTrajRight.resize(vRawSingleFrames.size());
            vTrajLeftRect.resize(vRawSingleFrames.size());
            vTrajRightRect.resize(vRawSingleFrames.size());
            vImagesWithCircles.resize(vRawSingleFrames.size());
            vCompleteTrajRectImages.resize(vRawSingleFrames.size());

            // blob detection parameters

            cv::SimpleBlobDetector::Params params;

            params.filterByArea = true;

            int minArea, maxArea, minThreshold, maxThreshold, thresholdStep, minDistBetweenBlobs;

            minArea = 25;
            maxArea = 800;
            minThreshold = 10;
            maxThreshold = 220;
            thresholdStep = 1;
            minDistBetweenBlobs = 115;

            // rod lens system parameters
            if(bRodLens)
            {
                minArea = 40;
                maxArea = 1000;
                minThreshold = 5;
                maxThreshold = 240;
                thresholdStep = 1;
                minDistBetweenBlobs = 100;

                params.minArea = 40;
                params.maxArea = 1000;
                params.minThreshold = 5;
                params.maxThreshold = 240;
                params.thresholdStep = 1;
                params.minDistBetweenBlobs = 100;
            }
            // fiberoptic system parameters
            else
            {
                minThreshold = 5;
                maxThreshold = 250;
                thresholdStep = 1;
                minDistBetweenBlobs = 80;

                params.minArea = 8;
                params.maxArea = 20;
                params.minThreshold = 5;
                params.maxThreshold = 250;
                params.thresholdStep = 1;
                params.minDistBetweenBlobs = 80;
            }

            // for all frames in frame sequence
            for (unsigned int i = 0; i < vRawSingleFrames.size(); i++)
                {
                    // GRAYSCALE conversion (required for blob detection)
                    cv::cvtColor(vRawSingleFrames[i], vRawSingleFrames[i], CV_BGR2GRAY);

                    // extract ROIs
                    splitImages(vRawSingleFrames[i], vTrajLeft[i], vTrajRight[i]);

                    // rectify left image
                    cv::remap(vTrajLeft[i], vTrajLeftRect[i], pMapLeft1, pMapLeft2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

                    // rectify right image
                    cv::remap(vTrajRight[i], vTrajRightRect[i], pMapRight1, pMapRight2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

                    // concatenate images
                    cv::hconcat(vTrajLeftRect[i], vTrajRightRect[i], vCompleteTrajRectImages[i]);
                }


            // find frames with droplet in midair

            // calculate simple difference image

            // declarations
            std::vector<cv::Mat> vForeground;
            std::vector<int> vImagesWithMovingDroplet;
            int nIndexLastMovingDroplet = 0;

            int nMinWhitePixels, nFoundWhites;

            if(bRodLens)
            {
                nMinWhitePixels = 10;
                nFoundWhites = 0;
            }
            else
            {
                nMinWhitePixels = 4;
                nFoundWhites = 0;
            }

            vForeground.resize(vRawSingleFrames.size());

            // find first and last frame showing droplet in midair
            std::cout << "Finding droplet impact frame..." << std::endl;

            // iterate over all frames of sequence
            for(unsigned int i = 1; i < vRawSingleFrames.size();i++)
            {
                // calculate simple difference image
                cv::absdiff(vRawSingleFrames[i-1],vRawSingleFrames[i], vForeground[i]);

                // binarize difference image to extract droplet

                // rod lens system
                if(bRodLens)
                {
                    cv::threshold(vForeground[i], vForeground[i], 20,255, THRESH_BINARY);
                }
                // fiberoptic system
                else
                {
                    cv::threshold(vForeground[i], vForeground[i], 5, 255, THRESH_BINARY);
                }

                // morphological closing for faulty pixel elimination
                cv::morphologyEx(vForeground[i], vForeground[i], MORPH_CLOSE,
                             getStructuringElement(MORPH_RECT, Size(nClosingRadius,nClosingRadius), Point(-1,-1)), Point(-1,-1),
                             1, BORDER_CONSTANT, 0);

                // counting of white pixels
                nFoundWhites = countNonZero(vForeground[i]);

                // if more than nMinWhitePixels found --> droplet in midair
                if (nFoundWhites > nMinWhitePixels)
                    {
                        vImagesWithMovingDroplet.push_back(i);
                        nIndexLastMovingDroplet = i;
                    }
            }

            // if no frames with moving droplet found --> end program execution
            if (vImagesWithMovingDroplet.size()==0)
            {
                std::cout << "No frames with droplet in midair found!\n" << "Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                // end program execution
                return 0;
            }

            // return number of frames showing droplet in midair
            std::cout << vImagesWithMovingDroplet.size() << " frames with moving droplet found! \n" << "Hit key to proceed..." << std::endl;

            // wait for user input
            cv::waitKey(0);

            // identify index of last valid frame
            int nLastFrameKey;
            int nCurrentIndex = nIndexLastMovingDroplet;
            int nLastFrame;

            std::cout << "Navigation:\n"
                         "k -> Show previous frame\n"
                         "l -> Show next frame\n"
                         "Enter -> Select current frame as droplet impact frame" << std::endl;

            while (true)
            {
                cv::imshow("LastFrame", vRawSingleFrames[nCurrentIndex]);

                // register pressed key
                nLastFrameKey = cv::waitKeyEx();

                // go back one frame
                if (nLastFrameKey == 107)
                {
                    nCurrentIndex -=1;
                }
                // advance one frame
                else if (nLastFrameKey == 108)
                {
                    nCurrentIndex +=1;

                    if (nCurrentIndex > nIndexLastMovingDroplet)
                    {
                        nCurrentIndex -= 1;
                    }
                }

                // if enter key pressed
                else if (nLastFrameKey == 13)
                {
                    nLastFrame = nCurrentIndex;
                    break;
                }
                else
                {
                    std::cout << "Invalid input!" << std::endl;
                }

            }

            cv::destroyWindow("LastFrame");

            std::cout << "Index of frame showing droplet impact: " << nLastFrame << std::endl;
            std::cout << "Hit key to proceed..." << std::endl;

            cv::namedWindow("Selected Droplet Impact Frame",WINDOW_AUTOSIZE);
            cv::imshow("Selected Droplet Impact Frame",vRawSingleFrames[nLastFrame]);

            // wait for user input
            cv::waitKey();
            cv::destroyWindow("Selected Droplet Impact Frame");

            // declarations for blob detection
            cv::Mat pBlobSearchTestLeft, pBlobDrawTestLeft, pBlobSearchTestRight, pBlobDrawTestRight;
            std::vector<Mat> vDropletImpact;
            int nBlobCounter = 0;

            vDropletImpact.resize(1);

            // perform blob detection

            vDropletImpact[0] = vRawSingleFrames[nLastFrame].clone();

            // copy frame with droplet impact to visualize impact of blob detector parameter values
            pBlobTempFrameLeft = vTrajLeftRect[nLastFrame].clone();
            pBlobTempFrameRight = vTrajRightRect[nLastFrame].clone();

            // show frame
            cv::namedWindow("Left blob detector input frame (rectified)", WINDOW_AUTOSIZE);
            cv::imshow("Left blob detector input frame (rectified)", vTrajLeftRect[nLastFrame]);

            std::cout << "Hit key to proceed..." << std::endl;

            // wait for user input
            cv::waitKey(0);

            cv::destroyWindow("Left blob detector input frame (rectified)");

            // convert frame to 8 bit grayscale image
            pBlobTempFrame.convertTo(pBlobTempFrame, CV_8UC1);

            std::cout << "Configuration of blob detector parameters.\n" <<
                         "Parameters must be adjusted to detect impacted droplet!\n" <<
                         "Hit escape key to close all windows and proceed." << std::endl;

            cv::Mat TempInvImgLeft, TempInvImgRight;

            // test current blob detector parameters
            while(true)
            {
                pBlobDrawTestLeft = pBlobTempFrameLeft.clone();
                pBlobDrawTestRight = pBlobTempFrameRight.clone();

                // background subtraction
                cv::absdiff(pBlobTempFrameLeft, vTrajLeftRect[0], pBlobSearchTestLeft);
                cv::absdiff(pBlobTempFrameRight, vTrajRightRect[0], pBlobSearchTestRight);

                // morphological closing

                cv::namedWindow("Left image before morphol. closing", cv::WINDOW_AUTOSIZE);

                cv::imshow("Left image before morphol. closing", pBlobSearchTestLeft);

                cv::waitKey(0);

                cv::morphologyEx(pBlobSearchTestLeft, pBlobSearchTestLeft, MORPH_CLOSE,
                             getStructuringElement(MORPH_RECT, Size(nClosingRadius,nClosingRadius), Point(-1,-1)), Point(-1,-1),
                             1, BORDER_CONSTANT, 0);

                cv::morphologyEx(pBlobSearchTestRight, pBlobSearchTestRight, MORPH_CLOSE,
                             getStructuringElement(MORPH_RECT, Size(nClosingRadius,nClosingRadius), Point(-1,-1)), Point(-1,-1),
                             1, BORDER_CONSTANT, 0);

                cv::namedWindow("Left image after morphol. closing", cv::WINDOW_AUTOSIZE);

                cv::imshow("Left image after morphol. closing", pBlobSearchTestLeft);

                cv::waitKey(0);

                // start configuration of blob detector parameters

                // declarations
                std::vector<KeyPoint> keypointsLeft, keypointsRight;

                params.minArea = minArea;
                params.maxArea = maxArea;
                params.minThreshold = minThreshold;
                params.maxThreshold = maxThreshold;
                params.thresholdStep = thresholdStep;
                params.minDistBetweenBlobs = minDistBetweenBlobs;

                // constructor for blob detector with adapted parameters
                Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

                // invert frames for blob detection
                TempInvImgLeft = cv::Scalar::all(255) - pBlobSearchTestLeft;
                TempInvImgRight = cv::Scalar::all(255) - pBlobSearchTestRight;

                // detect blobs
                detector->detect(TempInvImgLeft, keypointsLeft);
                detector->detect(TempInvImgRight, keypointsRight);

                // draw centroid and circumference of found blobs
                cv::drawKeypoints(pBlobDrawTestLeft, keypointsLeft, pBlobDrawTestLeft, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::drawKeypoints(pBlobDrawTestRight, keypointsRight, pBlobDrawTestRight, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // parameter configuration of blob detector
                cv::namedWindow("BlobDetectParams");
                cv::createTrackbar("minArea", "BlobDetectParams", &minArea, 100);
                cv::createTrackbar("maxArea", "BlobDetectParams", &maxArea, 1500);
                cv::createTrackbar("minThreshold", "BlobDetectParams", &minThreshold, 255);
                cv::createTrackbar("maxThreshold", "BlobDetectParams", &maxThreshold, 255);
                cv::createTrackbar("thresholdStep", "BlobDetectParams", &thresholdStep, 255);
                cv::createTrackbar("minDistBetweenBlobs", "BlobDetectParams", &minDistBetweenBlobs, 255);

                std::cout << "Iteration no. " << nBlobCounter << " finished." << std::endl;
                nBlobCounter++;

                cv::namedWindow("Blob detector configuration (left image)");
                cv::imshow("Blob detector configuration (left image)", pBlobDrawTestLeft);

                cv::namedWindow("Blob detector configuration (right image)");
                cv::imshow("Blob detector configuration (right image)", pBlobDrawTestRight);

                nKey = cv::waitKey();

                // escape key closes all windows
                if (nKey == 27)
                {
                    destroyAllWindows();
                    break;
                }
            }

            // blob detector parametrization now complete

            // apply adapted blob detector parameters on all valid frames

            // declarations
            std::vector<Mat> vValidFrames;  // vector contains image data of valid frames

            std::vector<Vec2f> vFoundDropletCentersLeft2D, vFoundDropletCentersRight2D; // vectors contain found circles

            std::vector<KeyPoint> keypointsFoundLeft, keypointsFoundRight;

            cv::Mat pOnlyDropletLeft, pOnlyDropletRight;

            std::vector <int> vValidFramePositions;

            std::cout << "Blob detector parametrization complete. Detecting actual droplet impact site..." << std::endl;

            // constructor for blob detector with final parameters
            Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

            // declarations
            cv::Vec2f tempCircleLeft, tempCircleRight;

            // conversion of frames for blob detector compatibility
            vTrajLeftRect[nLastFrame].convertTo(vTrajLeftRect[nLastFrame], CV_8UC1);
            vTrajRightRect[nLastFrame].convertTo(vTrajRightRect[nLastFrame], CV_8UC1);

            vTrajLeftRect[0].convertTo(vTrajLeftRect[0], CV_8UC1);
            vTrajRightRect[0].convertTo(vTrajRightRect[0], CV_8UC1);

            // background subtraction
            cv::absdiff(vTrajLeftRect[nLastFrame], vTrajLeftRect[0], pOnlyDropletLeft);
            cv::absdiff(vTrajRightRect[nLastFrame], vTrajRightRect[0], pOnlyDropletRight);

            // morphological closings
            cv::morphologyEx(pOnlyDropletLeft, pOnlyDropletLeft, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(nClosingRadius,nClosingRadius), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            cv::morphologyEx(pOnlyDropletRight, pOnlyDropletRight, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(nClosingRadius,nClosingRadius), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            // image inversion for blob detection
            TempInvImgLeft = cv::Scalar::all(255) - pOnlyDropletLeft;
            TempInvImgRight = cv::Scalar::all(255) - pOnlyDropletRight;

            // blob detection (left image)
            detector->detect(TempInvImgLeft, keypointsFoundLeft);

            // blob detection (right image)
            detector->detect(TempInvImgRight, keypointsFoundRight);

            // filtering of faulty detections

            // exclude frame if not exactly one blob found
            if(keypointsFoundLeft.size() != 1)
            {
                std::cout << "Error: " << keypointsFoundLeft.size() << " blobs found in left image!" << std::endl;
            }

            // exclude frame if not exactly one blob found
            if(keypointsFoundRight.size() != 1)
            {
                std::cout << "Error: " << keypointsFoundRight.size() << " blobs found in right image!" << std::endl;
            }

            // frames are rectified --> vertical coordinates of blob centroids must be similar
            if (((keypointsFoundLeft[0].pt.y-5) > keypointsFoundRight[0].pt.y) || ((keypointsFoundLeft[0].pt.y+5) < keypointsFoundRight[0].pt.y))
            {
                std::cout << "Blobs are not vertically aligned! Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                return 0;
            }

            // disparities must be positive
            if ((keypointsFoundLeft[0].pt.x - keypointsFoundRight[0].pt.x) <= 0)
            {
                std::cout << "Negative disparity identified! Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                return 0;
            }

            // draw found blobs

            // draw centroid and circumference of found blobs (left sub-image)
            cv::drawKeypoints(vTrajLeftRect[nLastFrame], keypointsFoundLeft, vTrajLeftRect[nLastFrame], cv::Scalar(255,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            cv::line(vTrajLeftRect[nLastFrame], Point(keypointsFoundLeft[0].pt.x-5.0, keypointsFoundLeft[0].pt.y), Point(keypointsFoundLeft[0].pt.x+5.0, keypointsFoundLeft[0].pt.y), Scalar(255,255,0), .5, 8);
            cv::line(vTrajLeftRect[nLastFrame], Point(keypointsFoundLeft[0].pt.x, keypointsFoundLeft[0].pt.y-5.0), Point(keypointsFoundLeft[0].pt.x, keypointsFoundLeft[0].pt.y+5.0), Scalar(255,255,0), .5, 8);

            // draw centroid and circumference of found blobs (right sub-image)
            cv::drawKeypoints(vTrajRightRect[nLastFrame], keypointsFoundRight, vTrajRightRect[nLastFrame], cv::Scalar(255,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            cv::line(vTrajRightRect[nLastFrame], Point(keypointsFoundRight[0].pt.x-5.0, keypointsFoundRight[0].pt.y), Point(keypointsFoundRight[0].pt.x+5.0, keypointsFoundRight[0].pt.y), Scalar(255,255,0), .5, 8);
            cv::line(vTrajRightRect[nLastFrame], Point(keypointsFoundRight[0].pt.x, keypointsFoundRight[0].pt.y-5.0), Point(keypointsFoundRight[0].pt.x, keypointsFoundRight[0].pt.y+5.0), Scalar(255,255,0), .5, 8);

            // concatenate images
            cv::hconcat(vTrajLeftRect[nLastFrame], vTrajRightRect[nLastFrame], vImagesWithCircles[nLastFrame]);

            vValidFramePositions.push_back(nLastFrame);
            vValidFrames.push_back(vImagesWithCircles[nLastFrame]);

            // store found droplet centroids in vector
            if(keypointsFoundLeft.size() != 0)
            {
                tempCircleLeft[0] = keypointsFoundLeft[0].pt.x;
                tempCircleLeft[1] = keypointsFoundLeft[0].pt.y;
            }

            if(keypointsFoundRight.size() !=0)
            {
                tempCircleRight[0] = keypointsFoundRight[0].pt.x;
                tempCircleRight[1] = keypointsFoundRight[0].pt.y;
            }

            vFoundDropletCentersLeft2D.push_back(tempCircleLeft);
            vFoundDropletCentersRight2D.push_back(tempCircleRight);

            saveOrShowImage(vValidFrames, "Detected ground truth droplet impact site", "detected_impact_site", false);

            // warning: less than one circle found in droplet impact frame
            if (vFoundDropletCentersLeft2D.size() < 1 || vFoundDropletCentersRight2D.size() < 1)
            {
                std::cout << "Warning: droplet impact site not detected." << std::endl;
                std::cout << "Calculation of impact site prediction accuracy not possible. Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                return 0;
            }

            // triangulate observed droplet impact site in space
            std::cout << "Calculating disparity of observed droplet impact site..." << std::endl;

            // declarations
            std::vector<Vec3f> vTrajDisparity, vTrajPoints;

            vTrajDisparity.resize(vFoundDropletCentersLeft2D.size());
            vTrajPoints.resize(vFoundDropletCentersLeft2D.size());

            // calculate disparity for all found droplet centroids
            for (unsigned int i  = 0; i < vFoundDropletCentersLeft2D.size(); i++)
            {
                // read horizontal coordinate
                vTrajDisparity[i][0] = vFoundDropletCentersLeft2D[i][0];

                // read vertical coordinate
                vTrajDisparity[i][1] = vFoundDropletCentersLeft2D[i][1];

                // calculate disparity
                vTrajDisparity[i][2] = vFoundDropletCentersLeft2D[i][0] - vFoundDropletCentersRight2D[i][0];
            }

            // transform disparity to 3D point
            cv::perspectiveTransform(vTrajDisparity, vTrajPoints, pQ);

            // vTrajPoints[0][0-2] now contains the 3D coordinates of the real droplet impact site

            if(bRodLens)
            {
                // add horizontal line for observed impact site in white
                cv::line(vMarkedImages[0], Point(vFoundDropletCentersLeft2D[0][0]-15, vFoundDropletCentersLeft2D[0][1]),
                         Point(vFoundDropletCentersLeft2D[0][0]+15, vFoundDropletCentersLeft2D[0][1]), Scalar(255,255,255), 1, LINE_AA);

                // add vertical line for observed impact site in white
                cv::line(vMarkedImages[0], Point(vFoundDropletCentersLeft2D[0][0], vFoundDropletCentersLeft2D[0][1]-15),
                         Point(vFoundDropletCentersLeft2D[0][0], vFoundDropletCentersLeft2D[0][1]+15), Scalar(255,255,255), 1, LINE_AA);

                std::cout << "keypointsFoundLeft[0].size: " << keypointsFoundLeft[0].size << std::endl;

                // add circumference of detected droplet at impact in white
                cv::circle(vMarkedImages[0], Point(vFoundDropletCentersLeft2D[0][0], vFoundDropletCentersLeft2D[0][1]), std::round(0.5*keypointsFoundLeft[0].size), Scalar(225,255,255), 1, LINE_AA);

                // add line between linear prediction and actual impact site to left rectified image in blue

                cv::line(vMarkedImages[0], Point(vFoundDropletCentersLeft2D[0][0], vFoundDropletCentersLeft2D[0][1]),
                         Point(P_imp_lin_2D.x, P_imp_lin_2D.y), Scalar(255,0,0), 1, LINE_AA);

                // add line between parabolic prediction and actual impact site to left rectified image in red

                cv::line(vMarkedImages[0], Point(vFoundDropletCentersLeft2D[0][0], vFoundDropletCentersLeft2D[0][1]),
                         Point(P_imp_para_2D.x, P_imp_para_2D.y), Scalar(0,0,255), 1, LINE_AA);

                // calculate inter-point distances in pixels in left rectified image

                fDistancePxLin = (float)std::sqrt(std::pow(vFoundDropletCentersLeft2D[0][0]-P_imp_lin_2D.x,2)+std::pow(vFoundDropletCentersLeft2D[0][1]-P_imp_lin_2D.y,2));

                fDistancePxPara = (float)std::sqrt(std::pow(vFoundDropletCentersLeft2D[0][0]-P_imp_para_2D.x,2)+std::pow(vFoundDropletCentersLeft2D[0][1]-P_imp_para_2D.y,2));

                std::cout << "Distance between actual impact site and linear prediction in px: " << fDistancePxLin << std::endl;
                std::cout << "Distance between actual impact site and parabolic prediction in px: " << fDistancePxPara << std::endl;

                // save marked left rectified image for manual error estimation in 2D image with known image scale

                saveOrShowImage(vMarkedImages, "Detected and predicted droplet impact sites", "result", false);
            }
            else
            {
                // add horizontal line for observed impact site in white
                cv::line(vMarkedImagesResized[0], Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0]-15.0, fDisplayScale*vFoundDropletCentersLeft2D[0][1]),
                         Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0]+15.0, fDisplayScale*vFoundDropletCentersLeft2D[0][1]), Scalar(255,255,255), 1, LINE_AA);

                // add vertical line for observed impact site in white
                cv::line(vMarkedImagesResized[0], Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0], fDisplayScale*vFoundDropletCentersLeft2D[0][1]-15.0),
                         Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0], fDisplayScale*vFoundDropletCentersLeft2D[0][1]+15.0), Scalar(255,255,255), 1, LINE_AA);

                // add circumference of detected droplet at impact in white
                cv::circle(vMarkedImagesResized[0], Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0], fDisplayScale*vFoundDropletCentersLeft2D[0][1]),
                        std::round(0.5*fDisplayScale*keypointsFoundLeft[0].size), Scalar(255,255,255), 1, LINE_AA);

                // add line between linear prediction and actual impact site to left rectified image

                cv::line(vMarkedImagesResized[0], Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0], fDisplayScale*vFoundDropletCentersLeft2D[0][1]),
                         Point(fDisplayScale*P_imp_lin_2D.x, fDisplayScale*P_imp_lin_2D.y), Scalar(0,0,0), 1, LINE_AA);

                // add line between parabolic prediction and actual impact site to left rectified image

                cv::line(vMarkedImagesResized[0], Point(fDisplayScale*vFoundDropletCentersLeft2D[0][0], fDisplayScale*vFoundDropletCentersLeft2D[0][1]),
                         Point(fDisplayScale*P_imp_para_2D.x, fDisplayScale*P_imp_para_2D.y), Scalar(255,255,255), 1, LINE_AA);

                // calculate inter-point distances in pixels in left rectified image

                fDistancePxLin = (float)std::sqrt(std::pow(vFoundDropletCentersLeft2D[0][0]-P_imp_lin_2D.x,2)+std::pow(vFoundDropletCentersLeft2D[0][1]-P_imp_lin_2D.y,2));
                fDistancePxPara = (float)std::sqrt(std::pow(vFoundDropletCentersLeft2D[0][0]-P_imp_para_2D.x,2)+std::pow(vFoundDropletCentersLeft2D[0][1]-P_imp_para_2D.y,2));

                std::cout << "Distance between actual impact site and linear prediction in px (original scale): " << fDistancePxLin << std::endl;
                std::cout << "Distance between actual impact site and parabolic prediction in px (original scale): " << fDistancePxPara << std::endl;

                // save marked left rectified image for manual error estimation in 2D image with known image scale

                saveOrShowImage(vMarkedImagesResized, "Detected and predicted droplet impact sites (magnified)", "result", false);
            }

            // show left rectified image with linear (in blue) and parabolic (in red) impact site prediction and observed (in white) impact site and inter-point distances

            cv::namedWindow("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)", WINDOW_AUTOSIZE);

            // prediction error quantification based on known image scale
            // get user input on image scale based on grid shown in target image/corners of MIT-LAR frame

            if(bRodLens)
            {
                std::cout << "Please click twice on image to indicate 35 mm distance in target plane and then click on image again to confirm!" << std::endl;
            }
            // reduce distance to 20 mm due to different optical properties of fiberoptic stereo laryngoscope
            else
            {
                std::cout << "Please click twice on image to indicate 20 mm distance in target plane and then click on image again to confirm!" << std::endl;
            }

            vClickedPointCoords.resize(0);

            cv::Mat pMarkedImagesScaleIndication;

            cv::Mat pMarkedImagesScaleIndicationMagnified;

            if(bRodLens)
            {
                pMarkedImagesScaleIndication = vMarkedImages[0].clone();

                vMarkedImagesScaleIndication.resize(0);

                // install mouse callback
                cv::setMouseCallback("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)", mouse_callback, (void*)&pMarkedImagesScaleIndication);

                while(nClickCounter<3)
                {
                    cv::imshow("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)", pMarkedImagesScaleIndication);

                    if(cv::waitKey(15) == 27) break;
                }
            }
            // magnify image for fiberoptical system (low resolution)
            else
            {
                pMarkedImagesScaleIndicationMagnified = vMarkedImagesResized[0].clone();

                vMarkedImagesResizedScaleIndication.resize(0);

                //                // enhance image contrast (OPTIONAL)

                //                // convert to HSV color space
                //                cv::cvtColor(pMarkedImagesScaleIndicationMagnified, pMarkedImagesScaleIndicationMagnified, CV_BGR2HSV);

                //                std::vector<Mat> HSV_channels(3);

                //                cv::split(pMarkedImagesScaleIndicationMagnified, HSV_channels);

                //                // apply CLAHE on Value channel
                //                Ptr<CLAHE> clahe = cv::createCLAHE();
                //                clahe->setClipLimit(4);
                //                clahe->apply(HSV_channels[2], HSV_channels[2]);

                //                cv::merge(HSV_channels, pMarkedImagesScaleIndicationMagnified);

                //                // convert image back to BGR color space
                //                cv::cvtColor(pMarkedImagesScaleIndicationMagnified, pMarkedImagesScaleIndicationMagnified, CV_HSV2BGR);

                // install mouse callback
                cv::setMouseCallback("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)", mouse_callback, (void*)&pMarkedImagesScaleIndicationMagnified);

                while(nClickCounter<3)
                {
                    cv::imshow("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)", pMarkedImagesScaleIndicationMagnified);

                    if(cv::waitKey(15) == 27) break;
                }

            }

            cv::destroyWindow("Prediction Accuracy Estimation (lin: blue, para: red, GT: white)");         

            // determine image scale

            float fDistancePx = (float)std::sqrt(std::pow(vClickedPointCoords[0].x-vClickedPointCoords[1].x,2)+std::pow(vClickedPointCoords[0].y-vClickedPointCoords[1].y,2));

            float fError2D_lin, fError2D_para, fError2D_lin_rounded, fError2D_para_rounded;

            std::string sAccuracyResultOverlayLin, sAccuracyResultOverlayPara;

            if(bRodLens)
            {
                fError2D_lin = (35.0/fDistancePx)*fDistancePxLin;
                fError2D_para = (35.0/fDistancePx)*fDistancePxPara;

                fError2D_lin_rounded = (float)std::round(fError2D_lin*100.0)/100.0;
                fError2D_para_rounded = (float)std::round(fError2D_para*100.0)/100.0;

                // add measured accuracy to image (rounded)

                sAccuracyResultOverlayLin = "Prediction accuracy (linear): " + std::to_string(fError2D_lin_rounded);
                sAccuracyResultOverlayPara = "Prediction accuracy (parabolic): " + std::to_string(fError2D_para_rounded);

                sAccuracyResultOverlayLin.erase (34,10);
                sAccuracyResultOverlayPara.erase (37,10);

                sAccuracyResultOverlayLin.append(" mm");
                sAccuracyResultOverlayPara.append(" mm");

                cv::putText(pMarkedImagesScaleIndication, sAccuracyResultOverlayLin, cv::Point(10.0, 20.0), 1, 1, Scalar(255,0,0), 1, LINE_AA);
                cv::putText(pMarkedImagesScaleIndication, sAccuracyResultOverlayPara, cv::Point(10.0, 40.0), 1, 1, Scalar(0,0,255), 1, LINE_AA);

                // save marked left rectified image with highlighted measured distance for image scale
                vMarkedImagesScaleIndication.push_back(pMarkedImagesScaleIndication);
                saveOrShowImage(vMarkedImagesScaleIndication, "Detected and predicted droplet impact sites with manual scale indication", "result_marked_scale", false);
            }
            // eliminate scale factor due to image magnification for fiberoptical system
            else
            {
                fDistancePx /= fDisplayScale;
                fError2D_lin = (20.0/fDistancePx)*fDistancePxLin;
                fError2D_para = (20.0/fDistancePx)*fDistancePxPara;

                fError2D_lin_rounded = (float)std::round(fError2D_lin*100.0)/100.0;
                fError2D_para_rounded = (float)std::round(fError2D_para*100.0)/100.0;

                // add measured (and rounded) errors to image

                sAccuracyResultOverlayLin = "Prediction accuracy (linear): " + std::to_string(fError2D_lin_rounded)  + " mm";
                sAccuracyResultOverlayPara = "Prediction accuracy (parabolic): " + std::to_string(fError2D_para_rounded) + " mm";

                sAccuracyResultOverlayLin.erase (34,10);
                sAccuracyResultOverlayPara.erase (37,10);

                sAccuracyResultOverlayLin.append(" mm");
                sAccuracyResultOverlayPara.append(" mm");

                cv::putText(pMarkedImagesScaleIndicationMagnified, sAccuracyResultOverlayLin, cv::Point(10.0, 20.0), 1, 1, Scalar(255,0,0), 1, LINE_AA);
                cv::putText(pMarkedImagesScaleIndicationMagnified, sAccuracyResultOverlayPara, cv::Point(10.0, 40.0), 1, 1, Scalar(0,0,255), 1, LINE_AA);

                // save marked left rectified image with highlighted measured distance for image scale
                vMarkedImagesResizedScaleIndication.push_back(pMarkedImagesScaleIndicationMagnified);
                saveOrShowImage(vMarkedImagesResizedScaleIndication, "Detected and predicted droplet impact sites with manual scale indication", "result_marked_scale", false);
            }

            std::cout << "Distance between linear impact site prediction and observed impact site in mm based on image scale: " << fError2D_lin << std::endl;
            std::cout << "Distance between parabolic impact site prediction and observed impact site in mm based on image scale: " << fError2D_para << std::endl;

            // save full reconstructed target point cloud with marked impact sites

            std::vector<int> nRowMark, nColMark;

            nRowMark.push_back(0);
            nColMark.push_back(0);

            // save resulting point cloud showing impacted droplet and superimposed crosshairs
            if(bRodLens)
            {
                savePointCloud("stereo_reconstructions_marked", "point_cloud_marked", vDepthMaps, nRowMark, nColMark, vMarkedImages);
            }
            else
                // restore original image dimensions
            {
                std::vector<Mat> vMarkedImagesOriginalScale;

                vMarkedImagesOriginalScale.resize(1);

                cv::resize(vMarkedImagesResized[0], vMarkedImagesOriginalScale[0], cv::Size(), 1.0/fDisplayScale, 1.0/fDisplayScale, INTER_LINEAR);

                savePointCloud("stereo_reconstructions_marked", "point_cloud_marked", vDepthMaps, nRowMark, nColMark, vMarkedImagesOriginalScale);


            }

            // estimate 3D distance between real and predicted impact sites (NOT USED FOR EVALUATION AS STEREO RECONSTRUCTION ERROR WOULD BE ADDED TO RESULT OF MEASUREMENT)

            // vTrajPoints[0][0-2] contains the 3D coordinates of the real droplet impact site in mm
            // P_imp_lin and P_imp_para contain the 3D coordinates of the predicted droplet impact site in mm

            // distance between linear impact site prediction (DEPRECATED MODEL) and real impact site in mm (neglecting difference in z direction)
            double fPredErrorEstimation3D_lin = double(std::sqrt(std::pow(vTrajPoints[0][0]-P_imp_lin.x,2) + std::pow(vTrajPoints[0][1]-P_imp_lin.y,2)));

            // distance between parabolic impact site prediction (STANDARD MODEL) and real impact site in mm (neglecting difference in z direction)
            double fPredErrorEstimation3D_para = double(std::sqrt(std::pow(vTrajPoints[0][0]-P_imp_para.x,2) + std::pow(vTrajPoints[0][1]-P_imp_para.y,2)));

            std::cout << "Distance between linear impact site prediction and observed impact site in mm (neglecting difference in z direction): " << fPredErrorEstimation3D_lin << std::endl;
            std::cout << "Distance between parabolic impact site prediction and observed impact site in mm (neglecting difference in z direction): " << fPredErrorEstimation3D_para << std::endl;
        }

        std::cout << "Impact site prediction procedure finished. Please hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        cv::destroyAllWindows();
    }

    vWLSDispMaps.clear();

    return 0;
}
