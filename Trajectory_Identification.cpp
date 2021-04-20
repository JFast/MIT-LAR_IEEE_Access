// Contributors: Adrian K. Rüppel/Maurice Krauth/Jacob F. Fast
//
// 2018-2021
//
// Trajectory_Identification.cpp
// Read raw stereolaryngoscopic (high-speed) frame sequence showing droplet flight and files with stereo camera settings and stereo calibration parameters.
// Perform spatial triangulation of detected droplet centroid positions using BLOB DETECTION and identify linear (DEPRECATED) and parabolic trajectory approximations and approximated trajectory plane.

#include <QApplication>
#include <QtCore>

#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>

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

// global variables

// project path
std::string sProjectPath;

// boolean variable for overwriting images
bool bAlreadyAsked = false;

// computation time measurement
struct timespec start, stop;
double duration;

// toggle for stereo laryngoscope system
bool bRodLens = true;

// toggle for triangulatePoints() method for 3D triangulation
bool bTriangulatePoints = false;

// variable for image rescaling
bool bRescaleFrames = false;

int nOverride = -1;

// define ROIs
int nLeftPicLeftUpperCorner[2] = {0, 0};        // in Px
int nRightPicLeftUpperCorner[2] = {0, 0};       // in Px

int nImageSize[2] = {0 ,0};                     // in Px

// calibration pattern parameters
float fDistanceBetweenCircles = 0.0;              // in mm!
Size circlesGridSize;

// FUNCTIONS

// save point cloud (without texture from image; color for all points can be set in RGB system)
static void savePointCloud(std::string sFolderName, std::string sFileName, std::vector<cv::Vec3i> vColors,
                    std::vector<Mat> &p3DClouds)
{

    // path of new directory
    std::string sDirPath = sProjectPath + "/" + sFolderName;

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
        // read point coordinates from "p3DClouds" vector of Mats
        const float fMaxZ = 200.0; // in mm
        std::size_t pointCount = 0;

        std::vector<cv::Vec3d> points;
        std::vector<std::int8_t> color(1);

        color[0] = (int)vColors[j][0];   // R value
        color[1] = (int)vColors[j][1];   // G value
        color[2] = (int)vColors[j][2];   // B value

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

                // skip points at origin
                if(currentPoint[0] == 0.0 && currentPoint[1] == 0.0 && currentPoint[2] == 0.0)
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
                             << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
        }

        points.clear();
        color.clear();

        file.close();
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
    Mat currentImage;

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
            currentImage = imread(sFullPath, IMREAD_GRAYSCALE);
        }
        else
        {
            currentImage = imread(sFullPath, IMREAD_COLOR);
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

// retrieve left and right image from CMOS of high-speed camera
void splitImages(Mat &pImage, Mat &pLeftImage, Mat &pRightImage)
{
    // define ROIs
    cv::Rect leftRectangle = cv::Rect(nLeftPicLeftUpperCorner[0], nLeftPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);
    cv::Rect rightRectangle = cv::Rect(nRightPicLeftUpperCorner[0], nRightPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);

    // extract left image
    pLeftImage = pImage(leftRectangle);

    // extract right image
    pRightImage = pImage(rightRectangle);
}

// save and/or show images
bool saveOrShowImage(std::vector<Mat> &vImages,std::string sFolderName, std::string sFileName, bool bShow)
{
    // path to new directory
    std::string sDirPath = sProjectPath + "/" + sFolderName;

    // conversion into char
    const char* cDirPath = sDirPath.c_str();

    if(vImages.empty())
    {
        std::cout << "No images available for storage! Hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);
        return 0;
    }

    // open directory
    DIR* pDir = opendir(cDirPath);

    // if directory not found
    if (pDir == NULL)
    {
        std::cout << "No directory \"" << cDirPath << "\" found. \n"
                     "Folder being created... \n" << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);

        nOverride = 1;
    }
    else
    {
        if (!bAlreadyAsked)
        {
            while(true)
            {
                std::cout << "One or more directories available for image storage. \n"
                        "Overwrite directory contents (1 = Yes, 0 = No)?" << std::endl;

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

        // if image display desired
        if (bShow == true)
        {
            cv::namedWindow("Image Display", CV_WINDOW_AUTOSIZE);

            cv::imshow("Image Display", vImages[i]);

            cv::waitKey(0);

            cv::destroyAllWindows();
        }

        // save files in directory
        if (bShow == false && nOverride == 1)
        {
            char cNumber[255];
            sprintf(cNumber, "%i", i);

            std::string sCountName = sFileName + cNumber;

            // embed image directory
            std::string sCompletePath = sDirPath + "/" + sCountName + ".png";
            const char* cCompletePath = sCompletePath.c_str();

//            // 8 bit conversion, if required
//            if(vImages[i].depth() != 1)
//            {
//                cv::Mat convertedImage;

//                cv::normalize(vImages[i], convertedImage, 0, 255, NORM_MINMAX, CV_8U);

//                vImages[i] = convertedImage;
//            }

            cv::imwrite(cCompletePath, vImages[i]);
        }
    }
    return 0;
}

// calculate average geometric distance between sampling points and linear fit function (DEPRECATED)
double calculateAverageGeometricDistanceLin(cv::Point3f &suppVec, cv::Point3f &dirVec, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // variable declaration for average geometric distance between sampling points and linear fit function in mm
    double avg_dist_lin = 0.0;

    // temporary helper point
    cv::Point3f TempTriangulatedPoint = cv::Point3f(0.0,0.0,0.0);

    std::cout << "vTriangulatedPoints.size(): " << vTriangulatedPoints.size() << std::endl;

    // for all sampling points
    for (unsigned int i=0; i<vTriangulatedPoints.size(); i++)
    {
        // convert current sampling point to point3f data type
        TempTriangulatedPoint.x = vTriangulatedPoints[i][0];
        TempTriangulatedPoint.y = vTriangulatedPoints[i][1];
        TempTriangulatedPoint.z = vTriangulatedPoints[i][2];

        // add geometric distance of current point to fit line to AGE_lin
        avg_dist_lin += cv::norm((TempTriangulatedPoint - suppVec).cross(dirVec))/cv::norm(dirVec);

        std::cout << "Current geometric distance from sampling points to linear fit in mm: " << cv::norm((TempTriangulatedPoint - suppVec).cross(dirVec))/cv::norm(dirVec) << std::endl;
    }

    // calculate average geometric distance in mm
    avg_dist_lin = avg_dist_lin/(double)vTriangulatedPoints.size();

    // output average geometric distance between sampling points and linear fit function
    std::cout << "Average distance from sampling points to linear fit in mm: " << avg_dist_lin << std::endl;

    // return average geometric distance between sampling points and linear fit function
    return avg_dist_lin;
}

// calculate average geometric distance between sampling points and parabolic fit function
double calculateAverageGeometricDistancePara(cv::Mat &vecF, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // variable for average geometric distance between sampling points and parabolic fit function
    double avg_dist_para = 0.0;

    // variable for squared distance between sampling point and parabolic trajectory model
    double dist_para_squared = 0.0;

    // variable for actual number of found distance values
    int nDistanceValues = 0;

    // vectors for coefficients and roots of cubic polynomial for distance calculation
    std::vector<double> coeffs(4,0.0);
    std::vector<double> roots(3,0.0);

    // variable for t_starred (in ms)
    double t_starred = 0.0;

    std::cout << "vecF.at<float>(0,0): " << vecF.at<float>(0,0) << std::endl;
    std::cout << "vecF.at<float>(1,0): " << vecF.at<float>(1,0) << std::endl;
    std::cout << "vecF.at<float>(2,0): " << vecF.at<float>(2,0) << std::endl;
    std::cout << "vecF.at<float>(3,0): " << vecF.at<float>(3,0) << std::endl;
    std::cout << "vecF.at<float>(4,0): " << vecF.at<float>(4,0) << std::endl;
    std::cout << "vecF.at<float>(5,0): " << vecF.at<float>(5,0) << std::endl;
    std::cout << "vecF.at<float>(6,0): " << vecF.at<float>(6,0) << std::endl;
    std::cout << "vecF.at<float>(7,0): " << vecF.at<float>(7,0) << std::endl;
    std::cout << "vecF.at<float>(8,0): " << vecF.at<float>(8,0) << std::endl;

    // coefficient of t³ (constant)
    coeffs[0] = (double)(4.0*(std::pow(vecF.at<float>(0,0),2)+std::pow(vecF.at<float>(3,0),2)+std::pow(vecF.at<float>(6,0),2)));

    std::cout << "coefficient of pow(t,3): " << coeffs.at(0) << std::endl;

    // coefficient of t² (constant)
    coeffs[1] = (double)(6.0*(vecF.at<float>(1,0)*vecF.at<float>(0,0)+vecF.at<float>(4,0)*vecF.at<float>(3,0)+vecF.at<float>(7,0)*vecF.at<float>(6,0)));

    std::cout << "coefficient of pow(t,2): " << coeffs.at(1) << std::endl;

    // temporary helper point
    cv::Point3f TempTriangulatedPoint = cv::Point3f(0.0,0.0,0.0);

    // for all sampling points
    for (unsigned int i=0; i<vTriangulatedPoints.size(); i++)
    {
        // convert current sampling point to point3f data type
        TempTriangulatedPoint.x = vTriangulatedPoints[i][0];
        TempTriangulatedPoint.y = vTriangulatedPoints[i][1];
        TempTriangulatedPoint.z = vTriangulatedPoints[i][2];

        // coefficient of t (depends of current sampling point)
        coeffs[2] = (double)(2.0*(std::pow(vecF.at<float>(1,0),2)+2.0*vecF.at<float>(2,0)*vecF.at<float>(0,0)-2.0*vecF.at<float>(0,0)*TempTriangulatedPoint.x+std::pow(vecF.at<float>(4,0),2)+2.0*vecF.at<float>(5,0)*vecF.at<float>(3,0)-2.0*vecF.at<float>(3,0)*TempTriangulatedPoint.y+std::pow(vecF.at<float>(7,0),2)+2.0*vecF.at<float>(8,0)*vecF.at<float>(6,0)-2.0*vecF.at<float>(6,0)*TempTriangulatedPoint.z));

        // coefficient of 1 (depends of current sampling point)
        coeffs[3] = (double)(2.0*(vecF.at<float>(2,0)*vecF.at<float>(1,0)-vecF.at<float>(1,0)*TempTriangulatedPoint.x+vecF.at<float>(5,0)*vecF.at<float>(4,0)-vecF.at<float>(4,0)*TempTriangulatedPoint.y+vecF.at<float>(8,0)*vecF.at<float>(7,0)-vecF.at<float>(7,0)*TempTriangulatedPoint.z));

        // calculate value of parameter t* (in ms) of point on parabolic function with lowest distance to current sampling point
        // highest-order coefficients come first
        cv::solveCubic(coeffs, roots);

        // root must be positive (first sampling point recorded at time stamp t_1 >= 0 ms)
        // only one positive root expected in vector roots[]

        if(roots[0] > 0)
        {
            t_starred = roots[0];
            nDistanceValues++;
        }
        else if(roots[1] > 0)
        {
            t_starred = roots[1];
            nDistanceValues++;
        }
        else if(roots[2] > 0)
        {
            t_starred = roots[2];
            nDistanceValues++;
        }
        else    // no physically accurate root found
        {
            continue;
        }

        std::cout << "t_starred in ms: " << t_starred << std::endl;

        // calculate shortest distance between current point and parabolical trajectory model
        dist_para_squared = (double)(std::pow(vecF.at<float>(2,0)+vecF.at<float>(1,0)*t_starred+vecF.at<float>(0,0)*std::pow(t_starred,2)-TempTriangulatedPoint.x, 2) + std::pow(vecF.at<float>(5,0)+vecF.at<float>(4,0)*t_starred+vecF.at<float>(3,0)*std::pow(t_starred,2)-TempTriangulatedPoint.y,2) + std::pow(vecF.at<float>(8,0)+vecF.at<float>(7,0)*t_starred+vecF.at<float>(6,0)*std::pow(t_starred,2)-TempTriangulatedPoint.z,2));

        // add geometric distance of current point to fit line
        avg_dist_para += std::sqrt(dist_para_squared);

        std::cout << "Current geometric distance from sampling point to parabolic fit in mm: " << std::sqrt(dist_para_squared) << std::endl;
    }

    // calculate average geometric distance    
    avg_dist_para /= (double)nDistanceValues++;

    // output average geometric distance between sampling points and parabolic fit function
    std::cout << "Average distance from sampling points to parabolic fit in mm: " << avg_dist_para << std::endl;

    // return average geometric distance between sampling points and parabolic fit function
    return avg_dist_para;
}


// calculate average geometric distance between sampling points and fit plane PI
double calculateAverageGeometricDistancePI(cv::Vec3f &vecN, double d, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // variable for average geometric distance between sampling points and fit plane PI
    double avg_dist_PI = 0.0;

    // variable for current geometric distance between sampling points and fit plane PI
    double curr_dist_PI = 0.0;

    // temporary helper point
    cv::Point3f TempTriangulatedPoint = cv::Point3f(0.0,0.0,0.0);

    // for all sampling points
    for (unsigned int i=0; i<vTriangulatedPoints.size(); i++)
    {
        // convert current sampling point to point3f data type
        TempTriangulatedPoint.x = vTriangulatedPoints[i][0];
        TempTriangulatedPoint.y = vTriangulatedPoints[i][1];
        TempTriangulatedPoint.z = vTriangulatedPoints[i][2];

        // calculate orthogonal distance between current point and fit plane PI
        curr_dist_PI = (double)std::abs(TempTriangulatedPoint.dot(vecN)-d);

        // add geometric distance of current point to fit line
        avg_dist_PI += curr_dist_PI;

        std::cout << "Current orthogonal distance from sampling point to fit plane PI in mm: " << curr_dist_PI << std::endl;
    }

    // calculate average orthogonal distance
    avg_dist_PI = avg_dist_PI/(double)vTriangulatedPoints.size();

    // output average geometric distance between sampling points and fit plane PI
    std::cout << "Average orthogonal distance from sampling point to fit plane PI in mm: " << avg_dist_PI << std::endl;

    // return average geometric distance between sampling points and fit plane PI
    return avg_dist_PI;
}

// MAIN PROGRAM

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    cv::ocl::setUseOpenCL(true);

    if ( cv::ocl::haveOpenCL ())
        {
            std::cout  <<  "OpenCL is available."  <<  std::endl ;
        }

    int nMinWhitePixels, nFoundWhites;

    int nKey = 0;

    // read current project path
    std::string pTempPath[256] = {argv[0]};
    std::string sExecPath;
    sExecPath = *pTempPath;

    // identify folder path
    int iPosOfChar = sExecPath.find_last_of("/");

    // extract project path from string
    sProjectPath = sExecPath.substr(0, iPosOfChar);

    std::cout  <<  sProjectPath  <<  std::endl;

    // initialize calibration parameter structures
    cv::Mat pCameraMatrixLeft, pCameraMatrixRight, pDistortionCoeffsLeft, pDistortionCoeffsRight;

    pCameraMatrixLeft = cv::Mat::eye(3,3, CV_64F);
    pCameraMatrixRight = cv::Mat::eye(3,3, CV_64F);

    double dRepErrorComplete = 9999.0;

    // image size
    cv::Size pImageSize;

    // rotation between virtual cameras
    cv::Mat pLeftToRightRotationMatrix;

    // translation between virtual cameras in mm
    cv::Mat pLeftToRightTranslationVector;

    // factor for OPTIONAL frame rescaling
    double dRescaleFactor = 1.0;

    // load file "settings.xml"
    std::cout << "Loading settings from file \"settings.xml\"..." << std::endl;

    cv::FileStorage set;

    // check if file "settings.xml" exists

    // end program execution if file "settings.xml" not found
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

        return 0;
    }

    // if file "settings.xml" available: read stored parameter values
    else
    {
        // read ROI settings from file
        cv::FileNode pictureSections = set["picture_sections"];
        cv::FileNodeIterator sectionIt = pictureSections.begin(), sectionItEnd = pictureSections.end();

        std::vector <int> vSectionInfo;
        std::vector <float> vGridInfo;

        // ROIs
        for ( ; (sectionIt != sectionItEnd); sectionIt++)
        {
            cv::FileNode childNode = *sectionIt;
            cv::FileNodeIterator childIt = childNode.begin(), childItEnd = childNode.end();

            for (; (childIt != childItEnd); childIt++)
            {
                int nTemp;
                *childIt >> nTemp;
                vSectionInfo.push_back(nTemp);
            }

        }

        // read calibration pattern information
        cv::FileNode gridSettings = set ["circles_grid_settings"];
        cv::FileNodeIterator gridIt = gridSettings.begin(), gridItEnd = gridSettings.end();

        for (; (gridIt != gridItEnd); gridIt++)
        {
            cv::FileNode childNode = *gridIt;
            cv::FileNodeIterator childIt = childNode.begin(), childItEnd = childNode.end();

            for (; (childIt != childItEnd); childIt++)
            {
                float nTemp;
                *childIt >> nTemp;
                vGridInfo.push_back(nTemp);
            }
        }

        // store parameter values in variables
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

    std::cout << "nLeftPicLeftUpperCorner[0]:" << nLeftPicLeftUpperCorner[0] << std::endl;

    // read calibration parameter values from file "calibration.xml"
    cv::FileStorage fs;

    // stop program execution if file "calibration.xml" not available
    if(!fs.open(sProjectPath + "/" +"calibration.xml", cv::FileStorage::READ))
    {
        std::cout << "No file \"calibration.xml\" with calibration parameters found.\n"
                     "Hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        return 0;
    }

    // read parameter values
    fs["imageSize"] >> pImageSize;
    fs["CameraMatrixLeft"] >> pCameraMatrixLeft;
    fs["DistortionCoefficientsLeft"] >> pDistortionCoeffsLeft;
    fs["CameraMatrixRight"] >> pCameraMatrixRight;
    fs["DistortionCoefficientsRight"] >> pDistortionCoeffsRight;
    fs["RotationMatrix"] >> pLeftToRightRotationMatrix;
    fs["TranslationVector"] >> pLeftToRightTranslationVector;
    fs["ReprojectionError"] >> dRepErrorComplete;

    // fs["FrameRescalingFactor"] >> dRescaleFactor;

    fs.release();

    std::cout << "Calibration parameters successfully read. \n"
                 "Hit key to start rectification procedure..." << std::endl;

    // wait for user input
    cv::waitKey(0);

    // declare structures for rectification parameters and look-up maps
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
                  CALIB_ZERO_DISPARITY,             // CALIB_ZERO_DISPARITY flag for disparity 0 at infinity (to simulate parallel stereo setup)
                  -1,                               // flag "alpha" for pixel range to be considered, was -1 originally
                  pImageSize,                       // image size after rectification
                  &validROIL,                       // valid ROI left
                  &validROIR);                      // valid ROI right

    // std::cout << "Q matrix: " << pQ << std::endl;

    // pre-computation of look-up maps for fast undistortion and rectification of virtual cameras

    // pre-compute look-up map for left virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixLeft, pDistortionCoeffsLeft, pR1Left,
                            pP1Left, pImageSize, CV_16SC2, pMapLeft1, pMapLeft2);

    // pre-compute look-up map for right virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixRight, pDistortionCoeffsRight, pR2Right,
                            pP2Right, pImageSize, CV_16SC2, pMapRight1, pMapRight2);

    // image border handling
    const Scalar borderValueRemap = (0);

    // identify droplet trajectory

    std::cout << "Identifying droplet trajectory..." << std::endl;

    // initialize file storage object
    FileStorage trajInfos;

    // parameter initialization

    // blob detection parameters

    cv::SimpleBlobDetector::Params params;

    params.filterByArea = true;

    int minArea, maxArea, minThreshold, maxThreshold, thresholdStep, minDistBetweenBlobs;

    // rod lens system parameters
    if(bRodLens)
    {
        minArea = 25;
        maxArea = 800;
        minThreshold = 10;
        maxThreshold = 220;
        thresholdStep = 1;
        minDistBetweenBlobs = 115;

        params.minArea = 25;
        params.maxArea = 1000;
        params.minThreshold = 5;
        params.maxThreshold = 255;
        params.thresholdStep = 1;
        params.minDistBetweenBlobs = 100;
    }
    // fiberoptic system parameters
    else
    {
        minArea = 3;
        maxArea = 60;
        minThreshold = 10;
        maxThreshold = 220;
        thresholdStep = 1;
        minDistBetweenBlobs = 10;

        params.minArea = 3;
        params.maxArea = 60;
        params.minThreshold = 10;
        params.maxThreshold = 220;
        params.thresholdStep = 1;
        params.minDistBetweenBlobs = 10;
    }



    // query parameter
    int nCalcTraj;

    // check if file "TrajInfos.yml" exists

    // if file not found: identify droplet trajectory from frame sequence
    if(!trajInfos.open(sProjectPath + "/" +"TrajInfos.yml", FileStorage::READ))
    {
        std::cout << "No file \"TrajInfos.yml\" with droplet trajectory information found.\n"
                     "Identifying droplet trajectory..." << std::endl;
        nCalcTraj = 1;
    }

    // if file found: query if new trajectory identification desired
    else
    {
        nCalcTraj = -1;

        while(true)
        {
            // query if new trajectory identification desired
            std::cout << "File \"TrajInfos.yml\" with droplet trajectory information found. \n";

            std::cout << "Proceed with trajectory identification? (1 = Yes, 0 = No) \n";

            std::cin >> nCalcTraj;

            if(nCalcTraj == 1 || nCalcTraj == 0)
            {
                break;
            }
        }

    }

    // if droplet trajectory identification desired
    if(nCalcTraj == 1)
    {
        // load frame sequence "Os7-S1 Camera.mp4"
        std::cout << "Loading frame sequence \"Os7-S1 Camera.mp4\"..." << std::endl;
        VideoCapture trajVidObj (sProjectPath + "/" + "Os7-S1 Camera.mp4");

        // end program execution if no frame sequence found
        if(!trajVidObj.isOpened())
        {
            std::cout << "No frame sequence showing droplet flight could be found.\n"
                         "Please add file \"Os7-S1 Camera.mp4\" to project path and re-run program.\n"
                         "Hit key to end program execution... \n";

            // wait for user input
            cv::waitKey(0);

            // return error
            return -1;
        }

        // declarations

        // vectors for single frame storage (left and right)
        std::vector<Mat> vTrajVidFrames, vTrajLeft, vTrajLeftRect, vTrajRight, vTrajRightRect, vImagesWithBlobs, vCompleteTrajRectImages;

        // declaration for current frame
        cv::Mat pCurrentFrame;

        // declaration for OPTIONALLY rescaled frame
        cv::Mat pCurrentFrameRescaled;

        // read frames from "Os7-S1 Camera.mp4" and store in "vTrajVidFrames" vector (COLOR)
        while(trajVidObj.read(pCurrentFrame))
        {

            if(bRescaleFrames && dRescaleFactor != 1.0)
            {
                if(dRescaleFactor < 1.0)
                {
                    cv::resize(pCurrentFrame, pCurrentFrameRescaled, cv::Size(), dRescaleFactor, dRescaleFactor, INTER_AREA);

                    vTrajVidFrames.push_back(pCurrentFrameRescaled.clone());
                }
                else if(dRescaleFactor > 1.0)
                {
                    cv::resize(pCurrentFrame, pCurrentFrameRescaled, cv::Size(), dRescaleFactor, dRescaleFactor, INTER_LINEAR);

                    vTrajVidFrames.push_back(pCurrentFrameRescaled.clone());
                }
            }
            else
            {
                vTrajVidFrames.push_back(pCurrentFrame.clone());
            }
        }

        // store last frame in "vTrajVidFrames" vector
        std::vector<Mat> vLastFrame;
        vLastFrame.push_back(vTrajVidFrames[vTrajVidFrames.size()-1]);

        // show frame, if desired
        // saveOrShowImage(vLastFrame, "InputFrames", "LastFrame", false);

        // show frame number 20, if desired
        cv::namedWindow("Frame no. 20", WINDOW_AUTOSIZE);
        cv::imshow("Frame no. 20", vTrajVidFrames[19]);

        std::cout << "Showing frame no. 20. \n" << std::endl;
        std::cout << "Hit key to proceed... \n";

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("Frame no. 20");

        trajVidObj.release();

        vTrajLeft.resize(vTrajVidFrames.size());
        vTrajRight.resize(vTrajVidFrames.size());
        vTrajLeftRect.resize(vTrajVidFrames.size());
        vTrajRightRect.resize(vTrajVidFrames.size());
        vImagesWithBlobs.resize(vTrajVidFrames.size());
        vCompleteTrajRectImages.resize(vTrajVidFrames.size());

        // separate left and right images, convert to grayscale and rectify
        for (unsigned int i = 0; i < vTrajVidFrames.size(); i++)
        {
            // in-place grayscale conversion (GRAYSCALE)
            cv::cvtColor(vTrajVidFrames[i], vTrajVidFrames[i], CV_BGR2GRAY);

            // extract ROIs
            // TO DO: resolve problem with resized images in function splitImages!
            splitImages(vTrajVidFrames[i], vTrajLeft[i], vTrajRight[i]);

            // rectify left image
            cv::remap(vTrajLeft[i], vTrajLeftRect[i], pMapLeft1, pMapLeft2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

            // rectify right image
            cv::remap(vTrajRight[i], vTrajRightRect[i], pMapRight1, pMapRight2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

            // concatenate rectified images horizontally
            cv::hconcat(vTrajLeftRect[i], vTrajRightRect[i], vCompleteTrajRectImages[i]);
        }

        // declarations
        cv::Mat rectImgWithLines;
        cv::Size pCompleteSize = vCompleteTrajRectImages[0].size();

        rectImgWithLines = vCompleteTrajRectImages[0];

        // draw horizontal lines (should be parallel to epipolar lines)

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*1/8), Point(pCompleteSize.width, pCompleteSize.height*1/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*2/8), Point(pCompleteSize.width, pCompleteSize.height*2/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*3/8), Point(pCompleteSize.width, pCompleteSize.height*3/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*4/8), Point(pCompleteSize.width, pCompleteSize.height*4/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*5/8), Point(pCompleteSize.width, pCompleteSize.height*5/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*6/8), Point(pCompleteSize.width, pCompleteSize.height*6/8), Scalar(255,0,0), .5, 8);

        cv::line(rectImgWithLines, Point(0, pCompleteSize.height*7/8), Point(pCompleteSize.width, pCompleteSize.height*7/8), Scalar(255,0,0), .5, 8);

        // show frame with (horizontal) epipolar lines
        cv::namedWindow("Rectification Result", WINDOW_AUTOSIZE);
        cv::imshow("Rectification Result", rectImgWithLines);

        std::cout << "Showing rectified image with horizontal lines." << std::endl;
        std::cout << "Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("Rectification Result");

        //        // contrast enhancement (according to alpha*bild + beta; alpha -> contrast factor; beta -> intensity addition)
        //        float fAlpha = 1.5;
        //        int nBeta = 0;

        //        for (unsigned int i = 0; i < vTrajVidFrames.size(); i++)
        //        {

        //            vTrajLeftRect[i].convertTo(vTrajLeftRect[i], -1, fAlpha, nBeta);
        //            vTrajRightRect[i].convertTo(vTrajRightRect[i], -1, fAlpha, nBeta);
        //        }


        // find first frame with moving droplet

        // consecutive frames are subtracted to find motion in frame sequence

        // declarations
        std::vector<cv::Mat> vForeground;
        std::vector<int> vImagesWithMovingDroplet;

        if(bRodLens)
        {
            nMinWhitePixels = 10; // was 20
            nFoundWhites = 0;
        }
        else
        {
            nMinWhitePixels = 4; // was 4
            nFoundWhites = 0;
        }

        vForeground.resize(vTrajLeft.size());

        // find first and last frame of droplet motion
        // analyze left sub-images only

        std::cout << "Finding valid frames for droplet trajectory identification..." << std::endl;

        cv::imshow("Left sub-frame no. 20", vTrajLeft[19]);

        cv::waitKey(0);

        cv::destroyWindow("Left sub-frame no. 20");

        for(unsigned int i = 1; i < vTrajLeft.size();i++)
        {
            // calculate difference image
            cv::absdiff(vTrajLeft[i-1],vTrajLeft[i], vForeground[i]);

            // binarize difference image (approximately containing droplet contour only)

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
                         getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            // counting of white pixels in frame
            nFoundWhites = cv::countNonZero(vForeground[i]);

            // threshold operation: more than nMinWhitePixels white pixels in frame -> droplet still in midair
            if (nFoundWhites > nMinWhitePixels)
            {
                vImagesWithMovingDroplet.push_back(i);
            }
        }

        // return number of valid frames
        std::cout << vImagesWithMovingDroplet.size() << " frames with moving droplet found!" << std::endl;

        std::cout << "Hit key to proceed...\n";

        // wait for user input
        cv::waitKey(0);

        // store first and last frame of droplet motion

        // identify indices of first and last valid frame
        int nFirstFrame = vImagesWithMovingDroplet.front();
        int nLastFrame = vImagesWithMovingDroplet.back();

        // identify frame index of first valid droplet triangulation
        int nFirstValidFrame = 0;

        // declarations for blob detection

        cv::Mat pBlobDetectorTempFrameLeft, pBlobDetectorTempFrameRight, pBlobDetectorTestLeft, pBlobDetectorTestRight, pBlobDrawTestLeft, pBlobDrawTestRight;

        std::vector<Mat> vMovingDroplet;

        int nBlobCounter = 0;

        // go through all valid frames and store in "vMovingDroplet" vector
        for (int i = nFirstFrame; i < nLastFrame+1; i++)
        {
            vMovingDroplet.push_back(vTrajVidFrames[i]);
        }

        // save/show frames with droplet in midair
        saveOrShowImage(vMovingDroplet, "Frames_With_Moving_Droplet", "movingdroplet", false);

        // copy one valid frame to visualize Blob detector parameter value influence

        pBlobDetectorTempFrameLeft = vTrajLeftRect[nFirstFrame+5].clone();
        pBlobDetectorTempFrameRight = vTrajRightRect[nFirstFrame+1].clone();

        // show frame
        cv::namedWindow("Blob detector input frame before rectification", WINDOW_AUTOSIZE);
        cv::imshow("Blob detector input frame before rectification", vTrajLeft[nLastFrame-5]);

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("Blob detector input frame before rectification");

        // convert frames to 8 bit grayscale image

        pBlobDetectorTempFrameLeft.convertTo(pBlobDetectorTempFrameLeft, CV_8UC1);
        pBlobDetectorTempFrameRight.convertTo(pBlobDetectorTempFrameRight, CV_8UC1);

        // show frames

        cv::namedWindow("Left Blob detector input frame after rectification (CV_8UC1)", WINDOW_AUTOSIZE);
        cv::imshow("Left Blob detector input frame after rectification (CV_8UC1)",  pBlobDetectorTempFrameLeft);

        std::cout << "Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        if(bRodLens)
        {
            cv::destroyWindow("Blob detector input frame after rectification (CV_8UC1)");
        }
        else
        {
            cv::destroyWindow("Left Blob detector input frame after rectification (CV_8UC1)");
        }

        // command line output
        std::cout << "Configuration of Blob detector parameters.\n"
                     "Set parameters so that only the droplet is detected, not its (lighter) shadow!" << std::endl;

        // evaluate current blob detector parameter values

        cv::Mat TempInvImgLeft, TempInvImgRight;

        while(true)
        {
            // create difference image
            pBlobDrawTestLeft = pBlobDetectorTempFrameLeft.clone();
            pBlobDrawTestRight = pBlobDetectorTempFrameRight.clone();

            cv::absdiff(pBlobDetectorTempFrameLeft, vTrajLeftRect[0], pBlobDetectorTestLeft);
            cv::absdiff(pBlobDetectorTempFrameRight, vTrajRightRect[0], pBlobDetectorTestRight);

            // morphological closing of difference images

            cv::namedWindow("Before morphol. closing", cv::WINDOW_AUTOSIZE);

            cv::imshow("Before morphol. closing", pBlobDetectorTestLeft);

            cv::waitKey(0);

            cv::destroyWindow("Before morphol. closing");

            cv::morphologyEx(pBlobDetectorTestLeft, pBlobDetectorTestLeft, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(10,10), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            cv::morphologyEx(pBlobDetectorTestRight, pBlobDetectorTestRight, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(10,10), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            cv::namedWindow("After morphol. closing", cv::WINDOW_AUTOSIZE);

            cv::imshow("After morphol. closing", pBlobDetectorTestLeft);

            cv::waitKey(0);

            cv::destroyWindow("After morphol. closing");

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

            //            // contrast enhancement (CLAHE)
            //            Ptr<CLAHE> clahe = cv::createCLAHE();
            //            clahe->setClipLimit(4);

            //            // apply CLAHE
            //            clahe->apply(pBlobDetectorTestLeft,pBlobDetectorTestLeft);
            //            clahe->apply(pBlobDetectorTestRight,pBlobDetectorTestRight);

            // invert images for blob detection

            TempInvImgLeft = cv::Scalar::all(255) - pBlobDetectorTestLeft;
            TempInvImgRight = cv::Scalar::all(255) - pBlobDetectorTestRight;

            // pBlobDetectorTestLeft = TempInvImgLeft.clone();
            // pBlobDetectorTestRight = TempInvImgRight.clone();

            // detect blobs
            detector->detect(TempInvImgLeft, keypointsLeft);
            detector->detect(TempInvImgRight, keypointsRight);

            // draw circumference of found blobs (left sub-image)
            cv::drawKeypoints(pBlobDrawTestLeft, keypointsLeft, pBlobDrawTestLeft, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            // draw centroid of found blobs (left sub-image)
            if(keypointsLeft.size() != 0)
            {
                cv::line(pBlobDrawTestLeft, Point(keypointsLeft[0].pt.x-5.0, keypointsLeft[0].pt.y), Point(keypointsLeft[0].pt.x+5.0, keypointsLeft[0].pt.y), Scalar(255,255,0), .5, 8);
                cv::line(pBlobDrawTestLeft, Point(keypointsLeft[0].pt.x, keypointsLeft[0].pt.y-5.0), Point(keypointsLeft[0].pt.x, keypointsLeft[0].pt.y+5.0), Scalar(255,255,0), .5, 8);
            }

            // draw circumference of found blobs (right sub-image)
            cv::drawKeypoints(pBlobDrawTestRight, keypointsRight, pBlobDrawTestRight, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            // draw centroid of found blobs (right sub-image)
            if(keypointsRight.size() != 0)
            {
                cv::line(pBlobDrawTestRight, Point(keypointsRight[0].pt.x-5.0, keypointsRight[0].pt.y), Point(keypointsRight[0].pt.x+5.0, keypointsRight[0].pt.y), Scalar(255,255,0), .5, 8);
                cv::line(pBlobDrawTestRight, Point(keypointsRight[0].pt.x, keypointsRight[0].pt.y-5.0), Point(keypointsRight[0].pt.x, keypointsRight[0].pt.y+5.0), Scalar(255,255,0), .5, 8);
            }

            // parameter configuration of blob detector
            cv::namedWindow("BlobDectParams");
            cv::createTrackbar("minArea", "BlobDectParams", &minArea, 100);
            cv::createTrackbar("maxArea", "BlobDectParams", &maxArea, 1500);
            cv::createTrackbar("minThreshold", "BlobDectParams", &minThreshold, 255);
            cv::createTrackbar("maxThreshold", "BlobDectParams", &maxThreshold, 255);
            cv::createTrackbar("thresholdStep", "BlobDectParams", &thresholdStep, 255);
            cv::createTrackbar("minDistBetweenBlobs", "BlobDectParams", &minDistBetweenBlobs, 255);

            std::cout << "Blob detector iteration no. " << nBlobCounter << " finished." << std::endl;
            nBlobCounter++;

            cv::namedWindow("Blob detector configuration (left image)");
            cv::imshow("Blob detector configuration (left image)", pBlobDrawTestLeft);
            cv::namedWindow("Blob detector configuration (right image)");
            cv::imshow("Blob detector configuration (right image)", pBlobDrawTestRight);

            // wait for user input
            nKey = cv::waitKey();

            // ESC key closes windows
            if (nKey == 27)
            {
                cv::destroyAllWindows();
                break;
            }
        }

        // blob detector parametrization now complete

        // apply adapted blob detector parameters on all valid frames

        // declarations
        std::vector<Mat> vValidFrames; // vector contains image data of valid frames

        std::vector<Vec2f> vFoundDropletCentersLeft2D, vFoundDropletCentersRight2D; // vectors contain found circles

        std::vector<KeyPoint> keypointsFoundLeft, keypointsFoundRight;  // vectors contain found KeyPoints

        cv::Mat pOnlyDropletLeft, pOnlyDropletRight; // Mat structures contain frames after background subtraction

        std::vector <int> vValidFramePositions; // vector contains frame indices of valid frames

        std::vector<float> vValidTimeStamps; // vector contains time stamps of valid frames (always starting at t=0)

        // resize vector to have 0 elements initially
        vValidTimeStamps.resize(0);

        int nFrameRate; // integer contains frame rate of droplet flight sequence in fps

        std::cout << "Droplet centroid positions being detected..." << std::endl;

        std::cout << "Please enter frame rate of droplet flight sequence in fps and hit enter key!" << std::endl;

        std::cin >> nFrameRate; // get user input from the keyboard

        // constructor for blob detector with final parameters
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // temporary vectors
        cv::Vec2f tempBlobLeft, tempBlobRight;

        // "+1" only for sequences with low frame rate
        // ("-10" to "-20" instead of  "+1" for high frame rates)
        for (int i = nFirstFrame; i < nLastFrame-25; i++)
        {
            // grayscale conversion for blob detector compatibility
            vTrajLeftRect[i].convertTo(vTrajLeftRect[i], CV_8UC1);
            vTrajRightRect[i].convertTo(vTrajRightRect[i], CV_8UC1);

            // background subtraction
            cv::absdiff(vTrajLeftRect[i], vTrajLeftRect[0], pOnlyDropletLeft);
            cv::absdiff(vTrajRightRect[i], vTrajRightRect[0], pOnlyDropletRight);

            // morphological closings
            cv::morphologyEx(pOnlyDropletLeft, pOnlyDropletLeft, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(10,10), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            cv::morphologyEx(pOnlyDropletRight, pOnlyDropletRight, MORPH_CLOSE,
                         getStructuringElement(MORPH_RECT, Size(10,10), Point(-1,-1)), Point(-1,-1),
                         1, BORDER_CONSTANT, 0);

            // inversion for blob detection
            TempInvImgLeft = cv::Scalar::all(255) - pOnlyDropletLeft;

            // pOnlyDropletLeft = TempInvImgLeft.clone();

            TempInvImgRight = cv::Scalar::all(255) - pOnlyDropletRight;

            // pOnlyDropletRight = TempInvImgRight.clone();

            // blob detection (left image)
            detector->detect(TempInvImgLeft, keypointsFoundLeft);

            // blob detection (right image)
            detector->detect(TempInvImgRight, keypointsFoundRight);

            // filtering of faulty detections

            // exclude frame if not exactly one blob found
            if(keypointsFoundLeft.size() != 1)
            {
                std::cout << "Error: " << keypointsFoundLeft.size() << " blobs found in left image!" << std::endl;
                continue;
            }

            // exclude frame if not exactly one blob found
            if(keypointsFoundRight.size() != 1)
            {
                std::cout << "Error: " << keypointsFoundRight.size() << " blobs found in right image!" << std::endl;
                continue;
            }


            // as images are rectified, vertical coordinates of droplet centroid must be approximately identical (tolerance of +/- 5 pixels found empirically to yield satisfying results)
            if (((keypointsFoundLeft[0].pt.y-5) > keypointsFoundRight[0].pt.y) || ((keypointsFoundLeft[0].pt.y+5) < keypointsFoundRight[0].pt.y))
            {
                std::cout << "Error: found droplet centroids not horizontally aligned!" << std::endl;
                continue;
            }

            // disparities must be positive
            if ((keypointsFoundLeft[0].pt.x - keypointsFoundRight[0].pt.x) <= 0)
            {
                std::cout << "Error: negative disparity!" << std::endl;
                continue;
            }

            // draw found blobs

            // draw circumference of found blobs (left sub-image)
            cv::drawKeypoints(vTrajLeftRect[i], keypointsFoundLeft, vTrajLeftRect[i], cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            // draw centroid of found blobs (left sub-image)
            cv::line(vTrajLeftRect[i], Point(keypointsFoundLeft[0].pt.x-5.0, keypointsFoundLeft[0].pt.y), Point(keypointsFoundLeft[0].pt.x+5.0, keypointsFoundLeft[0].pt.y), Scalar(255,255,0), .5, 8);
            cv::line(vTrajLeftRect[i], Point(keypointsFoundLeft[0].pt.x, keypointsFoundLeft[0].pt.y-5.0), Point(keypointsFoundLeft[0].pt.x, keypointsFoundLeft[0].pt.y+5.0), Scalar(255,255,0), .5, 8);

            // draw circumference of found blobs (right sub-image)
            cv::drawKeypoints(vTrajRightRect[i], keypointsFoundRight, vTrajRightRect[i], cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            // draw centroid of found blobs (right sub-image)
            cv::line(vTrajRightRect[i], Point(keypointsFoundRight[0].pt.x-5.0, keypointsFoundRight[0].pt.y), Point(keypointsFoundRight[0].pt.x+5.0, keypointsFoundRight[0].pt.y), Scalar(255,255,0), .5, 8);
            cv::line(vTrajRightRect[i], Point(keypointsFoundRight[0].pt.x, keypointsFoundRight[0].pt.y-5.0), Point(keypointsFoundRight[0].pt.x, keypointsFoundRight[0].pt.y+5.0), Scalar(255,255,0), .5, 8);

            // concatenate images
            cv::hconcat(vTrajLeftRect[i], vTrajRightRect[i], vImagesWithBlobs[i]);

            // add current frame index to vector of valid frame indices
            vValidFramePositions.push_back(i);
            vValidFrames.push_back(vImagesWithBlobs[i]);

            // add time stamp of current frame in ms to vector of valid time stamps

            // first valid frame must have time stamp t_1 = 0 for correct trajectory averaging based on several trajectories at identical system settings
            if(vValidTimeStamps.size()==0)
            {
                vValidTimeStamps.push_back(0.0);

                nFirstValidFrame = i;
            }
            // if consecutive valid frame found
            else
            {
                vValidTimeStamps.push_back(((1000.0*((float)i - (float)nFirstValidFrame))/(float)nFrameRate));
            }

            // save found droplet centroids in vector

            tempBlobLeft[0] = keypointsFoundLeft[0].pt.x;
            tempBlobLeft[1] = keypointsFoundLeft[0].pt.y;

            tempBlobRight[0] = keypointsFoundRight[0].pt.x;
            tempBlobRight[1] = keypointsFoundRight[0].pt.y;

            vFoundDropletCentersLeft2D.push_back(tempBlobLeft);
            vFoundDropletCentersRight2D.push_back(tempBlobRight);
        }

        // output valid time stamps

        std::cout << "Valid time stamps:" << std::endl;

        for (unsigned int i = 0; i < vValidTimeStamps.size(); i++)
        {
            std::cout << vValidTimeStamps[i] << std::endl;
        }

        // save frames with detected blobs
        saveOrShowImage(vValidFrames, "Frames_With_Blobs", "blob", false);

        // stop program execution if less than 2 valid frames found
        if (vFoundDropletCentersLeft2D.size() < 2)
        {
            std::cout << "Error: less than 2 valid frames found." << std::endl;
            std::cout << "Droplet trajectory approximation not possible. Hit key to end program execution..." << std::endl;

            // wait for user input
            cv::waitKey(0);

            return 0;
        }

        // warning if less than five droplet centroid pairs found
        if (vFoundDropletCentersLeft2D.size() < 5)
        {
            std::cout << "Warning: only " << vFoundDropletCentersLeft2D.size() << "valid frames found.\n"
                                                                                 "Droplet trajectory approximation may be imprecise. Hit key to proceed..." << std::endl;

            // wait for user input
            cv::waitKey(0);
        }

        // triangulate found droplet centroid positions in space
        std::cout << "Calculating droplet centroid disparities..." << std::endl;

        // declaration of vector for (x,y,z) coordinates of droplet centroid positions in coordinate system of left virtual camera
        std::vector<cv::Vec3f> vTrajPoints;
        vTrajPoints.resize(vFoundDropletCentersLeft2D.size());

        // if use of function triangulatePoints() for spatial droplet triangulation desired (not required)
        if(bTriangulatePoints)
        {
            // declaration of vectors containing sensor coordinates of droplet centroid positions
            std::vector<cv::Point2f> vTrajPointsLeft, vTrajPointsRight;
            vTrajPointsLeft.resize(vFoundDropletCentersLeft2D.size());
            vTrajPointsRight.resize(vFoundDropletCentersLeft2D.size());

            cv::Mat vTrajPointsHomogeneous(4, vFoundDropletCentersLeft2D.size(), CV_64F);

            // for all found droplet positions on trajectory
            for (unsigned int i  = 0; i < vFoundDropletCentersLeft2D.size(); i++)
            {
                // read horizontal sensor coordinate (left image)
                vTrajPointsLeft[i].x = vFoundDropletCentersLeft2D[i][0];
                // read vertical sensor coordinate (left image)
                vTrajPointsLeft[i].y = vFoundDropletCentersLeft2D[i][1];

                // read horizontal sensor coordinate (right image)
                vTrajPointsRight[i].x = vFoundDropletCentersRight2D[i][0];
                // read vertical sensor coordinate (right image)
                vTrajPointsRight[i].y = vFoundDropletCentersRight2D[i][1];
            }

            // transform sensor coordinates of droplet centroid to 3D point in coordinate system of left virtual camera
            cv::triangulatePoints(pP1Left, pP2Right, vTrajPointsLeft, vTrajPointsRight, vTrajPointsHomogeneous);

            std::cout << "vTrajPointsHomogeneous.at<float>(0,1): " << vTrajPointsHomogeneous.at<float>(0,1) << std::endl;
            std::cout << "vTrajPointsHomogeneous.at<float>(1,1): " << vTrajPointsHomogeneous.at<float>(1,1) << std::endl;
            std::cout << "vTrajPointsHomogeneous.at<float>(2,1): " << vTrajPointsHomogeneous.at<float>(2,1) << std::endl;
            std::cout << "vTrajPointsHomogeneous.at<float>(3,1): " << vTrajPointsHomogeneous.at<float>(3,1) << std::endl;

            // transform homogeneous coordinates (X, Y, Z, W) into Cartesian coordinates (X/W, Y/W, Z/W)

            for (unsigned int i  = 0; i < vFoundDropletCentersLeft2D.size(); i++)
            {
                vTrajPoints[i][0] = vTrajPointsHomogeneous.at<float>(0,i)/vTrajPointsHomogeneous.at<float>(3,i);
                vTrajPoints[i][1] = vTrajPointsHomogeneous.at<float>(1,i)/vTrajPointsHomogeneous.at<float>(3,i);
                vTrajPoints[i][2] = vTrajPointsHomogeneous.at<float>(2,i)/vTrajPointsHomogeneous.at<float>(3,i);
            }

            std::cout << "vTrajPoints[1][0]: " << vTrajPoints[1][0] << std::endl;
            std::cout << "vTrajPoints[1][1]: " << vTrajPoints[1][1] << std::endl;
            std::cout << "vTrajPoints[1][2]: " << vTrajPoints[1][2] << std::endl;
        }
        // if use of function perspectiveTransform() desired for spatial droplet triangulation (STANDARD)
        else
        {
            // declaration of vector containing (u,v,d) values of droplet centroid positions
            std::vector<cv::Vec3f> vTrajDisparity;
            vTrajDisparity.resize(vFoundDropletCentersLeft2D.size());

            // calculate disparity values of found droplet centroid positions
            for (unsigned int i  = 0; i < vFoundDropletCentersLeft2D.size(); i++)
            {
                // read u coordinate
                vTrajDisparity[i][0] = vFoundDropletCentersLeft2D[i][0];

                // read v coordinate
                vTrajDisparity[i][1] = vFoundDropletCentersLeft2D[i][1];

                // calculate disparity d
                vTrajDisparity[i][2] = vFoundDropletCentersLeft2D[i][0] - vFoundDropletCentersRight2D[i][0];
            }

            // transform sensor coordinates and corresponding disparity value to 3D point in coordinate system of left virtual camera
            cv::perspectiveTransform(vTrajDisparity, vTrajPoints, pQ);

            std::cout << "Image coordinates and disparities (u,v,d) of all found droplet centroid positions: " << std::endl;

            for(unsigned int i=0; i < vTrajDisparity.size(); i++)
            {
                std::cout << vTrajDisparity[i] << std::endl;
            }
        }

        std::cout << "Coordinates (x,y,z) of all spatially triangulated droplet positions in mm: " << std::endl;

        for(unsigned int i=0; i < vTrajPoints.size(); i++)
        {
            std::cout << vTrajPoints[i] << std::endl;
        }


        //        // droplet velocity profile estimation
        //        std::vector <double> vDistancesForVel;
        //        std::vector <double> vDistancePerFrame;

        //        vDistancesForVel.resize(vTrajPoints.size());
        //        vDistancePerFrame.resize(vTrajPoints.size());


        //        std::cout << "Calculating droplet velocity profile..." << std::endl;
        //        for (unsigned int i = 0; i < vTrajPoints.size()-1; i++)
        //        {
        //            vDistancesForVel[i] = norm(Mat (vTrajPoints[i]) - Mat (vTrajPoints[i+1]), NORM_L2);

        //            vDistancePerFrame[i] = vDistancesForVel[i] / (double)(vValidFramePositions[i+1] - vValidFramePositions[i]);
        //        }

        //        // save velocities
        //        std::ofstream velocityFile;
        //        velocityFile.open("VelocityPerFrame.txt", std::ios_base::binary);

        //        for(std::size_t i = 0; i < vDistancePerFrame.size(); i++)
        //        {
        //                velocityFile << vDistancePerFrame[i] << "," << vValidFramePositions[i] <<"\n";
        //        }

        //        velocityFile.close();


        // identify fit functions for droplet trajectory
        std::cout << "Hit key to start droplet trajectory approximation procedure... \n";

        // wait for user input
        cv::waitKey();

        // number of available sampling points
        int nSamplingPoints = (int)vTrajPoints.size();

        std::cout << "nSamplingPoints: " << nSamplingPoints << std::endl;

        // construct (4 x nSamplingPoints) matrix of sampling point parameters (x,y,z,t)
        std::vector<cv::Vec4f> vSamplingPoints(nSamplingPoints);

        for(int i=0; i < nSamplingPoints; i++)
        {
            vSamplingPoints[i][0] = vTrajPoints[i][0];
            vSamplingPoints[i][1] = vTrajPoints[i][1];
            vSamplingPoints[i][2] = vTrajPoints[i][2];
            vSamplingPoints[i][3] = vValidTimeStamps[i];    // time stamps of sampling points are contained in vector "vValidTimeStamps"
        }

        std::cout << "Matrix of sampling point parameters (x,y,z,t) generated." << std::endl;

        // find linear fit function for droplet trajectory (DEPRECATED)
        std::cout << "Linear droplet trajectory approximation in progress..." << std::endl;

        // non-shifted sampling points must be stored as 3D points
        std::vector<cv::Point3f> vSamplingPointsLin;

        // temporary helper point
        cv::Point3f TempSamplingPoint = cv::Point3f(0.0,0.0,0.0);

        for (int i=0; i<nSamplingPoints; i++)
        {
            TempSamplingPoint.x = vSamplingPoints[i][0];
            TempSamplingPoint.y = vSamplingPoints[i][1];
            TempSamplingPoint.z = vSamplingPoints[i][2];

            vSamplingPointsLin.push_back(TempSamplingPoint);
        }

        std::cout << "Vector of sampling points for linear trajectory approximation: " << vSamplingPointsLin << std::endl;

        // parameter set of linear trajectory approximation
        cv::Vec6f T_lin;

        cv::Point3f suppVecLin = Point3f(0.0,0.0,0.0);
        cv::Point3f dirVecLin = Point3f (0.0,0.0,0.0);

        // find fit line parameters using target metric "CV_DIST_L2"
        cv::fitLine(vSamplingPointsLin, T_lin, CV_DIST_L2, 0, 0.01, 0.01);

        // direction vector of fit line
        dirVecLin.x = T_lin[0];
        dirVecLin.y = T_lin[1];
        dirVecLin.z = T_lin[2];

        // support vector of fit line
        suppVecLin.x =  T_lin[3];
        suppVecLin.y =  T_lin[4];
        suppVecLin.z =  T_lin[5];

        // store defining parameters of linear approximation in YML document
        std::string filename = "TrajInfos.yml";

        trajInfos.open(filename, FileStorage::WRITE);

        // get current system time
        time_t rawtime;
        time(&rawtime);

        // save current system time
        trajInfos << "Date" << asctime(localtime(&rawtime));
        trajInfos << "dirVecLin" << dirVecLin;
        trajInfos << "suppVecLin" << suppVecLin;

        std::cout << "Support and direction vector of linear approximation calculated and stored. " << std::endl;

        // find best-fit plane PI through sampling points using SVD-based method

        // calculate centroid C of sampling points
        cv::Point3f C;

        C.x = (float)cv::mean(vSamplingPoints)[0];  // mean of all x coordinates
        C.y = (float)cv::mean(vSamplingPoints)[1];  // mean of all y coordinates
        C.z = (float)cv::mean(vSamplingPoints)[2];  // mean of all z coordinates

        std::cout << "Centroid calculated." << std::endl;
        std::cout << "C.x: " << C.x << std::endl;
        std::cout << "C.y: " << C.y << std::endl;
        std::cout << "C.z: " << C.z << std::endl;

        // shift sampling points to origin of coordinate frame (CF)_C of left virtual camera

        // (3 x nSamplingPoints) vector of shifted sampling points (x,y,z)
        std::vector<cv::Vec3f> vSamplingPointsShifted(nSamplingPoints);

        // for each sampling point
        for (int i=0 ; i<nSamplingPoints ; i++)
        {
            // for each set of coordinates (x,y,z)
            vSamplingPointsShifted[i][0] = vSamplingPoints[i][0] - C.x;
            vSamplingPointsShifted[i][1] = vSamplingPoints[i][1] - C.y;
            vSamplingPointsShifted[i][2] = vSamplingPoints[i][2] - C.z;
        }

        std::cout << "Sampling points shifted (now average-free)." << std::endl;

        // calculate new centroid of shifted sampling points
        cv::Point3f C_new;

        C_new.x = (float)cv::mean(vSamplingPointsShifted)[0];  // mean of all x coordinates
        C_new.y = (float)cv::mean(vSamplingPointsShifted)[1];  // mean of all y coordinates
        C_new.z = (float)cv::mean(vSamplingPointsShifted)[2];  // mean of all z coordinates

        std::cout << "Shifted centroid C_new (should be zero): " << " " << C_new.x << " " << C_new.y << " " << C_new.z << std::endl;

        // perform singular value decomposition (SVD) of matrix of shifted sampling point coordinates

        // SVD declarations
        cv::Mat w, u, vt;

        // declaration for sampling point refactoring structure
        cv::Mat M = cv::Mat::zeros(3, nSamplingPoints, CV_32F);

        // for all rows of M (iterate over sampling point coordinates x,y,z)
        for (int i=0; i<3 ;i++)
        {
            // for all columns of M (iterate over all sampling points)
            for (int j=0; j<nSamplingPoints; j++)
            {
                M.at<float>(i,j) = vSamplingPointsShifted[j][i];
            }
        }

        // perform SVD (w: calculated singular values, u: calculated left singular vectors, vt: transposed matrix of right singular values)
        cv::SVD::compute(M, w, u, vt);

        std::cout << "Calculated left singular vectors: " << u << std::endl;
        std::cout << "Last column of u: " << u.col(2) << std::endl;

        // left singular vector to smallest singular value is normal vector n_PI of plane PI (=right column of u)
        cv::Vec3f n_PI;

        n_PI = u.col(2);

        std::cout << "n_PI: " << n_PI << std::endl;
        std::cout << "cv::norm(n_PI): " << cv::norm(n_PI) << std::endl;

        // calculate Hesse normal form of fit plane PI

        float dotP = C.dot(n_PI);

        std::cout << "Dot product C * n_PI: " << dotP << std::endl;

        cv::Vec3f n_PI_0;

        // calculate new unit normal vector n_PI_0
        if (dotP < 0)
        {
            n_PI_0 = - n_PI/cv::norm(n_PI);
        }
        else if (dotP > 0)
        {
            n_PI_0 = n_PI/cv::norm(n_PI);
        }

        std::cout << "n_PI_0: " << n_PI_0 << std::endl;
        std::cout << "cv::norm(n_PI_0): " << cv::norm(n_PI_0) << std::endl;

        // calculate d value of Hesse normal form
        double d;
        d = C.dot(n_PI_0);

        std::cout << "d = " << d << std::endl;

        // save fit plane parameters
        trajInfos << "n_PI_0" << n_PI_0;
        trajInfos << "d" << d;

        std::cout << "Best-fit plane in normal form calculated and stored." << std::endl;

        // find parabolic trajectory approximation (STANDARD MODEL)

        // construct (3*nSamplingPoints x 9) matrix T of sampling time points
        cv::Mat T = cv::Mat::zeros(3*nSamplingPoints, 9, CV_32F);

        // (vValidTimeStamps contains time stamps in ms)

        // fill matrix T with correct parameters
        for (int i=0; i<nSamplingPoints; ++i)
        {
            T.at<float>(i,0) = vValidTimeStamps[i]*vValidTimeStamps[i];
            T.at<float>(i,1) = vValidTimeStamps[i];
            T.at<float>(i,2) = 1.0;
        }
        for (int i=nSamplingPoints;i<2*nSamplingPoints; ++i)
        {
            T.at<float>(i,3) = vValidTimeStamps[i-nSamplingPoints]*vValidTimeStamps[i-nSamplingPoints] ;
            T.at<float>(i,4) = vValidTimeStamps[i-nSamplingPoints];
            T.at<float>(i,5) = 1.0;
        }
        for (int i=2*nSamplingPoints; i<3*nSamplingPoints; ++i)
        {
            T.at<float>(i,6) = vValidTimeStamps[i-2*nSamplingPoints]*vValidTimeStamps[i-2*nSamplingPoints];
            T.at<float>(i,7) = vValidTimeStamps[i-2*nSamplingPoints];
            T.at<float>(i,8) = 1.0;
        }

        std::cout << "Time stamp matrix T: " << T << std::endl;

        // construct (9x1) column coordinate vector F of trajectory-defining vectors
        cv::Mat F = cv::Mat::zeros(9, 1, CV_32F);

        // construct (3*nSamplingPoints x 1) column vector P of sampling point coordinates (x,y,z)
        cv::Mat P = cv::Mat::zeros(3*nSamplingPoints, 1, CV_32F);

        for (int i=0; i<nSamplingPoints; ++i)
        {
            P.at<float>(i,0) = vSamplingPoints[i][0];
        }
        for (int i=nSamplingPoints; i<2*nSamplingPoints; ++i)
        {
            P.at<float>(i,0) = vSamplingPoints[i-nSamplingPoints][1];
        }
        for (int i=2*nSamplingPoints; i<3*nSamplingPoints; ++i)
        {
            P.at<float>(i,0) = vSamplingPoints[i-2*nSamplingPoints][2];
        }

        std::cout << "Vector P of sampling point coordinates (x_i,y_i,z_i): " << P << std::endl;

        // "solve" over-determined system T*F = P using SVD method
        cv::solve(T, P, F, DECOMP_SVD);

        std::cout << "Identified parameter vector F: " << F << std::endl;

        // store defining parameters of parabolical approximation (components f_i_x, f_i_y, f_i_z of the three vectors) in YML document
        trajInfos << "F" << F;

        trajInfos.release();

        // variables for distances between trajectory approximations and fit plane PI and spatial sampling points
        double avgDistLin, avgDistPara, avgDistPI;

        // calculate and print average geometric distance from sampling points to linear fit function in mm (DEPRECATED MODEL)
        avgDistLin = calculateAverageGeometricDistanceLin(suppVecLin, dirVecLin, vSamplingPoints);

        // calculate and print average geometric distance from sampling points to parabolic fit function in mm (STANDARD MODEL)
        avgDistPara = calculateAverageGeometricDistancePara(F, vSamplingPoints);

        // calculate and print average orthogonal distances from sampling points to fit plane PI in mm
        avgDistPI = calculateAverageGeometricDistancePI(n_PI_0, d, vSamplingPoints);

        // save triangulated droplet centroid positions and trajectories as point clouds for visualization

        // declare vector for triangulated droplet centroid positions on trajectory and linear and parabolic trajectories
        std::vector<cv::Mat> vTrajectory, vTrajectoryPoints, vPlanes;

        vTrajectory.resize(2);
        vTrajectoryPoints.resize(1);
        vPlanes.resize(1);

        // store found droplet centroid positions
        vTrajectoryPoints[0] = cv::Mat(1,vTrajPoints.size(),CV_64FC3);

        // "draw" found droplet centroids
        for (unsigned int k = 0; k < vTrajPoints.size(); k++)
        {
            vTrajectoryPoints[0].at<Vec3d>(0,k) = vTrajPoints[k];
        }

        std::vector<cv::Vec3i> vTrajColors(2);

        vTrajColors[0][0] = 255;
        vTrajColors[0][1] = 0;
        vTrajColors[0][2] = 0;

        vTrajColors[1][0] = 0;
        vTrajColors[1][1] = 0;
        vTrajColors[1][2] = 255;

        std::vector<cv::Vec3i> vPointColor(1);

        vPointColor[0][0] = 0;
        vPointColor[0][1] = 0;
        vPointColor[0][2] = 255;

        std::vector<cv::Vec3i> vPlaneColors(1);

        vPlaneColors[0][0] = 255;
        vPlaneColors[0][1] = 255;
        vPlaneColors[0][2] = 0;

        // save triangulated droplet positions
        savePointCloud("Droplet Centroid Positions", "triangulatedCentroids", vPointColor, vTrajectoryPoints);

        // "draw" linear trajectory approximation in space (DEPRECATED)

        // declarations
        double fZMin = -10.0;   // minimum z coordinate value
        double fZMax = 200.0;   // maximum z coordinate value
        double fFactor = 0.0;   // factor for point density along trajectory
        int nPosition = 0;      // frame column iterator

        // calibration performed in mm
        int nCounter = -10000;

        cv::Vec3d tempPoint;

        // initialize z coordinate
        tempPoint[2] = -20.0;

        vTrajectory[0] = cv::Mat::zeros(5000, 5000, CV_64FC3); // for linear trajectory

//        // inject average support and direction vectors to save average trajectory point cloud

//        suppVecLin.x = 1.9592044830322266e+01;
//        suppVecLin.y = -8.0904874801635742e+00;
//        suppVecLin.z = 7.1137718200683594e+01;

//        dirVecLin.x = 1.2977286241948605e-02;
//        dirVecLin.y = -1.0824790596961975e-01;
//        dirVecLin.z = 9.9403917789459229e-01;

//        // for all frame pixel columns
//        while(nPosition < vTrajectory[0].cols)
//        {

//            fFactor = .01 * (double)nCounter;

//            tempPoint = cv::Vec3d(suppVecLin.x+fFactor*dirVecLin.x,
//                              suppVecLin.y+fFactor*dirVecLin.y,
//                              suppVecLin.z+fFactor*dirVecLin.z);

//            nPosition++;
//            nCounter++;

//            if ((tempPoint[2] > fZMax) || (tempPoint[2] < fZMin))
//            {
//               // tempPoint = cv::Vec3d(suppVecLin.x, suppVecLin.y, suppVecLin.z);
//               continue;
//            }

//            vTrajectory[0].at<cv::Vec3d>(0,nPosition) = tempPoint;

//        }

        while((tempPoint[2] < fZMax))
        {
            fFactor = .01 * (double)nCounter;

            tempPoint = cv::Vec3d(suppVecLin.x+fFactor*dirVecLin.x,
                              suppVecLin.y+fFactor*dirVecLin.y,
                              suppVecLin.z+fFactor*dirVecLin.z);

            nCounter++;

            if ((tempPoint[2] < fZMin))
            {
               continue;
            }

            vTrajectory[0].at<cv::Vec3d>(0,nPosition) = tempPoint;

            nPosition++;
        }


        // "draw" parabolic trajectory approximation in space

        // reset parameters
        fZMin = 0.0;
        fFactor = 0.0;  // factor for point density along trajectory
        nPosition = 0;  // frame column iterator

        // initialize z coordinate of point on parabola
        tempPoint[2] = -20.0;

        // inject average trajectory vector components to save average trajectory point cloud

        //        F.at<float>(0,0) = -3.43937683e-03;
        //        F.at<float>(1,0) = -1.86496973e-01;
        //        F.at<float>(2,0) = 1.53170671e+01;
        //        F.at<float>(3,0) = -7.16989336e-04;
        //        F.at<float>(4,0) = -2.32360393e-01;
        //        F.at<float>(5,0) = -7.57515430e+00;
        //        F.at<float>(6,0) = 2.36416631e-03;
        //        F.at<float>(7,0) = 1.08415759e+00;
        //        F.at<float>(8,0) = 3.33103065e+01;

        vTrajectory[1] = cv::Mat::zeros(5000, 5000, CV_64FC3); // for parabolic trajectory

        // start time point
        fFactor = 0.0;

        while((tempPoint[2] < fZMax))
        {
            // if more than 200 ms elapsed on trajectory
            if(fFactor > 200.0)
            {
                break;
            }

            tempPoint = cv::Vec3d(F.at<float>(2,0)+fFactor*F.at<float>(1,0)+fFactor*fFactor*F.at<float>(0,0),
                              F.at<float>(5,0)+fFactor*F.at<float>(4,0)+fFactor*fFactor*F.at<float>(3,0),
                              F.at<float>(8,0)+fFactor*F.at<float>(7,0)+fFactor*fFactor*F.at<float>(6,0));

            fFactor += 0.01;                             // increment time point

            if ((tempPoint[2] < fZMin))
            {
                // tempPoint = cv::Vec3d(0.0,0.0,0.0);
                continue;
            }

            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

            nPosition++;
        }

        // save linear and parabolic trajectory approximations
        savePointCloud("Trajectories", "trajectories", vTrajColors, vTrajectory);


        // "draw" fit plane PI

        // inject average plane parameters to save average trajectory fit plane as point cloud

        // d = 6.21654033660889e+00;

        // n_PI_0[0] = 2.14835733175278e-01;
        // n_PI_0[1] = 9.50740933418274e-01;
        // n_PI_0[2] = 2.23466515541077e-01;

        // reset parameters
        nPosition = 0;  // iterator

        vPlanes[0] = cv::Mat::zeros(5000, 5000, CV_64FC3);

        double dX_PI = 0.0;

        // go through z coordinate values
        for (double z = 0.0 ; z < 200.0 ; z=z+0.8)
        {
            // go through y coordinate values
            for (double y = -100.0 ; y < 100.0 ; y=y+0.8)
            {
                dX_PI = (double)(d-(double)y*n_PI_0[1]-(double)z*n_PI_0[2])/((double)n_PI_0[0]);

                if (dX_PI < -100.0 || dX_PI > 100.0)
                {
                    continue;
                }

                tempPoint = cv::Vec3d(dX_PI, (double)y, (double)z);

                vPlanes[0].at<cv::Vec3d>(0,nPosition) = tempPoint;

                nPosition++;
            }
        }


        // save fit plane PI
        savePointCloud("Fit Plane", "planePI", vPlaneColors, vPlanes);

        fs.release();
        }


    // end program execution

    std::cout << "Droplet trajectory identification complete. Hit key to end program execution..." << std::endl;

    // wait for user input
    cv::waitKey(0);

    return 0;
}
