// Creators: Adrian K. Rüppel/Jacob F. Fast
// 2017-2021
//
// Calibration_Procedure.cpp
//
// Read calibration images or calibration frame sequence showing asymmetrical dot pattern and save XML file with (stereo) calibration results.

#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

// OpenCV
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
#include <ctime>
#include <cmath>

using namespace cv;
using namespace cv::ximgproc;

// global variables

// project path
std::string sProjectPath;

// variables to overwrite already present images
bool bAlreadyAsked = false;
int nOverride = -1;

// flag for frame sequence processing
bool bFrameSequence = true;

// flag for image rescaling
bool bRescaleFrames = false;

// variable for image rescaling factor
double dRescaleFactor = 1.0;

// define ROIs (will be read from XML file)
int nLeftPicLeftUpperCorner[2] = {0, 0};    // in px
int nRightPicLeftUpperCorner[2] = {0, 0};   // in px

// define image size (will be read from XML file)
int nImageSize[2] = {0, 0}; // in px

// calibration pattern information (will be read from XML file)
float fDistanceBetweenCircles = 0.0;  // here: in mm
Size circlesGridSize;

// FUNCTIONS

// read all calibration images from folder
bool loadImages(std::string sFolderName, std::vector<std::string> &vNames, std::vector<Mat> &vLoadedImages, bool bGray, double dScaleFactor)
{
    // path to folder
    std::string sDirPath = sProjectPath + "/" + sFolderName;
    const char* cDirPath = sDirPath.c_str();

    // further variable declarations
    bool bError = false;
    std::string sFullPath;

    // open folder "sFolderName"
    DIR* pDir = opendir(cDirPath);

    // find files in folder
    struct dirent* pFilesInDir;

    // Mat structure for current image
    cv::Mat pCurrentImage;

    // Mat structure for rescaled current image
    cv::Mat pCurrentImageRescaled;

    // check if folder exists
    if (pDir == NULL)
    {
        std::cout << "Folder \"" << cDirPath << "\" not found. \n"
                     "Folder being created. \n"
                     "Please add images to folder \"" << cDirPath << "\" "
                     "and re-run program." << std::endl;

        mkdir(cDirPath, S_IRWXU | S_IRWXG | S_IRWXO);

        bError = true;

        // wait for user input
        cv::waitKey(0);

        return bError;
    }

    // read images
    while((pFilesInDir = readdir(pDir)) != NULL)
    {
        // this check is required to not count folders "." and ".." which are always present
        if (!(strcmp(pFilesInDir->d_name, ".")) || !(strcmp(pFilesInDir->d_name, "..")))
        {
            continue;
        }

        // read images

        // complete path to current image
        sFullPath = sDirPath + "/" + std::string(pFilesInDir->d_name);

        // load current image
        if (bGray)
        {
            pCurrentImage = cv::imread(sFullPath, IMREAD_GRAYSCALE);
        }
        else
        {
            pCurrentImage = cv::imread(sFullPath, IMREAD_COLOR);
        }

        // skip file if no image could be read
        if(pCurrentImage.empty())
        {
            continue;
        }

        if(dScaleFactor < 1.0)
        {
            cv::resize(pCurrentImage, pCurrentImageRescaled, cv::Size(), dScaleFactor, dScaleFactor, INTER_AREA);
            // save image
            vLoadedImages.push_back(pCurrentImageRescaled);
        }
        else if(dScaleFactor > 1.0)
        {
            cv::resize(pCurrentImage, pCurrentImageRescaled, cv::Size(), dScaleFactor, dScaleFactor, INTER_LINEAR);
            // save image
            vLoadedImages.push_back(pCurrentImageRescaled);
        }
        else
        {
            vLoadedImages.push_back(pCurrentImage);
        }

        // safe file name
        vNames.push_back(pFilesInDir->d_name);
    }

    closedir(pDir);
    pCurrentImage.release();
    pCurrentImageRescaled.release();
    sDirPath.clear();
    sFullPath.clear();

    return bError;
}

// retrieve left and right image from sensor of high-speed camera
void splitImages(cv::Mat &pImage, cv::Mat &pLeftImage, cv::Mat &pRightImage)
{
    // define ROIs
    // pixel coordinates determined empirically (depending on optical setup)
    Rect leftRectangle = Rect(nLeftPicLeftUpperCorner[0], nLeftPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);
    Rect rightRectangle = Rect(nRightPicLeftUpperCorner[0], nRightPicLeftUpperCorner[1], nImageSize[0], nImageSize[1]);

    // retrieve left image
    pLeftImage = pImage(leftRectangle);

    // retrieve right image
    pRightImage = pImage(rightRectangle);
}

// save/show image
void saveOrShowImage(std::vector<cv::Mat> &vImages,std::string sFolderName, std::string sFileName, bool bShow)
{
    // folder path
    std::string sDirPath = sProjectPath + "/" + sFolderName;

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

// MAIN PROGRAM

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // read path of this program from argv
    std::string pTempPath[256] = {argv[0]};
    std::string sExecPath;
    sExecPath = *pTempPath;

    // find last slash in path to extract folder path of this program
    int iPosOfChar = sExecPath.find_last_of("/");

    // save folder path of this program
    sProjectPath = sExecPath.substr(0, iPosOfChar);

    std::cout  <<  sProjectPath  <<  std::endl ;

    // calibration of stereo laryngoscope system

    // error handling (image file processing)
    bool bCalibLoadingError = false;
    int nDoCalibration = -1;

    // flag for laryngoscopic system
    bool bRodLens = true;

    // parameters of virtual cameras

    // declare Mat structures for intrinsic parameters and distortion coefficients
    cv::Mat pCameraMatrixLeft, pCameraMatrixRight, pDistortionCoeffsLeft, pDistortionCoeffsRight;

    // pCameraMatrixLeft = cv::Mat::eye(3,3, CV_64F);
    // pCameraMatrixRight = cv::Mat::eye(3,3, CV_64F);

    // declare variables for reprojection errors (in px)
    double dRepErrorComplete = 9999.0;
    double dRepErrorLeft = 9999.0;
    double dRepErrorRight = 9999.0;

    // declare object for ROI size (identical for left and right image)
    cv::Size pImageSize;

    // declarations for calibration output storage

    // rotation matrix and translation vector from left to right virtual camera
    cv::Mat pLeftToRightRotationMatrix, pLeftToRightTranslationVector;

    // read file "settings.xml" with pre-defined parameters of raw images

    std::cout << "Loading pre-defined settings..." << std::endl;

    std::cout << "Project path:" << sProjectPath << std::endl;

    // FileStorage object for XML file handling
    cv::FileStorage set;

    // check if "settings.xml" file exists
    if(!set.open(sProjectPath + "/" +"settings.xml", FileStorage::READ))
    {
        std::cout << "No file \"settings.xml\" found.\n"
                     "File is being created.\n"
                     "Please fill in required information (coordinates in pixels) and re-run program." << std::endl;

        // configure XML file
        FileStorage set ("settings.xml", FileStorage::WRITE);

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
    else
    {
        // load pre-defined settings from file
        FileNode pictureSections = set["picture_sections"];
        FileNodeIterator sectionIt = pictureSections.begin(), sectionItEnd = pictureSections.end();

        // declare vectors for pre-defined settings
        std::vector <int> vSectionInfo;
        std::vector <float> vGridInfo;

        // read coordinates of left and right image
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

        // read calibration pattern properties
        FileNode gridSettings = set["circles_grid_settings"];
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

        // save data
        nLeftPicLeftUpperCorner[0] = vSectionInfo[0];
        nLeftPicLeftUpperCorner[1] = vSectionInfo[1];

        nRightPicLeftUpperCorner[0] = vSectionInfo[2];
        nRightPicLeftUpperCorner[1] = vSectionInfo[3];

        nImageSize[0] = vSectionInfo[4];
        nImageSize[1] = vSectionInfo[5];

        fDistanceBetweenCircles = vGridInfo[0];
        circlesGridSize = Size(((int)vGridInfo[1]), ((int) vGridInfo[2]));
    }

    std::cout << "Calibration pattern dimensions (without unit): " << circlesGridSize << std::endl;
    std::cout << "Hit key to proceed..." << std::endl;

    // wait for user confirmation
    cv::waitKey(0);

    set.release();

    // FileStorage object for XML file handling
    FileStorage fs;

    // check if XML file "calibration.xml" with parameters of virtual cameras already exists
    if(!fs.open(sProjectPath + "/" +"calibration.xml", FileStorage::READ))
    {
        std::cout << "No camera calibration file \"calibration.xml\" found.\n"
                     "Calibration must be performed. Hit key to proceed." << std::endl;

        // wait for user confirmation
        cv::waitKey(0);

        nDoCalibration = 1;

        fs.release();
    }
    else
    {
        nDoCalibration = -1;

        // wait for valid user input
        while(true)
        {

            // query user input for camera calibration
            std::cout << "File \"calibration.xml\" with camera parameters already exists.\n"
                         "Continue calibration procedure? (1 = Yes, 0 = No)" << std::endl;

            // user input handling
            std::cin >> nDoCalibration;

            if (nDoCalibration == 1)
            {
                fs.release();
            }

            if (nDoCalibration == 0)
            {
                fs.release();

                return 0;
            }

            if(nDoCalibration == 1 || nDoCalibration == 0)
            {
                break;
            }
        }
    }

    // perform stereo calibration
    if (nDoCalibration)
    {
        // read calibration images

        // declare vector for read images
        std::vector<cv::Mat> vCompleteCalibImages;

        // declare vector for image file names
        std::vector<std::string> vCompleteCalibFileNames;

        // initialize folder name for file path of calibration images
        std::string sCalibDirPath = "CalibrationImages";

        // read images
        std::cout << "Reading calibration images..." << std::endl;

        // read frames from frame sequence
        if(bFrameSequence)
        {
            cv::VideoCapture calVidObj(sProjectPath + "/" + "CalibrationSequence" + "/" + "Os7-S1 Camera.mp4");

            // end program if no frame sequence found
            if(!calVidObj.isOpened())
            {
                std::cout << "No calibration frame sequence could be found! \n"
                             "Please add frame sequence to folder \"CalibrationSequence\" in project directory and re-run program. \n"
                             "Hit key to end program execution..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                // stop program execution
                return 0;
            }

            // declaration for current frame
            cv::Mat pCurrentFrame;
            // declaration for (OPTIONALLY) resized frame
            cv::Mat pResizedFrame;

            // (number of frames to be read from sequence) -1
            int nFrameNumber = 100;

            // number of frames that were read
            int nFrameCounter = 0;

            // store single frames from frame sequence in vector
            while(calVidObj.read(pCurrentFrame))
            {
                if(nFrameCounter > nFrameNumber)
                {
                    break;
                }

                if(bRescaleFrames)
                {
                    // isometrically rescale frame by desired scale factor

                    if(dRescaleFactor < 1.0)
                    {
                        cv::resize(pCurrentFrame, pResizedFrame, cv::Size(), dRescaleFactor, dRescaleFactor, INTER_AREA);
                    }
                    else
                    {
                        cv::resize(pCurrentFrame, pResizedFrame, cv::Size(), dRescaleFactor, dRescaleFactor, INTER_LINEAR);
                    }

                    vCompleteCalibImages.push_back(pResizedFrame.clone());
                }
                else
                {
                    vCompleteCalibImages.push_back(pCurrentFrame.clone());
                }

                nFrameCounter++;
            }

            calVidObj.release();
        }
        // else: read single frames from folder
        else
        {
            bCalibLoadingError = loadImages(sCalibDirPath,vCompleteCalibFileNames,vCompleteCalibImages, false, dRescaleFactor);

            // error handling

            if (bCalibLoadingError)
            {
                std::cout << "Error during reading of calibration images. Hit key to stop program execution." << std::endl;

                // wait for user confirmation
                cv::waitKey(0);

                return 0;
            }
        }

        std::cout << "vCompleteCalibImages.size(): " << vCompleteCalibImages.size() << std::endl;

        // check if >= 2 images could be found (at least 2 images required for calibration)
        if (vCompleteCalibImages.size() < 2)
        {
            std::cout << "Less than 2 images for calibration found.\n"
                         "Calibration not possible. Hit key to stop program execution..." << std::endl;

            // wait for user confirmation
            cv::waitKey(0);

           return 0;
        }

        // check if < 10 images could be found (low calibration accuracy expected)
        if(vCompleteCalibImages.size() < 10)
        {
            std::cout << "Warning! Less than 10 images found for calibration." << std::endl;
            std::cout << "At least 20 images should be available for sufficient calibration accuracy." << std::endl;
            std::cout << "Hit key to proceed..." << std::endl;

            // wait for user confirmation
            cv::waitKey(0);
        }

        // extract ROI from images
        std::cout << "Extracting ROIs..." << std::endl;

        // declare vectors of Mat structures for left, right and complete image storage
        std::vector<cv::Mat> vLeftCalibImages, vLeftCalibImagesGray, vRightCalibImages, vRightCalibImagesGray, vCompletePatternImages;

        vLeftCalibImages.resize(vCompleteCalibImages.size());
        vRightCalibImages.resize(vCompleteCalibImages.size());

        vLeftCalibImagesGray.resize(vCompleteCalibImages.size());
        vRightCalibImagesGray.resize(vCompleteCalibImages.size());

        vCompletePatternImages.resize(vCompleteCalibImages.size());

        // go through images and extract left and right ROIs
        for (unsigned int i = 0; i < vCompleteCalibImages.size(); i++)
        {
            // TO DO: resolve problem with resized images in function splitImages!
            splitImages(vCompleteCalibImages[i], vLeftCalibImages[i], vRightCalibImages[i]);
        }

        // identify image size
        pImageSize = vLeftCalibImages[0].size();

        cv::namedWindow("Left ROI", WINDOW_AUTOSIZE);
        cv::imshow("Left ROI", vLeftCalibImages[0]);

        cv::namedWindow("Right ROI", WINDOW_AUTOSIZE);
        cv::imshow("Right ROI", vRightCalibImages[0]);

        std::cout << "Hit key to proceed..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        cv::destroyWindow("Left ROI");
        cv::destroyWindow("Right ROI");

        // computation of stereo laryngoscope parameters

        // declare vectors for image and object point coordinates
        std::vector<std::vector<Point2f>> vImagePointsLeft, vImagePointsRight;
        std::vector<std::vector<Point3f>> vObjectPoints(1);

        // detect calibration pattern in all available images
        std::cout << "Calibration pattern detection..." << std::endl;

        // declare vectors for found circle centers
        std::vector<Point2f> vCentersLeft, vCentersRight;

        // flags for state of pattern detection
        bool bFoundPatternLeft = false;
        bool bFoundPatternRight = false;

        // instantiate blob detector parameters object
        cv::SimpleBlobDetector::Params params;

        // params.filterByArea = true;
        // params.minArea = 8;     // was 8
        // params.maxArea = 60;    // 20 may be used here

        // change thresholds
        params.minThreshold = 3;
        params.maxThreshold = 255;
        params.thresholdStep = 1;
        params.minDistBetweenBlobs = 5;

        // constructor for blob detector with adapted parameters
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // // evaluate blob detection
        // std::vector<KeyPoint> keypoints;
        // cv::Mat im_with_keypoints;

        // CLAHE histogram equalization
        Ptr<CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2);

        // flag for CLAHE application
        bool bCLAHE = false;

        // ask if CLAHE should be applied
        std::cout << "Use CLAHE histogram equalization? (1 = Yes, 0 = No)" << std::endl;

        std::cin >> bCLAHE;

        // go through all images
        for (unsigned int i = 0 ; i < vLeftCalibImages.size() ; i++)
        {

            if(i==4)
            {
                cv::imshow("Calibration image no. 5 before grayscale conversion",vLeftCalibImages[i]);

                std::cout << "Hit key to proceed..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                cv::destroyWindow("Calibration image no. 5 before grayscale conversion");
            }

            // convert color images to grayscale
            if (vLeftCalibImages[i].channels() == 3)
            {
                cv::cvtColor(vLeftCalibImages[i],vLeftCalibImagesGray[i],CV_BGR2GRAY);
            }
            else
            {
                vLeftCalibImagesGray[i] = vLeftCalibImages[i];
            }

            // convert color images to grayscale
            if (vRightCalibImages[i].channels() == 3)
            {
                cv::cvtColor(vRightCalibImages[i],vRightCalibImagesGray[i],CV_BGR2GRAY);
            }
            else
            {
                vRightCalibImagesGray[i] = vRightCalibImages[i];
            }

            if(i==4)
            {
                cv::imshow("Calibration image no. 5 after grayscale conversion",vLeftCalibImagesGray[i]);

                std::cout << "Hit key to proceed..." << std::endl;

                // wait for user input
                cv::waitKey(0);

                cv::destroyWindow("Calibration image no. 5 after grayscale conversion");
            }

//            // test: histogram equalization
//            cv::equalizeHist(vLeftCalibImagesGray[i],vLeftCalibImagesGray[i]);
//            cv::equalizeHist(vRightCalibImagesGray[i],vRightCalibImagesGray[i]);

//            // apply Gaussian filtering on image data from fiberoptic laryngoscope
//            if(!bRodLens)
//            {
//                // set sigma in both image directions
//                cv::GaussianBlur(vLeftCalibImagesGray[i],vLeftCalibImagesGray[i],Size(-1,-1),0.8,0.8);
//                cv::GaussianBlur(vRightCalibImagesGray[i],vRightCalibImagesGray[i],Size(-1,-1),0.8,0.8);

//                if(i==4)
//                {
//                    cv::imshow("Calibration image no. 5 after Gaussian filtering",vLeftCalibImagesGray[i]);

//                    std::cout << "Hit key to proceed..." << std::endl;

//                    // wait for user input
//                    cv::waitKey(0);

//                    cv::destroyWindow("Calibration image no. 5 after Gaussian filtering");
//                }
//            }


            // apply CLAHE method
            if(bCLAHE)
            {
                clahe->apply(vLeftCalibImagesGray[i],vLeftCalibImagesGray[i]);
                clahe->apply(vRightCalibImagesGray[i],vRightCalibImagesGray[i]);

                // show sample image after CLAHE application
                if(i==4)
                {
                    cv::imshow("Calibration image no. 5 after CLAHE",vLeftCalibImagesGray[i]);

                    std::cout << "Hit key to proceed..." << std::endl;

                    // wait for user input
                    cv::waitKey(0);

                    cv::destroyWindow("Calibration image no. 5 after CLAHE");
                }
            }

//            // apply morphological closing on image data from fiberoptic laryngoscope (remove white centers from calibration circles)
//            if(!bRodLens)
//            {
//                // apply morphological closing operator on left and right image

//                cv::morphologyEx(vLeftCalibImagesGray[i], vLeftCalibImagesGray[i], MORPH_CLOSE,
//                             cv::getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(-1,-1)), Point(-1,-1),
//                             1, BORDER_CONSTANT, 0);

//                cv::morphologyEx(vRightCalibImagesGray[i], vRightCalibImagesGray[i], MORPH_CLOSE,
//                             cv::getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(-1,-1)), Point(-1,-1),
//                             1, BORDER_CONSTANT, 0);

//                if(i==4)
//                {
//                    cv::imshow("Calibration image no. 5 after morphological closing",vLeftCalibImagesGray[i]);

//                    std::cout << "Hit key to proceed..." << std::endl;

//                    // wait for user input
//                    cv::waitKey(0);

//                    cv::destroyWindow("Calibration image no. 5 after morphological closing");
//                }
//            }


//            // test: blob detection
//            if(i==4)
//            {
//            detector->detect(vLeftCalibImagesGray[i], keypoints);
//            cv::drawKeypoints(vLeftCalibImagesGray[i], keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//            // show detected blobs
//            cv::imshow("keypoints", im_with_keypoints);
//            cv::waitKey(0);
//            }

            // find calibration pattern in images
            if(bRodLens)
            {
                // find pattern in left image (with standard blob detector suitable for rod lens stereolaryngoscope)
                bFoundPatternLeft = cv::findCirclesGrid(vLeftCalibImagesGray[i], circlesGridSize, vCentersLeft, cv::CALIB_CB_ASYMMETRIC_GRID);
            }
            else
            {
                // find pattern in left image (blob detector adapted to fiberoptic stereolaryngoscope)
                // flag combination (cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING) may also be useful to increase circle detection performance
                bFoundPatternLeft = cv::findCirclesGrid(vLeftCalibImagesGray[i], circlesGridSize, vCentersLeft, cv::CALIB_CB_ASYMMETRIC_GRID, detector);
            }

            // warning: left pattern not found
            if (!bFoundPatternLeft)
            {
                if(bFrameSequence)
                {
                    std::cout << "No pattern detected in left sub-image of frame no. " << i << "!" << std::endl;
                }
                else
                {
                std::cout << "No pattern detected in left sub-image of image \"" << vCompleteCalibFileNames[i] << "\"!" << std::endl;
                }
            }
            else
            {
                cv::drawChessboardCorners(vLeftCalibImages[i], circlesGridSize, vCentersLeft, bFoundPatternLeft);
            }

            // search calibration pattern in right image if pattern was found in left image
            if (bFoundPatternLeft)
            {
                // adapt blob detector parameters according to optical system
                if(bRodLens)
                {
                    // find pattern in right image (with standard blob detector suitable for rod lens stereo laryngoscope)
                    bFoundPatternRight = cv::findCirclesGrid(vRightCalibImagesGray[i], circlesGridSize, vCentersRight, cv::CALIB_CB_ASYMMETRIC_GRID);
                }
                else
                {
                    // find pattern in right image (blob detector adapted to fiberoptic stereo laryngoscope)
                    // flag combination (cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING) may also be useful to increase circle detection performance
                    bFoundPatternRight = cv::findCirclesGrid(vRightCalibImagesGray[i], circlesGridSize, vCentersRight, cv::CALIB_CB_ASYMMETRIC_GRID,detector);
                }

                // warning: right pattern not found
                if (!bFoundPatternRight)
                {
                    if(bFrameSequence)
                    {
                        std::cout << "No pattern detected in right sub-image of frame no. " << i << "!" << std::endl;
                    }
                    else
                    {
                    std::cout << "No pattern detected in right sub-image of image \"" << vCompleteCalibFileNames[i] << "\"!" << std::endl;
                    }
                }
                // proceed, if pattern found in both sub-images
                else
                {
                    cv::drawChessboardCorners(vRightCalibImages[i], circlesGridSize, vCentersRight, bFoundPatternRight);

                    // concatenate sub-images horizontally
                    cv::hconcat(vLeftCalibImages[i], vRightCalibImages[i], vCompletePatternImages[i]);

                    // add image points to vectors
                    vImagePointsLeft.push_back(vCentersLeft);
                    vImagePointsRight.push_back(vCentersRight);
                }
            }
        }

        // clear vectors
        vCentersLeft.clear();
        vCentersRight.clear();

        std::cout << "vCompletePatternImages.size():" << vCompletePatternImages.size() <<std::endl;

        // show and save calibration images with found patterns
        saveOrShowImage(vCompletePatternImages, "Images_with_pattern", "pattern", false);

        // count valid calibration images
        std::cout << "Calibration pattern found in " << vImagePointsRight.size() << " images. Hit key to proceed..." << std::endl;

        // wait for user confirmation
        cv::waitKey(0);

        if(vImagePointsRight.size()<2)
        {
            std::cout << "Less than 2 valid calibration images with detected pattern available.\n"
                         "Calibration not possible. Hit key to stop program execution..." << std::endl;

            // wait for user confirmation
            cv::waitKey(0);

            return 0;
        }

        if(vImagePointsRight.size()<10)
        {
            std::cout << "Less than 10 valid calibration images with detected pattern available.\n"
                         "Calibration may be inaccurate. Hit key to proceed..." << std::endl;

            // wait for user confirmation
            cv::waitKey(0);
        }

        // calculate position of circles in calibration pattern in pattern coordinate system in multiples of fDistanceBetweenCircles
        // algorithm according to OpenCV documentation (pattern turned 90 degress CCW, go through pattern column per column, starting on top left

        for (int k = 0; k < circlesGridSize.height; k++)
        {
            for (int j = 0; j < circlesGridSize.width; j++)
            {
                vObjectPoints[0].push_back(cv::Point3f((float)((2*j+k%2)*fDistanceBetweenCircles), (float)(k*fDistanceBetweenCircles), 0.0f));
            }
        }

        vObjectPoints.resize(vImagePointsLeft.size(), vObjectPoints[0]);

        // perform calibration
        std::cout << "Left camera calibration ongoing..." << std::endl;

        // perform calibration of left virtual camera
        // use rational (complex optical system with reflecting prisms/wide-angle view) and/or tilted (manual adjustment) model

        dRepErrorLeft = cv::calibrateCamera(vObjectPoints, vImagePointsLeft, pImageSize, pCameraMatrixLeft,
                                        pDistortionCoeffsLeft, cv::noArray(), cv::noArray(),
                                        CV_CALIB_TILTED_MODEL,
                                        TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 200, 1e-6));

        std::cout << "Reprojection error for left virtual camera: " << dRepErrorLeft << std::endl;

        // perform calibration
        std::cout << "Right camera calibration ongoing..." << std::endl;

        // perform calibration of right virtual camera
        // use rational (complex optical system with reflecting prisms/wide-angle view) and/or tilted (manual adjustment) model

        dRepErrorRight = cv::calibrateCamera(vObjectPoints, vImagePointsRight, pImageSize, pCameraMatrixRight,
                                        pDistortionCoeffsRight, cv::noArray(), cv::noArray(),
                                        CV_CALIB_TILTED_MODEL,
                                        TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 200, 1e-6));

        std::cout << "Reprojection error for right virtual camera: " << dRepErrorRight << std::endl;

        // stereo calibration

        // perform calibration
        std::cout << "Stereo calibration ongoing..." << std::endl;

        // declaration of essential matrix (transformation from left to right camera in camera coordinate system)
        cv::Mat pEssentialMatrix;

        // declaration of fundamental matrix (transformation from left to right camera in sensor coordinates, hence extends essential matrix by intrinsic camera parameters)
        cv::Mat pFundamentalMatrix;

        // perform stereo calibration
        // use rational (complex optical system with reflecting prisms/wide-angle view) and/or tilted (manual adjustment) model

        dRepErrorComplete = cv::stereoCalibrate(vObjectPoints,
                                    vImagePointsLeft,
                                    vImagePointsRight,
                                    pCameraMatrixLeft,
                                    pDistortionCoeffsLeft,
                                    pCameraMatrixRight,
                                    pDistortionCoeffsRight,
                                    pImageSize,
                                    pLeftToRightRotationMatrix,
                                    pLeftToRightTranslationVector,
                                    pEssentialMatrix,
                                    pFundamentalMatrix,
                                    CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_TILTED_MODEL,
                                    TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 200, 1e-6));

        if (dRepErrorComplete < 1)
        {
            std::cout << "Calibration of stereoscopic laryngoscope system successful." << std::endl;
            std::cout << "Global reprojection error in px: " << dRepErrorComplete << std::endl;
        }
        else
        {
            std::cout << "Global reprojection error in px: " << dRepErrorComplete
                      << " > First iteration of calibration procedure not sufficiently accurate." << std::endl;
        }

    }
    // if nDoCalibration = false:
    else
    {
        // read camera parameters from XML file, if it already exists
        fs["imageSize"] >> pImageSize;
        fs["CameraMatrixLeft"] >> pCameraMatrixLeft;
        fs["DistortionCoefficientsLeft"] >> pDistortionCoeffsLeft;
        fs["CameraMatrixRight"] >> pCameraMatrixRight;
        fs["DistortionCoefficientsRight"] >> pDistortionCoeffsRight;
        fs["RotationMatrix"] >> pLeftToRightRotationMatrix;
        fs["TranslationVector"] >> pLeftToRightTranslationVector;
        fs["ReprojectionError"] >> dRepErrorComplete;

        fs.release();

        std::cout << "Calibration data of stereoscopic laryngoscope system successfully read." << std::endl;

        if (dRepErrorComplete < 1)
        {
            std::cout << "Global reprojection error in px: " << dRepErrorComplete << std::endl;
        }
        else
        {
            std::cout << "Global reprojection error in px: " << dRepErrorComplete
                      << " > First iteration of calibration procedure not sufficiently accurate." << std::endl;
        }
    }

    // save camera parameters in XML file

    // FileStorage object for XML file handling
    FileStorage fs2 ("calibration.xml", FileStorage::WRITE);

    time_t rawtime;
    time(&rawtime);

    fs2 << "Date" << asctime(localtime(&rawtime))
        << "imageSize" << pImageSize
        << "CameraMatrixLeft" << pCameraMatrixLeft
        << "DistortionCoefficientsLeft" << pDistortionCoeffsLeft
        << "CameraMatrixRight" << pCameraMatrixRight
        << "DistortionCoefficientsRight" << pDistortionCoeffsRight
        << "RotationMatrix" << pLeftToRightRotationMatrix
        << "TranslationVector" << pLeftToRightTranslationVector
        << "ReprojectionError" << dRepErrorComplete
        << "FrameRescalingFactor" << dRescaleFactor;

    //       << "pR1Left" << pR1Left
    //       << "pR2Right" << pR2Right
    //       << "pP1Left" << pP1Left
    //       << "pP2Right" << pP2Right
    //       << "pQ" << pQ
    //       << "pMapLeft1" << pMapLeft1
    //       << "pMapLeft2" << pMapLeft2
    //       << "pMapRight1" << pMapRight1
    //       << "pMapRight2" << pMapRight2;

    fs2.release();

    // calibration complete
    std::cout << "Calibration complete. Hit key to end program execution..." << std::endl;

    // wait for user input
    cv::waitKey();

    return 0;
}
