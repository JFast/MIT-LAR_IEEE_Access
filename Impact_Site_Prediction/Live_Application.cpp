// Contributors: Adrian K. Rüppel/Robin Szymanski/Hardik R. Dava/Jacob F. Fast
//
// 2017-2021
//
// Live_Application.cpp
//
// Connect to high-speed camera (here: Os-v3-7-S1, IDT, Inc., Pasadena, USA).
// Read raw stereolaryngoscopic single frame and corresponding camera calibration and droplet trajectory parameters and compute droplet impact site prediction.
// Indicate predicted impact site and standard deviation of this prediction in the left rectified or undistorted laryngoscopic image, if desired by user.

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

// high-speed camera API
#include "XstrmAPI.h"

#define NanoSec 1000000000L

#define PI 3.14159265

using namespace cv;
using namespace cv::ximgproc;
using namespace cv::stereo;

// GLOBAL VARIABLES

// project path
std::string sProjectPath;

// flag for image overwriting
bool bAlreadyAsked = false;

// flag for stereo laryngoscope system
bool bRodLens = true;

// flag for binary SGBM (DO NOT CHANGE)
bool bBinarySGBM = true;

// initialize SGBM correspondence matcher

// negative nDisp12MaxDiff: difference check disabled
// uniqueness ratio must be non-negative
// num of disparities must be > 0 and divisible by 16
// full 8-direction algorithm used here (slower, but more accurate)
// alternatively use cv::stereo::StereoBinarySGBM::MODE_SGBM (5-direction mode)
Ptr<cv::stereo::StereoBinarySGBM> pLeftMatcherSGBMBinary = cv::stereo::StereoBinarySGBM::create(0, 112, 5, 200, 1000, 2, 20, 5, 150, 2, cv::stereo::StereoBinarySGBM::MODE_HH);

// initialize WLS filter

// normal WLS filter method does not work for CENSUS method. Set parameters manually.
// see: https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html (accessed on 03/02/2021)
// (calculate right disparity map to enable LR check)

// create WLS filter instance without left-right consistency check (thus flag false)
Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);

// flag for WLS filter (DO NOT CHANGE)
bool bWLS = true;

// variables for OPTIONAL frame rescaling
bool bRescaleFrames = true;
double dRescaleFactor = 0.5;

// flag for raw image display
bool bUseRawImage = true;

// variables for frame rate overlay
bool bShowFrameRate = false;
int nCompTimeMS = 0.0;
double dFrameRate = 0.0;
double dFrameRateRounded = 0.0;
std::string sFrameRateOverlay;

// flag for overlay of image processing information
bool bShowInfo = false;

// flag for impact site prediction
bool bPredictImpactSite = true;

// flag for visualization of standard deviation of impact site prediction as elliptical overlay
bool bShowSigma = true;

// deviation of droplet shooting angle between shots in degrees
double dVarAlpha = 2.5; // 5 * (approximate value for medium nozzle velocity) (i.e., 5-sigma zone)

// integer for program control by user
int nKey = 0;

// variables for stereo reconstruction parameters
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

// variables for WLS filter parameters
int nLambda;
float fSigmaColor;

// variables for tolerances
double tol_pi, tol_para;

// parameter declarations for droplet trajectory fit plane
cv::Vec3f n_PI_0;   // normal vector of Hesse normal form of plane PI
double d;           // distance from plane PI to coordinate system origin

// parameter declarations for linear fit function (DEPRECATED MODEL)
cv::Point3f suppVecLin = cv::Point3f(0.0,0.0,0.0);
cv::Point3f dirVecLin = cv::Point3f(0.0,0.0,0.0);

// vector for parabolic trajectory parameters (STANDARD MODEL)
cv::Mat vF = cv::Mat::zeros(9, 1, CV_32F);

// scale factor for fiberoptical system image display
float fDisplayScale = 3.5;

int nOverride = -1;

// define ROIs
int nLeftPicLeftUpperCorner[2] = {0, 0};        // in px
int nRightPicLeftUpperCorner[2] = {0, 0};       // in px

int nImageSize[2] = {0 ,0};                     // in px

// calibration pattern properties
float fDistanceBetweenCircles = 0.0;            // in mm
cv::Size circlesGridSize;

// camera parameters
int nFramerateLive = 30;
XS_HANDLE hCamera;
XS_FRAME frame;
XSULONG32 nROIOriginX, nROIOriginY, nHeight, nWidth;
XS_SETTINGS xsCfg;

int nOriginLeftX, nOriginLeftY, nOriginRightX, nOriginRightY;

bool bCamInitErr = false;

// FUNCTIONS

// initialize high-speed camera connection
bool initOS7(XS_HANDLE &hCamera, XS_FRAME &frame, XSULONG32 &nROIOrigX, XSULONG32 &nROIOrigY, XSULONG32 &nH, XSULONG32 &nW, XS_SETTINGS &xsCfg){

    XS_ENUMITEM xsl[10];
    XSULONG32 nEnumFlt, nListLen = sizeof(xsl)/sizeof(XS_ENUMITEM);
    XSULONG32 nSnsType, nPD, nValHi, nBufferSize;
    XS_ERROR nErrCodeDrv, nErrCodeEnum, nErrSettings;

    nErrCodeDrv = XsLoadDriver(0);

    if (nErrCodeDrv !=XS_SUCCESS)
    {
        std::cout << "High-speed camera driver could not be loaded! Terminating application..." << std::endl;
        cv::waitKey();
        return true;
    }

    // filter settings for camera identification
    nEnumFlt = XS_EF_GE_Y|XS_EF_GE_N;

    nErrCodeEnum = XsEnumCameras(&xsl[0], &nListLen, nEnumFlt);

    if (nErrCodeEnum !=XS_SUCCESS)
    {
        std::cout << "Identification of high-speed camera failed! Terminating application..." << std::endl;
        XsUnloadDriver();
        cv::waitKey();
        return true;
    }

    if (nListLen==0 || xsl[0].bIsOpen==1)
    {
        std::cout << "No camera found! Hit key to end program execution..." << std::endl;
        cv::waitKey();
        XsUnloadDriver();
        return true;
    }

    // open camera
    XsOpenCamera( xsl[0].nCameraId, &hCamera );

    std::cout << "Connection to high-speed camera established!" << std::endl;

    // read camera configuration
    xsCfg.cbSize = sizeof(XS_SETTINGS);
    XsReadCameraSettings( hCamera, &xsCfg );

    // enable Jumbo Packets
    // information: MTU of network adapter must be changed using terminal command: ifconfig eno1 mtu 9000 (or similar)
    XSULONG32 netMTU;
    netMTU = xsl[0].nGeAdpMTU;
    qDebug() << "Network Adapter MTU: " <<  netMTU;

    // check if jumbo packets are supported
    XSULONG32 suppDGR = xsl[0].bDgrSize;
    if (suppDGR == 1 && netMTU == 9000){

        XS_ERROR jumboErr = XsSetParameter(hCamera, &xsCfg, XSP_DGR_SIZE, XS_DGR_8648);

        if(jumboErr != XS_SUCCESS){
            std::cerr << "Error while setting MTU size (required for jumbo packet support)!\n"
                         "Error code: " << jumboErr << std::endl;
        }
        else qDebug() << "Jumbo packets enabled!";

    }
    else{
        qDebug() << "Jumbo packets not enabled!\n"
                    "Check if camera supports jumbo packets and change MTU to 9000 in network adapter configuration.";
    }

    // frame rate 30 fps -> T approx. 33 ms
    XSULONG32 nPeriodNS = (int)(1000000000./(double)nFramerateLive + 0.5);
    XSULONG32 nMaxExposure;

    // exposure time in nanoseconds
    XSULONG32 nExposure = 7000000;

    // set frame recording period according to desired frame rate
    XsSetParameter(hCamera, &xsCfg, XSP_PERIOD, nPeriodNS);

    // read maximum exposure time
    XsGetParameter(hCamera, &xsCfg, XSP_EXPOSURE_MAX, &nMaxExposure);

    // set exposure time to maximum value
    // XsSetParameter(hCamera, &xsCfg, XSP_EXPOSURE, nMaxExposure);

    // set exposure time
    XsSetParameter(hCamera, &xsCfg, XSP_EXPOSURE, nExposure);

    // get information about image and set pixel depth
    XsGetCameraInfo(hCamera, XSI_SNS_TYPE, &nSnsType, &nValHi);

    XSULONG32 nMaxWid,nMaxHgt;

    // get the maximum image size of camera sensor
    XsGetParameter(hCamera, &xsCfg, XSP_MAX_WIDTH, &nMaxWid);
    XsGetParameter(hCamera, &xsCfg, XSP_MAX_HEIGHT, &nMaxHgt);

    //    // set recording ROI to cover full sensor
    //    XsSetParameter(hCamera, &xsCfg, XSP_ROIX, 0);
    //    XsSetParameter(hCamera, &xsCfg, XSP_ROIY, 0);
    //    nW = nMaxWid;
    //    nH = nMaxHgt;
    //    XsSetParameter(hCamera, &xsCfg, XSP_ROIWIDTH, nW);
    //    XsSetParameter(hCamera, &xsCfg, XSP_ROIHEIGHT, nH);

    // set recording ROI to cover left and right sub-images according to "settings.xml" file
    XsSetParameter(hCamera, &xsCfg, XSP_ROIX, nROIOrigX);
    XsSetParameter(hCamera, &xsCfg, XSP_ROIY, nROIOrigY);
    XsSetParameter(hCamera, &xsCfg, XSP_ROIWIDTH, nW);
    XsSetParameter(hCamera, &xsCfg, XSP_ROIHEIGHT, nH);

    if(nSnsType==XS_ST_COLOR) nPD = 24;
    else nPD = 8;

    XsSetParameter( hCamera, &xsCfg, XSP_PIX_DEPTH, nPD );

    // set recording mode to "normal"
    XsSetParameter( hCamera, &xsCfg, XSP_REC_MODE, XS_RM_NORMAL );

    // fill out fields in XS_FRAME structure
    if(nPD<9) nBufferSize = nW*nH;
    else if(nPD<17) nBufferSize = 2*nW*nH;
    else if(nPD<25) nBufferSize = 3*nW*nH;
    else nBufferSize = 6*nW*nH;

    frame.nBufSize = nBufferSize;
    frame.pBuffer = malloc(frame.nBufSize);
    frame.nImages = 1;

    // validate camera settings before sending to camera
    nErrSettings=XsValidateCameraSettings(hCamera, &xsCfg);

    if (nErrSettings != XS_SUCCESS)
    {
        std::cout << "Camera settings faulty! Terminating application..." << std::endl;
        qDebug() << "Settings error code: " << nErrSettings;
        waitKey();
        XsCloseCamera(hCamera);
        XsUnloadDriver();
        return true;
    }

    // send settings to the camera
    XsRefreshCameraSettings(hCamera, &xsCfg);

    return false;
}

// terminate high-speed camera connection
void exitOS7(XS_HANDLE &hCamera, XS_FRAME &frame){

    // free the buffer
    free(frame.pBuffer);

    // close the camera
    XsCloseCamera(hCamera);

    // unload the driver
    XsUnloadDriver();
}

// record and save frame sequence (NOT USED HERE)
void recVid(XS_HANDLE hCamera, XS_FRAME frame, XSULONG32 nHeight, XSULONG32 nWidth, XS_SETTINGS xsCfg, const char* name, int framerate, int length_in_sec)
{
    // record frame sequence
    std::cout << "\n Frame sequence is being recorded... (" << length_in_sec << "sec)" << std::endl;

    // deactivate Fast Live Mode
    if (XsLive(hCamera, XS_LIVE_STOP) != XS_SUCCESS){
        std::cout << "Fast Live Mode could not be deactivated.\n"
                     "Recording mode finished." << std::endl;
        return;
    }

    // code section below written by Robin Szymanski
    XSULONG32 nStartAddLo=0, nStartAddHi=0;
    XSULONG32 nBusy, nSts, ROI_hgt_rec;
    XSULONG32 period_rec;
    cv::Mat rec_img;

    cv::VideoWriter writer;
    bool bIsColor;
    ROI_hgt_rec = XsGetParameter(hCamera, &xsCfg, XSP_ROIHEIGHT, &ROI_hgt_rec);

    /// ============ set parameters for recording ============================

    // set frame rate
    period_rec =  (int)(1000000000./(double)framerate + 0.5);
    if(XsSetParameter( hCamera, &xsCfg, XSP_PERIOD, period_rec ) != XS_SUCCESS)
        std::cerr << "Error while setting frame acquisition period in frame sequence recording function recVid()!" << std::endl;

    std::cout << "Frame rate: " << framerate << "    Frame acquisition period: " << period_rec << std::endl;

    // total number of frames in recording
    int nOfFrames = framerate * length_in_sec;

    // stay under 1350 fps
//            // ROI - Varies with FPS - Has to be adjusted, otherwise the program might crash
//            if (framerate > 1350) { // below 1350 fps, all resolutions up to the maximum resolution are possible

//                std::ifstream tab_fps_res("data/tab_fps_res.txt");
//                unsigned int tab_fps, tab_ROI_height;

//                while(tab_fps_res >> tab_fps >> tab_ROI_height)
//                    if( framerate <= tab_fps )
//                        break;
//                std::cout << "Found in table: FPS:" << tab_fps << " . ROI-Height: " << tab_ROI_height << std::endl;

//                XSULONG32 ROI_y_rec = (int) ((1280 - tab_ROI_height)/2); // center of updated ROI in y direction. Don't do anything in x. 1280 is maximum size.
//                XsSetParameter( hCamera, &xsCfg, XSP_ROIY, ROI_y_rec );
//                XsSetParameter( hCamera, &xsCfg, XSP_ROIHEIGHT, XSULONG32(tab_ROI_height) );
//            } else {
//                ROI_hgt_rec = ROI_height;
//            }

    ROI_hgt_rec = nHeight;

    XSULONG32 nExpRecMax;

    // maximum exposure time changes with frame rate
    XsGetParameter(hCamera,&xsCfg, XSP_EXPOSURE_MAX, &nExpRecMax);

    if(XsSetParameter( hCamera, &xsCfg, XSP_EXPOSURE, nExpRecMax ) != XS_SUCCESS)
        std::cerr << "Error while setting exposure time in frame sequence recording function recVid()!" << std::endl;

    std::cout << "Exposure time set to " << nExpRecMax << std::endl;

    // update camera with new settings
    if(XsRefreshCameraSettings(hCamera, &xsCfg)!=XS_SUCCESS)
        std::cerr << "Error while updating camera settings!" << std::endl;

    /// ==============================================================

    // read live offset and use it as start address
    XsGetCameraInfo(hCamera,XSI_LIVE_BUF_SIZE,&nStartAddLo, &nStartAddHi);

    // start frame acquisition (do not install any callback)
    /* XS_ERROR XsMemoryStartGrab (XS_HANDLE hCamera, XSULONG32 nStartAddLo, XSULONG32 nStartAddHi, XSULONG32 nFrames,
     *  XSULONG32 nPreTrigFrames, XS_AsyncCallback pfnCallback, XSULONG32 nFlags, void *pUserData)
     * nStartAddLo and nStartAddHi: low and High byte of starting address.
     * nFrames: specifies the number of frames which have to be acquired.
     * nPreTrigFrames: specifies the number of frames to be acquired before the trigger; it’s valid only if the trigger source is a single pulse.
     * pUserData: specifies a parameter passed to the callback routine; it may be a pointer to user data.
     * XS_CF_DONE: callback is called only when the operation is completed. */

    if (XsMemoryStartGrab(hCamera, nStartAddLo, nStartAddHi, nOfFrames, 0, NULL, 0, NULL) != XS_SUCCESS)
    {
        std::cout << "Error in XsMemoryStartGrab!" << std::endl;
        return;
    }

    XSULONG32 nPD;

    // DELETE LATER =======================================
    // just in case the image size depends on the current ROI & image format
    XsGetParameter(hCamera, &xsCfg, XSP_ROIWIDTH, &nWidth);
    XsGetParameter(hCamera, &xsCfg, XSP_ROIHEIGHT, &ROI_hgt_rec);
    XsGetParameter(hCamera, &xsCfg, XSP_PIX_DEPTH, &nPD);
    //=======================================================

    // fill out fields in XS_FRAME structure

    frame.nImages = 1;
    if(nPD<9) frame.nBufSize = nWidth*nHeight*frame.nImages;
    else if(nPD<17) frame.nBufSize = 2*nWidth*nHeight*frame.nImages;
    else if(nPD<25) frame.nBufSize = 3*nWidth*nHeight*frame.nImages; // 24 -> 3 bytes per pixel
    else frame.nBufSize = 6*nWidth*nHeight*frame.nImages;

    frame.pBuffer = malloc(frame.nBufSize);

    // time stamps
    time_t now, cur;
    time(&now);
    cur = now;

    while( difftime(cur,now)<3.) {
        nBusy = nSts = 0;
        XsGetCameraStatus(hCamera,&nBusy,&nSts,0,0,0,0);
        if(nBusy==0 && nSts!=XSST_REC_PRETRG && nSts!=XSST_REC_POSTRG)
                break;
       // calculate elapsed time
       time(&cur);
    }

    XsMemoryReadFrame(hCamera, nStartAddLo, nStartAddHi, 5, frame.pBuffer);

    // initialize here once to open video writer before loop
    rec_img = cv::Mat(nHeight, nWidth, CV_8UC3, frame.pBuffer);
    bIsColor = (rec_img.type() == CV_8UC3);

    writer.open(name, CV_FOURCC('m', 'p', '4', 'v'), 30, rec_img.size(), bIsColor);

    // check success
    if (!writer.isOpened()) {
        std::cerr << "Could not initialize output video file for frame sequence storage! \n" << std::endl;
        return;
    }

    // GRAB AND WRITE LOOP
    std::cout << "Writing frame sequence to file " << name << std::endl;

    // start at i=1 or 2! i=0 is a white picture!

    for(int i=1; i<nOfFrames; i++) {
        if(XsMemoryReadFrame(hCamera, nStartAddLo, nStartAddHi, i, frame.pBuffer)!=XS_SUCCESS) {
            std::cout << "Frame no: " << i+1 << ". No more frames available or error in XsMemoryReadFrame!" << std::endl;
            break;
        }

        rec_img = Mat(nHeight, nWidth, CV_8UC3, frame.pBuffer);
        if(rec_img.empty()) {
            std::cerr << "rec_frame is empty" << std::endl;
            break;
        }

        // encode the frame into the videofile stream
        writer.write(rec_img);
    }

    writer.release();

    std::cout << "\n Frame sequence recording completed.\n" << std::endl;

    // set exposure time and frame rate
    XSULONG32 nPeriodNS = (int)(1000000000./(double)nFramerateLive + 0.5);
    XSULONG32 nMaxExposure;

    // set frame rate
    XsSetParameter(hCamera, &xsCfg, XSP_PERIOD, nPeriodNS);

    // get maximum exposure time
    XsGetParameter(hCamera, &xsCfg, XSP_EXPOSURE_MAX, &nMaxExposure);

    // set maximum exposure time
    XsSetParameter( hCamera, &xsCfg, XSP_EXPOSURE, nMaxExposure);

    if(XsRefreshCameraSettings(hCamera, &xsCfg)!=XS_SUCCESS)
        std::cerr << "Error while updating camera settings!" << std::endl;

    // reactivate Fast Live Mode
    XS_ERROR nErrFastLive = XsLive (hCamera, XS_LIVE_START);

    if (nErrFastLive!=XS_SUCCESS)
    {
        XS_ERROR errReset = XsReset(hCamera);
        XS_ERROR nErrFastLiveRepeat = XsLive (hCamera, XS_LIVE_START);

        if (nErrFastLiveRepeat)
        {

        std::cout << "Fast Live Mode could not be reactivated! \n"
                     "Hit key to continue..." << std::endl;

        // wait for user input
        cv::waitKey();

        // free camera buffer
        exitOS7(hCamera, frame);
        return;
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

// erase all files in folder
bool eraseFiles(std::string sFolderName)
{
    // path to folder
    std::string sDirPath = sProjectPath + "/" + sFolderName;
    const char* cDirPath = sDirPath.c_str();

    // loading error
    bool bError = false;

    // folder
    DIR* pDir = opendir(cDirPath);
    std::string sFullPath;

    // files in folder
    struct dirent* pFilesInDir;

    // check if folder exists
    if (pDir == NULL)
    {
        std::cout << "No folder \"" << cDirPath << "\" found. \n" << std::endl;

        bError = true;

        return bError;
    }

    // counter
    int nCount = 0;

    // delete images
    while((pFilesInDir = readdir(pDir)) != NULL)
    {
        // exclude folders "." and ".."
        if (!(strcmp (pFilesInDir->d_name, ".")) || !(strcmp(pFilesInDir->d_name, "..")))
        {
            continue;
        }

        // full path
        sFullPath = sDirPath + "/" + std::string(pFilesInDir->d_name);
        const char* cFullPath = sFullPath.c_str();

        // delete current image
        bool bDelErr = std::remove(cFullPath);

        if (bDelErr)
        {
            std::cout << "Warning! File " << pFilesInDir->d_name << " could not be deleted." << std::endl;

            bError = true;
        }
        else
        {
            nCount++;
        }

    }

    std::cout << nCount << " files deleted from folder " << cDirPath << "." << std::endl;

    closedir(pDir);
    return bError;
}

// retrieve left and right stereoscopic sub-image from sensor of high-speed camera
void splitImages(cv::Mat &pImage, int nOrigLeftX, int nOrigLeftY, int nOrigRightX, int nOrigRightY, cv::Mat &pLeftImage, cv::Mat &pRightImage)
{
    // define ROIs
    cv::Rect leftRectangle = cv::Rect(nOrigLeftX, nOrigLeftY, nImageSize[0], nImageSize[1]);
    cv::Rect rightRectangle = cv::Rect(nOrigRightX, nOrigRightY, nImageSize[0], nImageSize[1]);

    // extract left image
    pLeftImage = pImage(leftRectangle);

    // extract right image
    pRightImage = pImage(rightRectangle);
}

// calculate average geometric distance between sampling points and parabolic fit function
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
        t_starred = 0.0;    // value will yield very high distance as this time stamp corresponds to droplet position at first observation by stereo laryngoscope after ejection from nozzle
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

// MAIN PROGRAM

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    cv::ocl::setUseOpenCL(true);

    // image boundary handling
    const Scalar borderValueRemap = (0);

    // configure stereo matching parameters

    int nTextureThreshold = 10;

    // shared parameters for both stereo laryngoscopes
    // (heuristically optimized by Mr Dava in 2020)

    nUniquenessRatio = 5;   // must be > 0
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
        nNumOfDisparities = 64; // was 112, must be >0 and divisible by 16 for BinarySGBM
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

    // kernel size of OPTIONAL median and Gaussian filter
    // int nKernelSize = 9;

    // parameters of OPTIONAL WLS filter
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

    // declare calibration data matrices (RAW IMAGE DISPLAY)
    cv::Mat pCameraMatrixLeftRAW, pCameraMatrixRightRAW;

    // initialize camera matrices
    pCameraMatrixLeft = cv::Mat::eye(3,3, CV_64F);
    pCameraMatrixRight = cv::Mat::eye(3,3, CV_64F);
    pCameraMatrixLeftRAW = cv::Mat::eye(3,3, CV_64F);
    pCameraMatrixRightRAW = cv::Mat::eye(3,3, CV_64F);

    // initialize stereo reprojection error in px
    double dRepErrorComplete = 9999.0;

    // declare structure for ROI size
    cv::Size pImageSize;

    // declare structure for ROI size (RAW IMAGE DISPLAY)
    cv::Size pImageSizeRAW;

    // declare Mat structures for rotation and translation between virtual cameras
    cv::Mat pLeftToRightRotationMatrix, pLeftToRightTranslationVector;

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
    fs["imageSize"] >> pImageSizeRAW;
    fs["imageSize"] >> pImageSize;
    fs["CameraMatrixLeft"] >> pCameraMatrixLeftRAW;
    fs["DistortionCoefficientsLeft"] >> pDistortionCoeffsLeft;
    fs["CameraMatrixRight"] >> pCameraMatrixRightRAW;
    fs["DistortionCoefficientsRight"] >> pDistortionCoeffsRight;
    fs["RotationMatrix"] >> pLeftToRightRotationMatrix;
    fs["TranslationVector"] >> pLeftToRightTranslationVector;
    fs["ReprojectionError"] >> dRepErrorComplete;

    fs.release();

    std::cout << "Calibration parameters successfully read. Stereo rectification parameter caculation ongoing... \n" << std::endl;

    // declare structures for rectification parameters and look-up-maps
    cv::Mat pR1Left, pR2Right, pP1Left, pP2Right, pQ, pMapLeft1, pMapLeft2, pMapRight1, pMapRight2;

    // declare structures for rectification parameters and look-up-maps (RAW IMAGE DISPLAY)
    cv::Mat pR1LeftRAW, pR2RightRAW, pP1LeftRAW, pP2RightRAW, pQRAW, pMapLeft1RAW, pMapLeft2RAW, pMapRight1RAW, pMapRight2RAW;

    // declare structures for look-up-maps (UNDISTORTED IMAGE DISPLAY)
    cv::Mat pMapLeft1UNDIST, pMapLeft2UNDIST;

    // ROI declarations
    cv::Rect validROIL, validROIR;

    // ROI declarations (RAW IMAGE DISPLAY)
    cv::Rect validROIL_RAW, validROIR_RAW;

    // adapt calibration parameters for frame rescaling
    if(bRescaleFrames)
    {
        // f_x (left)
        pCameraMatrixLeft.at<double>(0,0) = pCameraMatrixLeftRAW.at<double>(0,0) * dRescaleFactor;

        // c_x (left)
        pCameraMatrixLeft.at<double>(0,2) = pCameraMatrixLeftRAW.at<double>(0,2) * dRescaleFactor;

        // f_y (left)
        pCameraMatrixLeft.at<double>(1,1) = pCameraMatrixLeftRAW.at<double>(1,1) * dRescaleFactor;

        // c_y (left)
        pCameraMatrixLeft.at<double>(1,2) = pCameraMatrixLeftRAW.at<double>(1,2) * dRescaleFactor;

        // f_x (right)
        pCameraMatrixRight.at<double>(0,0) = pCameraMatrixRightRAW.at<double>(0,0) * dRescaleFactor;

        // c_x (right)
        pCameraMatrixRight.at<double>(0,2) = pCameraMatrixRightRAW.at<double>(0,2) * dRescaleFactor;

        // f_y (right)
        pCameraMatrixRight.at<double>(1,1) = pCameraMatrixRightRAW.at<double>(1,1) * dRescaleFactor;

        // c_y (right)
        pCameraMatrixRight.at<double>(1,2) = pCameraMatrixRightRAW.at<double>(1,2) * dRescaleFactor;

        std::cout << "pCameraMatrixLeft: " << pCameraMatrixLeft << std::endl;
        std::cout << "pCameraMatrixRight: " << pCameraMatrixRight << std::endl;

        // image size
        pImageSize.height *= dRescaleFactor;
        pImageSize.width *= dRescaleFactor;
    }

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

    // calculate stereo image rectification parameters (RAW IMAGE DISPLAY)
    cv::stereoRectify(pCameraMatrixLeftRAW,
                      pDistortionCoeffsLeft,
                      pCameraMatrixRightRAW,
                      pDistortionCoeffsRight,
                      pImageSizeRAW,
                      pLeftToRightRotationMatrix,
                      pLeftToRightTranslationVector,
                      pR1LeftRAW,
                      pR2RightRAW,
                      pP1LeftRAW,
                      pP2RightRAW,
                      pQRAW,
                      CALIB_ZERO_DISPARITY,             // flag for disparity at infinity (0 for parallel setup)
                      -1,                               // flag "alpha" for pixel range to be considered
                      pImageSizeRAW,                    // image size after rectification
                      &validROIL_RAW,                   // valid ROI left
                      &validROIR_RAW);                  // valid ROI right


    std::cout << "Q matrix: " << pQ << std::endl;

    std::cout << "Rectification rotation matrix R1: " << pR1LeftRAW << std::endl;

    // pre-calculation of look-up maps for fast undistortion and rectification

    // calculate look-up maps for left virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixLeft, pDistortionCoeffsLeft, pR1Left,
                                pP1Left, pImageSize, CV_16SC2, pMapLeft1, pMapLeft2);

    // calculate look-up maps for right virtual camera
    cv::initUndistortRectifyMap(pCameraMatrixRight, pDistortionCoeffsRight, pR2Right,
                                pP2Right, pImageSize, CV_16SC2, pMapRight1, pMapRight2);

    // calculate look-up maps for left virtual camera (RAW IMAGE DISPLAY)
    cv::initUndistortRectifyMap(pCameraMatrixLeftRAW, pDistortionCoeffsLeft, pR1LeftRAW,
                                pP1LeftRAW, pImageSizeRAW, CV_16SC2, pMapLeft1RAW, pMapLeft2RAW);

    // calculate look-up maps for right virtual camera (RAW IMAGE DISPLAY)
    cv::initUndistortRectifyMap(pCameraMatrixRightRAW, pDistortionCoeffsRight, pR2RightRAW,
                                pP2RightRAW, pImageSizeRAW, CV_16SC2, pMapRight1RAW, pMapRight2RAW);

    // calculate look-up maps for left virtual camera (UNDISTORTED IMAGE DISPLAY)
    cv::initUndistortRectifyMap(pCameraMatrixLeftRAW, pDistortionCoeffsLeft, cv::Mat(),
                                pCameraMatrixLeftRAW, pImageSizeRAW, CV_16SC2, pMapLeft1UNDIST, pMapLeft2UNDIST);

    // read trajectory information from file "TrajInfos.yml"

    // load defining parameters of plane PI from file "TrajInfos.yml"
    std::string filename = "TrajInfos.yml";
    cv::FileStorage fs_traj;

    if(!fs_traj.open(filename, cv::FileStorage::READ))
    {
        std::cout << "No file \"TrajInfos.yml\" available.\n"
                     "Hit key to end program execution..."<< std::endl;

        cv::waitKey(0);

        return 0;
    }
    else
    {
        // load parameters of best-fit plane PI in Hesse normal form from file "TrajInfos.yml"
        fs_traj["n_PI_0"] >> n_PI_0;
        fs_traj["d"] >> d;

        // load parameters of linear trajectory model (DEPRECATED) from file "TrajInfos.yml"

        // load parameters from file "TrajInfos.yml"
        fs_traj["dirVecLin"] >> dirVecLin;
        fs_traj["suppVecLin"] >> suppVecLin;

        // load parameter vector of parabolic trajectory model (STANDARD) from file "TrajInfos.yml"
        fs_traj["F"] >> vF;
        fs_traj.release();
    }

    std::cout << "Settings, calibration and trajectory parameter files successfully loaded!" << std::endl;

    /// ____________________________________________________LIVE STEREO RECONSTRUCTION PROCEDURE___________________________________________________________

    // initialize connection to high-speed camera

    // set horizontal coordinate of origin of global ROI
    nROIOriginX = nLeftPicLeftUpperCorner[0];

    if(nLeftPicLeftUpperCorner[1] < nRightPicLeftUpperCorner[1])
    {
        // set vertical coordinate of origin of global ROI
        nROIOriginY = nLeftPicLeftUpperCorner[1];

        // set height of global ROI
        nHeight = nRightPicLeftUpperCorner[1] + nImageSize[1] - nLeftPicLeftUpperCorner[1];

        // set origin of left sub-image for later extraction
        nOriginLeftX = 0;
        nOriginLeftY = 0;
        // set origin of right sub-image for later extraction
        nOriginRightX = nRightPicLeftUpperCorner[0] - nROIOriginX;
        nOriginRightY = nRightPicLeftUpperCorner[1] - nLeftPicLeftUpperCorner[1];

    }
    else
    {
        // set vertical coordinate of origin of global ROI
        nROIOriginY = nRightPicLeftUpperCorner[1];

        // set height of global ROI
        nHeight = nLeftPicLeftUpperCorner[1] + nImageSize[1] - nRightPicLeftUpperCorner[1];

        // set origin of left sub-image for later extraction
        nOriginLeftX = 0;
        nOriginLeftY = nLeftPicLeftUpperCorner[1] - nRightPicLeftUpperCorner[1];
        // set origin of right sub-image for later extraction
        nOriginRightX = nRightPicLeftUpperCorner[0] - nROIOriginX;
        nOriginRightY = 0;
    }

    // set width of global ROI
    nWidth = nRightPicLeftUpperCorner[0] + nImageSize[0] - nROIOriginX;

    std::cout << "nROIOriginX: " << nROIOriginX << std::endl;
    std::cout << "nROIOriginY: " << nROIOriginY << std::endl;
    std::cout << "nHeight: " << nHeight << std::endl;
    std::cout << "nWidth: " << nWidth << std::endl;

    bCamInitErr = initOS7(hCamera, frame, nROIOriginX, nROIOriginY, nHeight, nWidth, xsCfg);

    if (bCamInitErr) qDebug() << "Error during camera initialization.";

    cv::waitKey(1000);

    // enable Fast Live Mode
    XS_ERROR nErrFastLive;

    nErrFastLive = XsLive(hCamera, XS_LIVE_START);

    // error handling
    if (nErrFastLive != XS_SUCCESS)
    {
        // reset camera
        XS_ERROR errReset = XsReset(hCamera);

        cv::waitKey(2000);

        XS_ERROR nErrFastLiveRepeat = XsLive(hCamera, XS_LIVE_START);

        if(nErrFastLiveRepeat)
        {

        std::cout << "Fast Live Mode could not be started!\n"
                     "Hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey();

        // free camera buffer
        exitOS7(hCamera, frame);
        return 0;
        }
    }

    std::cout << "Fast Live Mode successfully activated!" << std::endl;

    // initialize cv::Mat objects

    cv::Mat pCurFrame = cv::Mat(nHeight, nWidth, CV_8UC3);

    cv::Mat pLeftImg, pRightImg, pLeftImgScaled, pRightImgScaled, pLeftImgRect, pRightImgRect, pLeftImgRectGrayscale, pRightImgRectGrayscale, pDispMap, pDispMapWLS;

    // declarations for RAW IMAGE DISPLAY

    cv::Mat pLeftImgRectRAW, pLeftImgUndist, pBlackAreas, pWhiteAreas, pTempLeft, pDepthMap;

    // temporary helper point for impact site prediction
    cv::Point3f TempSamplingPoint = cv::Point3f(0.0,0.0,0.0);

    // vector for stereo reconstruction 3D points
    std::vector<cv::Point3f> vSamplingPoints;
    vSamplingPoints.resize(0);

    // vector for impact site candidate points
    std::vector<cv::Point3f> vImpactSiteCandidates;
    vImpactSiteCandidates.resize(0);

    // 3D point representing parabolic impact site prediction
    cv::Point3f P_imp_para = cv::Point3f(0.0,0.0,5000.0);

    // 3D point representing parabolic impact site prediction (RAW IMAGE DISPLAY)
    cv::Point3f P_imp_para_RAW = cv::Point3f(0.0,0.0,5000.0);

    // vector for parabolic impact site prediction
    std::vector<cv::Point3f> vImpactSiteCandidatesPara;
    vImpactSiteCandidatesPara.resize(0);

    // distance between 3D point and parabolic trajectory approximation function (here: in mm)
    double dist_para = 0.0;

    // 2D image coordinates of impact site prediction
    cv::Point2f P_imp_para_2D = cv::Point2f(0.0,0.0);

    // vectors for zero rotation and translation (for reprojection of impact site into RECTIFIED left image)
    std::vector<double> vRotZero(3, 0.0);
    std::vector<double> vTranslatZero(3, 0.0);

    // vectors for compatibility with cv::projectPoints()
    std::vector<cv::Point2f> vImpPredPara_2D;
    vImpPredPara_2D.resize(1);

    std::vector<cv::Vec3f> vP_imp_para;
    vP_imp_para.resize(1);

    std::vector<cv::Vec3f> vP_imp_para_RAW;
    vP_imp_para_RAW.resize(1);

    // rectangle for extraction of sub-matrix of pP1Left
    cv::Rect rectSubMat = cv::Rect(0,0,3,3);

    // new camera matrix for left image
    cv::Mat pNewCamMatrixLeftRAW = pP1LeftRAW(rectSubMat);

    // calculate inverse matrix of R1
    cv::Mat pR1LeftRAWInverse;
    cv::transpose(pR1LeftRAW, pR1LeftRAWInverse);

    // declarations for color-coded disparity map
    double min, max;
    cv::Mat pCM_disp, pScaledDispMap, pCM_dispView;

    // declarations for sigma visualization (3D)
    cv::Point3f P_Vector_u = cv::Point3f(0.0,0.0,0.0);
    cv::Point3f P_Vector_v = cv::Point3f(0.0,0.0,0.0);
    double dSigmaRadius = 0.0;

    cv::Point3f P_Point_on_Circle = cv::Point3f(0.0,0.0,0.0);

    std::vector<cv::Point3f> vSigmaCirclePoints;
    vSigmaCirclePoints.resize(0);

    // vector for compatibility with cv::projectPoints() (2D)
    std::vector<cv::Point2f> vCirclePoints2D;
    vCirclePoints2D.resize(0);

    // initialize SGBM correspondence matcher
    if(bBinarySGBM)
    {
        pLeftMatcherSGBMBinary->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_ALGORITHM); // speckle removal technique

        pLeftMatcherSGBMBinary->setSubPixelInterpolationMethod(cv::stereo::CV_QUADRATIC_INTERPOLATION); // interpolation technique

        // binary descriptor type
        // SOME KERNEL TYPES DO NOT YIELD 100 % REPRODUCIBLE STEREO RECONSTRUCTION RESULTS (POINT CLOUDS)
        // KERNEL IDs: 0 CV_DENSE_CENSUS, 1 CV_SPARSE_CENSUS, 2 CV_CS_CENSUS, 3 CV_MODIFIED_CS_CENSUS, 4 CV_MODIFIED_CENSUS_TRANSFORM, 5 CV_MEAN_VARIATION, 6 CV_STAR_KERNEL
        // source code for calculation of descriptors available here: https://github.com/opencv/opencv_contrib/blob/master/modules/stereo/src/descriptor.cpp
        // (test with 3D phantom) 0: not reproducible, 1: not reproducible, 2: reproducible, 3: not reproducible, 4: reproducible, 5: not reproducible, 6: not reproducible
        // (test with 2D target): 0: not reproducible, 1: not reproducible, 2: reproducible, 3: not reproducible, 4: not reproducible, 5: not reproducible, 6: not reproducible
        pLeftMatcherSGBMBinary->setBinaryKernelType(2);
    }

    // initialize WLS filter
    if(bWLS)
    {
        wls_filter->setDepthDiscontinuityRadius((int)std::ceil(0.5*nSADWinSize));
        wls_filter->setLambda(nLambda);
        wls_filter->setSigmaColor(fSigmaColor);
    }

    // initialize windows
    cv::namedWindow("Live View", CV_WINDOW_NORMAL);
    cv::namedWindow("Disparity Image", CV_WINDOW_NORMAL);

    // initialize frame rate calculation
    timespec timeStartCalc, timeEndCalc;

    std::cout << "Starting MIT-LAR application phase." << std::endl;
    std::cout << "p: toggle impact site prediction overlay" << "\n" << "r: toggle image undistortion/rectification" << "\n" << "i: toggle image information overlay" << "\n" << "f: toggle frame rate overlay" << "\n" << "e: quit program" << std::endl;

    ///______________________________________________________________________LIVE APPLICATION LOOP STARTING HERE__________________________________________________________________________

    while(true)
    {
        // std::cout << "nKey: " << nKey << std::endl;
        // if key "p" pressed
        if(nKey == 112)
        {
            if(bPredictImpactSite)
            {
                bPredictImpactSite = false;
            }
            else
            {
                bPredictImpactSite = true;
            }
            nKey = 0;
        }

        // if key "f" pressed
        if(nKey == 102)
        {
            // flip boolean variable for frame rate overlay
            bShowFrameRate = !bShowFrameRate;
        }

        // if key "r" pressed
        if(nKey == 114)
        {
            // flip boolean variable for raw image display
            bUseRawImage = !bUseRawImage;
        }

        // if key "i" pressed
        if(nKey == 105)
        {
            // flip boolean variable for image processing information overlay
            bShowInfo = !bShowInfo;
        }

        // if key "e" pressed
        if(nKey == 101)
        {
            // terminate camera connection
            exitOS7(hCamera, frame);

            cv::waitKey(500);

            // quit program
            return 0;
        }

        clock_gettime(CLOCK_MONOTONIC, &timeStartCalc);

        // grab live frame from camera
        XsMemoryPreview(hCamera, &frame, NULL);

        pCurFrame = cv::Mat(nHeight, nWidth, CV_8UC3, frame.pBuffer);

        if(pCurFrame.empty())
        {
            std::cout << "Live frame display not possible.\n"
                         "Ending program execution..." << std::endl;
            break;
        }

        // extract RAW sub-images
        splitImages(pCurFrame, nOriginLeftX, nOriginLeftY, nOriginRightX, nOriginRightY, pLeftImg, pRightImg);

        // (OPTIONALLY) resize frame
        if(bRescaleFrames && dRescaleFactor != 1.0)
        {
            // isometrically rescale frame by desired scale factor
            cv::resize(pLeftImg, pLeftImgScaled, cv::Size(), dRescaleFactor, dRescaleFactor, (dRescaleFactor < 1.0) ? INTER_AREA : INTER_LINEAR);
            cv::resize(pRightImg, pRightImgScaled, cv::Size(), dRescaleFactor, dRescaleFactor, (dRescaleFactor < 1.0) ? INTER_AREA : INTER_LINEAR);

        }
        else
        {
            pLeftImgScaled = pLeftImg.clone();
            pRightImgScaled = pRightImg.clone();
        }

        // rectify RESCALED images
        cv::remap(pLeftImgScaled, pLeftImgRect, pMapLeft1, pMapLeft2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);
        cv::remap(pRightImgScaled, pRightImgRect, pMapRight1, pMapRight2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

        // rectify left RAW image
        cv::remap(pLeftImg, pLeftImgRectRAW, pMapLeft1RAW, pMapLeft2RAW, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

        // if impact site prediction NOT desired
        if(!bPredictImpactSite)
        {
            if(!bUseRawImage)
            {
                // scale live view to original resolution
                // cv::resize(pLeftImgRect, pLeftImgView, cv::Size(), 1.0/dRescaleFactor, 1.0/dRescaleFactor, (dRescaleFactor < 1.0)? INTER_LINEAR : INTER_AREA);

                // show UNDISTORTED image (not rectified)
                cv::remap(pLeftImg, pLeftImgUndist, pMapLeft1UNDIST, pMapLeft2UNDIST, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

                if(bShowFrameRate)
                {
                    // add current frame rate to live view

                    dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                    sFrameRateOverlay = std::to_string(dFrameRateRounded);

                    sFrameRateOverlay.erase(2,10);

                    sFrameRateOverlay.append(" fps");

                    cv::putText(pLeftImgUndist, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                }

                if(bShowInfo)
                {
                    cv::putText(pLeftImgUndist, "UNDISTORTED IMAGE", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                }

                // display live rectified left image (COLOR)
                cv::imshow("Live View", pLeftImgUndist);
            }
            else
            {
                // show RAW image (without undistortion or rectification)

                if(bShowFrameRate)
                {
                    // add current frame rate to live view

                    dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                    sFrameRateOverlay = std::to_string(dFrameRateRounded);

                    sFrameRateOverlay.erase(2,10);

                    sFrameRateOverlay.append(" fps");

                    cv::putText(pLeftImg, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                }

                if(bShowInfo)
                {
                    cv::putText(pLeftImg, "RAW IMAGE", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                }

                // display live left raw image (COLOR)
                cv::imshow("Live View", pLeftImg);
            }

            clock_gettime(CLOCK_MONOTONIC, &timeEndCalc);

            // calculate computation time
            nCompTimeMS = ((uint64_t)timeEndCalc.tv_sec * 1000LL + (uint64_t)timeEndCalc.tv_nsec / 1000000LL) -
                    ((uint64_t)timeStartCalc.tv_sec * 1000LL + (uint64_t)timeStartCalc.tv_nsec / 1000000LL);

            dFrameRate = 1000.0/(double)nCompTimeMS;
        }
        // if impact site prediction desired
        else
        {
            // convert rectified left and right images to grayscale for BINARY SGBM method
            cv::cvtColor(pLeftImgRect, pLeftImgRectGrayscale, CV_BGR2GRAY);
            cv::cvtColor(pRightImgRect, pRightImgRectGrayscale, CV_BGR2GRAY);

            // perform stereo reconstruction of target
            if(bBinarySGBM)
            {
                // perform stereo matching
                pLeftMatcherSGBMBinary->compute(pLeftImgRectGrayscale, pRightImgRectGrayscale, pDispMap);

                // WLS filtering of raw disparity map (Fast Global Smoother)
                if(bWLS)
                {
                    // perform filtering
                    wls_filter->filter(pDispMap, pLeftImgRectGrayscale, pDispMapWLS, cv::Mat());
                }
                else
                {
                    pDispMapWLS = pDispMap.clone();
                }
            }

            // divide disparity map by 16 (required due to the stereo matcher's output data type)
            pDispMapWLS.convertTo(pDispMapWLS, CV_32F, 1./16);

            // remove areas with high uncertainty (i.e., dark and bright areas) from disparity image

            pTempLeft = pLeftImgRectGrayscale.clone();

            // identify dark and bright images in grayscale image and binarize (thresholding operation)

            // dark areas
            // all pixels with intensity < 20 are set to 255
            // all pixels with intensity > 20 are set to 0
            cv::threshold(pTempLeft, pBlackAreas, 20, 255, THRESH_BINARY_INV);

            // bright areas
            // all pixels with intensity > 240 are set to 255
            // all pixels with intensity < 240 are set to 0
            cv::threshold(pTempLeft, pWhiteAreas, 240, 255, THRESH_BINARY);

            // data type conversion
            pBlackAreas.convertTo(pBlackAreas, CV_32F);
            pWhiteAreas.convertTo(pWhiteAreas, CV_32F);

            // remove dark and bright areas from disparity image
            pDispMapWLS = pDispMapWLS - pBlackAreas;
            pDispMapWLS = pDispMapWLS - pWhiteAreas;

            // remove negative values (resulting from subtraction of a float image)

            // disparities are set to 0 for (-pDispMapWLS > 0) and remain unchanged otherwise

            // all pixels in pDispMapWLS with "NEGATIVE intensity" after subtraction of dark and bright areas (value of these areas: 255) are set to 0
            cv::threshold(-pDispMapWLS, pDispMapWLS, 0, 0, THRESH_TRUNC);

            // invert resulting disparity image
            pDispMapWLS = -pDispMapWLS;

            // compute stereo reconstruction in coordinate system of left virtual camera (point coordinates are returned in calibration pattern units: mm)
            cv::reprojectImageTo3D(pDispMapWLS, pDepthMap, pQ, false, CV_32F);

            /// ________________________________________________________________START IMPACT SITE PREDICTION STEP____________________________________________________________________

            // set tolerances
            if(bRodLens)
            {
                // maximum allowed distance to plane PI in mm
                tol_pi = 0.4;

                // maximum allowed distance to parabolic trajectory approximation in mm (radius of tolerance tube around T_para)
                tol_para = 0.2;
            }
            else
            {
                // maximum allowed distance to plane PI in mm
                tol_pi = 0.8;

                // maximum allowed distance to parabolic trajectory approximation in mm (radius of tolerance tube around T_para)
                tol_para = 0.4;
            }

            // reset vectors
            vSamplingPoints.resize(0);
            vImpactSiteCandidates.resize(0);
            vImpactSiteCandidatesPara.resize(0);

            // for all rows of pDepthMap
            for(int r = 0; r < pDepthMap.rows; r++)
            {
                // for all columns of pDepthMap
                for(int c = 0; c < pDepthMap.cols; c++)
                {
                    TempSamplingPoint.x = pDepthMap.at<Vec3f>(r, c)[0];
                    TempSamplingPoint.y = pDepthMap.at<Vec3f>(r, c)[1];
                    TempSamplingPoint.z = pDepthMap.at<Vec3f>(r, c)[2];

                    // add current point to vector of all points of stereo reconstruction
                    vSamplingPoints.push_back(TempSamplingPoint);
                }
            }

            // ________________________________________remove all points from stereo reconstruction that are too far away from plane PI_________________________________________________

            // go through spatially reconstructed target points
            for (unsigned int i=0; i<vSamplingPoints.size(); ++i)
            {
                // if distance to fit plane lower than tol_pi: add point to set of impact site candidates
                if ((double)std::abs(vSamplingPoints[i].dot(n_PI_0)-d) < tol_pi)
                {
                    vImpactSiteCandidates.push_back(vSamplingPoints[i]);
                }
            }

            // __________________________________________calculation of impact site prediction (parabolic trajectory model)______________________________________________________________

            // for all impact site candidate points
            for (unsigned int i=0; i<vImpactSiteCandidates.size(); ++i)
            {
                // calculate shortest distance between current point and parabolic trajectory model
                dist_para = calculateAverageGeometricDistanceParaSinglePoint(vF, vImpactSiteCandidates[i]);

                // if current candidate located in tolerance tube around parabolic trajectory approximation AND closer to left virtual camera in z direction than previous candidate:
                // update impact site prediction
                if (dist_para < tol_para && vImpactSiteCandidates[i].z < P_imp_para.z)
                {
                    P_imp_para = vImpactSiteCandidates[i];

                    // std::cout << "Current distance to parabola in mm: " << std::sqrt(dist_para_squared) << std::endl;
                }
            }

            // __________________________________________visualization of predicted impact site in laryngoscopic live image______________________________________________________________

            if(P_imp_para.z != 5000.0)
            {
                vImpactSiteCandidatesPara.push_back(P_imp_para);

                // project spatial impact site prediction into left sub-image (COLOR)

                vP_imp_para[0][0] = P_imp_para.x;
                vP_imp_para[0][1] = P_imp_para.y;
                vP_imp_para[0][2] = P_imp_para.z;

                if(!bUseRawImage)
                {
                    // project impact site prediction into left RECTIFIED (and non-rescaled) image
                    cv::projectPoints(vP_imp_para, vRotZero, vTranslatZero, pNewCamMatrixLeftRAW, cv::noArray(), vImpPredPara_2D);

                    std::cout << "vP_imp_para[0]: " << vP_imp_para[0] << std::endl;
                }
                else
                {
                    // transform coordinates of 3D impact site prediction to coordinate system of left UNDISTORTED camera (before rectification)

                    cv::transform(vP_imp_para, vP_imp_para_RAW, pR1LeftRAWInverse);

                    P_imp_para_RAW.x = vP_imp_para_RAW[0][0];
                    P_imp_para_RAW.y = vP_imp_para_RAW[0][1];
                    P_imp_para_RAW.z = vP_imp_para_RAW[0][2];

                    // vP_imp_para_RAW[0][0] = P_imp_para_RAW.x;
                    // vP_imp_para_RAW[0][1] = P_imp_para_RAW.y;
                    // vP_imp_para_RAW[0][2] = P_imp_para_RAW.z;

                    // vP_imp_para_RAW[0][0] = pR1LeftRAW.at<double>(0, 0) * vP_imp_para[0][0] + pR1LeftRAW.at<double>(1, 0) * vP_imp_para[0][1] + pR1LeftRAW.at<double>(2, 0) * vP_imp_para[0][2];
                    // vP_imp_para_RAW[0][1] = pR1LeftRAW.at<double>(0, 1) * vP_imp_para[0][0] + pR1LeftRAW.at<double>(1, 1) * vP_imp_para[0][1] + pR1LeftRAW.at<double>(2, 1) * vP_imp_para[0][2];
                    // vP_imp_para_RAW[0][2] = pR1LeftRAW.at<double>(0, 2) * vP_imp_para[0][0] + pR1LeftRAW.at<double>(1, 2) * vP_imp_para[0][1] + pR1LeftRAW.at<double>(2, 2) * vP_imp_para[0][2];

                    std::cout << "vP_imp_para_RAW[0]: " << vP_imp_para_RAW[0] << std::endl;

                    // project impact site prediction into left UNDISTORTED image (before rectification)
                    cv::projectPoints(vP_imp_para_RAW, vRotZero, vTranslatZero, pCameraMatrixLeftRAW, cv::noArray(), vImpPredPara_2D);
                }

                P_imp_para_2D = vImpPredPara_2D[0];

                std::cout << "P_imp_para_2D: " << P_imp_para_2D << std::endl;

                // show left sub-image (COLOR) WITH impact site prediction to MIT-LAR operator

                // mark impact site prediction on left sub-image (COLOR) in RED

                if(!bUseRawImage)
                {
                    // horizontal line
                    cv::line(pLeftImgRectRAW, cv::Point(P_imp_para_2D.x-15, P_imp_para_2D.y),
                             cv::Point(P_imp_para_2D.x+15, P_imp_para_2D.y), Scalar(0,0,255), 1.5, LINE_AA);
                    // vertical line
                    cv::line(pLeftImgRectRAW, cv::Point(P_imp_para_2D.x, P_imp_para_2D.y-15),
                             cv::Point(P_imp_para_2D.x, P_imp_para_2D.y+15), Scalar(0,0,255), 1.5, LINE_AA);

                    // show sigma circle on RECTIFIED left sub-image (COLOR) in ORANGE (0,136,255)
                    if(bShowSigma)
                    {
                        // calculate sigma radius
                        dSigmaRadius = (double)(tan((0.5*dVarAlpha)*PI/180.0) * P_imp_para.z);

                        // calculate and normalize vector u
                        P_Vector_u.x = 1.0;
                        P_Vector_u.y = 1.0;
                        P_Vector_u.z = -(dirVecLin.x + dirVecLin.y)/dirVecLin.z;

                        P_Vector_u /= cv::norm(P_Vector_u);

                        // calculate and normalize vector v
                        P_Vector_v = P_Vector_u.cross(dirVecLin);

                        P_Vector_v /= cv::norm(P_Vector_v);

                        // reset vector of circle points
                        vSigmaCirclePoints.resize(0);

                        // reset vector of circle points (2D)
                        vCirclePoints2D.resize(0);

                        // draw sigma circle in space
                        for (float fIterator = 0; fIterator < 2*PI; fIterator += 0.05)
                        {
                            // draw point on sigma circle in space
                            P_Point_on_Circle = P_imp_para + dSigmaRadius * (P_Vector_u * cos(fIterator) + P_Vector_v * sin(fIterator));
                            vSigmaCirclePoints.push_back(P_Point_on_Circle);
                        }

                        // project sigma circle onto RECTIFIED left sub-image (COLOR) in ORANGE (0,136,255)

                        // project spatial impact site prediction into RECTIFIED left sub-image (COLOR)
                        cv::projectPoints(vSigmaCirclePoints, vRotZero, vTranslatZero, pNewCamMatrixLeftRAW, cv::noArray(), vCirclePoints2D);

                        for(unsigned int i = 0; i<vCirclePoints2D.size(); ++i)
                        {
                            cv::line(pLeftImgRectRAW, cv::Point(vCirclePoints2D[i].x - 0.5, vCirclePoints2D[i].y),cv::Point(vCirclePoints2D[i].x + 0.5, vCirclePoints2D[i].y), Scalar(0,136,255), 0.6, LINE_AA);
                            cv::line(pLeftImgRectRAW, cv::Point(vCirclePoints2D[i].x, vCirclePoints2D[i].y - 0.5),cv::Point(vCirclePoints2D[i].x, vCirclePoints2D[i].y + 0.5), Scalar(0,136,255), 0.6, LINE_AA);
                        }

                    }

                    if(bShowFrameRate)
                    {
                        // add current frame rate to live view

                        dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                        sFrameRateOverlay = std::to_string(dFrameRateRounded);

                        sFrameRateOverlay.erase(2,10);

                        sFrameRateOverlay.append(" fps");

                        cv::putText(pLeftImgRectRAW, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    if(bShowInfo)
                    {
                        cv::putText(pLeftImgRectRAW, "RECTIFIED IMAGE (PRED. ACTIVE)", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    // display live rectified left image (COLOR)
                    cv::imshow("Live View", pLeftImgRectRAW);

                    // show color-coded disparity image
                    cv::minMaxIdx(pDispMapWLS, &min, &max);
                    cv::convertScaleAbs(pDispMapWLS, pScaledDispMap, 255/(max - min));
                    cv::applyColorMap(pScaledDispMap, pCM_disp, cv::COLORMAP_JET);

                    cv::imshow("Disparity Image", pCM_disp);
                }
                else
                {
                    // UNDISTORT RAW image
                    // cv::undistort(pLeftImg, pLeftImgUndist, pCameraMatrixLeftRAW, pDistortionCoeffsLeft, cv::noArray());

                    cv::remap(pLeftImg, pLeftImgUndist, pMapLeft1UNDIST, pMapLeft2UNDIST, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

                    // draw horizontal line
                    cv::line(pLeftImgUndist, cv::Point(P_imp_para_2D.x-15, P_imp_para_2D.y),
                             cv::Point(P_imp_para_2D.x+15, P_imp_para_2D.y), Scalar(0,0,255), 1.5, LINE_AA);
                    // draw vertical line
                    cv::line(pLeftImgUndist, cv::Point(P_imp_para_2D.x, P_imp_para_2D.y-15),
                             cv::Point(P_imp_para_2D.x, P_imp_para_2D.y+15), Scalar(0,0,255), 1.5, LINE_AA);

                    // show sigma circle on UNDISTORTED left sub-image (COLOR) in ORANGE (0,136,255)
                    if(bShowSigma)
                    {
                        // calculate sigma radius
                        dSigmaRadius = (double)(tan((0.5*dVarAlpha)*PI/180.0) * P_imp_para.z);

                        // calculate and normalize vector u
                        P_Vector_u.x = 1.0;
                        P_Vector_u.y = 1.0;
                        P_Vector_u.z = -(dirVecLin.x + dirVecLin.y)/dirVecLin.z;

                        P_Vector_u /= cv::norm(P_Vector_u);

                        // calculate and normalize vector v
                        P_Vector_v = P_Vector_u.cross(dirVecLin);

                        P_Vector_v /= cv::norm(P_Vector_v);

                        // reset vector of circle points
                        vSigmaCirclePoints.resize(0);

                        // reset vector of circle points (2D)
                        vCirclePoints2D.resize(0);

                        // draw sigma circle in space
                        for (float fIterator = 0; fIterator < 2*PI; fIterator += 0.05)
                        {
                            // draw point on sigma circle in space
                            P_Point_on_Circle = P_imp_para_RAW + dSigmaRadius * (P_Vector_u * cos(fIterator) + P_Vector_v * sin(fIterator));
                            vSigmaCirclePoints.push_back(P_Point_on_Circle);
                        }

                        // project sigma circle onto UNDISTORTED left sub-image (COLOR) in ORANGE (0,136,255)

                        // project spatial impact site prediction into UNDISTORTED left sub-image (COLOR)
                        cv::projectPoints(vSigmaCirclePoints, vRotZero, vTranslatZero, pCameraMatrixLeftRAW, cv::noArray(), vCirclePoints2D);

                        for(unsigned int i = 0; i<vCirclePoints2D.size(); ++i)
                        {
                            cv::line(pLeftImgUndist, cv::Point(vCirclePoints2D[i].x - 0.5, vCirclePoints2D[i].y),cv::Point(vCirclePoints2D[i].x + 0.5, vCirclePoints2D[i].y), Scalar(0,136,255), 0.6, LINE_AA);
                            cv::line(pLeftImgUndist, cv::Point(vCirclePoints2D[i].x, vCirclePoints2D[i].y - 0.5),cv::Point(vCirclePoints2D[i].x, vCirclePoints2D[i].y + 0.5), Scalar(0,136,255), 0.6, LINE_AA);
                        }

                    }

                    if(bShowFrameRate)
                    {
                        // add current frame rate to live view

                        dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                        sFrameRateOverlay = std::to_string(dFrameRateRounded);

                        sFrameRateOverlay.erase(2,10);

                        sFrameRateOverlay.append(" fps");

                        cv::putText(pLeftImgUndist, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    if(bShowInfo)
                    {
                        cv::putText(pLeftImgUndist, "UNDISTORTED IMAGE (PRED. ACTIVE)", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    // display live RAW left image (COLOR)
                    cv::imshow("Live View", pLeftImgUndist);

                    // show color-coded disparity image
                    cv::minMaxIdx(pDispMapWLS, &min, &max);
                    cv::convertScaleAbs(pDispMapWLS, pScaledDispMap, 255/(max - min));
                    cv::applyColorMap(pScaledDispMap, pCM_disp, cv::COLORMAP_JET);

                    cv::imshow("Disparity Image", pCM_disp);
                }

                clock_gettime(CLOCK_MONOTONIC, &timeEndCalc);

                // calculate computation time
                nCompTimeMS = ((uint64_t)timeEndCalc.tv_sec * 1000LL + (uint64_t)timeEndCalc.tv_nsec / 1000000LL) -
                        ((uint64_t)timeStartCalc.tv_sec * 1000LL + (uint64_t)timeStartCalc.tv_nsec / 1000000LL);

                dFrameRate = 1000.0/(double)nCompTimeMS;
            }
            // if no impact site prediction found
            else
            {
                // show left sub-image (COLOR) WITHOUT impact site prediction to MIT-LAR operator

                if(!bUseRawImage)
                {
                    if(bShowFrameRate)
                    {
                        // add current frame rate to live view

                        dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                        sFrameRateOverlay = std::to_string(dFrameRateRounded);

                        sFrameRateOverlay.erase(2,10);

                        sFrameRateOverlay.append(" fps");

                        cv::putText(pLeftImgRectRAW, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    if(bShowInfo)
                    {
                        cv::putText(pLeftImgRectRAW, "RECTIFIED IMAGE (PRED. ACTIVE)", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    // display live rectified left image (COLOR)
                    cv::imshow("Live View", pLeftImgRectRAW);

                    // show color-coded disparity image
                    cv::minMaxIdx(pDispMapWLS, &min, &max);
                    cv::convertScaleAbs(pDispMapWLS, pScaledDispMap, 255/(max - min));
                    cv::applyColorMap(pScaledDispMap, pCM_disp, cv::COLORMAP_JET);

                    cv::imshow("Disparity Image", pCM_disp);
                }
                else
                {
                    // undistort RAW image
                    // cv::undistort(pLeftImg, pLeftImgUndist, pCameraMatrixLeftRAW, pDistortionCoeffsLeft, cv::noArray());

                    cv::remap(pLeftImg, pLeftImgUndist, pMapLeft1UNDIST, pMapLeft2UNDIST, cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValueRemap);

                    if(bShowFrameRate)
                    {
                        // add current frame rate to live view

                        dFrameRateRounded = (float)std::round(dFrameRate*100.0)/100.0;

                        sFrameRateOverlay = std::to_string(dFrameRateRounded);

                        sFrameRateOverlay.erase(2,10);

                        sFrameRateOverlay.append(" fps");

                        cv::putText(pLeftImgUndist, sFrameRateOverlay, cv::Point(10.0, 30.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    if(bShowInfo)
                    {
                        cv::putText(pLeftImgUndist, "UNDISTORTED IMAGE (PRED. ACTIVE)", cv::Point(10.0, 50.0), 1.0, 1.0, Scalar(255,255,255), 1, LINE_AA);
                    }

                    // display live rectified left image (COLOR)
                    cv::imshow("Live View", pLeftImgUndist);

                    // show color-coded disparity image
                    cv::minMaxIdx(pDispMapWLS, &min, &max);
                    cv::convertScaleAbs(pDispMapWLS, pScaledDispMap, 255/(max - min));
                    cv::applyColorMap(pScaledDispMap, pCM_disp, cv::COLORMAP_JET);

                    cv::imshow("Disparity Image", pCM_disp);
                }

                clock_gettime(CLOCK_MONOTONIC, &timeEndCalc);

                // calculate computation time
                nCompTimeMS = ((uint64_t)timeEndCalc.tv_sec * 1000LL + (uint64_t)timeEndCalc.tv_nsec / 1000000LL) -
                        ((uint64_t)timeStartCalc.tv_sec * 1000LL + (uint64_t)timeStartCalc.tv_nsec / 1000000LL);

                dFrameRate = 1000.0/(double)nCompTimeMS;
            }

            // reset impact site prediction for next frame
            P_imp_para.z = 5000.0;
        }

        nKey = cv::waitKey(1);
    }

    cv::destroyAllWindows();

    return 0;
}
