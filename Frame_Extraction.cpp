// Creator: Jacob F. Fast
// 2021
//
// Frame_Extraction.cpp
//
// Read frame sequence and extract desired set of frames. Store set for further processing.

#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

//OpenCV
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

// toggle for random frame selection
bool bRandomFrames = false;

int nNumberOfSkippedFrames = 0;

// FUNCTIONS

// save/show image
void saveOrShowImage(std::vector<Mat> &vImages,std::string sFolderName, std::string sFileName, bool bShow)
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
                std::cout << "One or more folders for frame storage already exist.\n"
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

    // read frame sequence
    std::cout << "Reading frame sequence..." << std::endl;

    // read frames from frame sequence stored in folder "FrameSequence"

    // declare vector for storage of extracted frames
    std::vector<cv::Mat> vExtractedFrames;

    cv::VideoCapture vidObj(sProjectPath + "/" + "FrameSequence" + "/" + "Os7-S1 Camera.mp4");

    // end program if no frame sequence found
    if(!vidObj.isOpened())
    {
        std::cout << "No frame sequence could be found! \n"
                     "Please add frame sequence to folder \"CalibrationSequence\" in project directory and re-run program. \n"
                     "Hit key to end program execution..." << std::endl;

        // wait for user input
        cv::waitKey(0);

        // stop program execution
        return 0;
    }

    // declarations for current frame
    cv::Mat pCurrentFrame;

    // number of frames to be read from sequence
    int nFrameNumber = 60;

    // number of read frames
    int nFrameCounter = 0;

    // index of current frame (reset after each extracted frame)
    int nCurrentFrameIdx = 0;

    // absolute index of current frame in frame sequence
    int nAbsoluteFrameIdx = 0;

    // query number of frames to be extracted from sequence

    std::cout << "Please enter number of frames to be extracted!" << std::endl;

    std::cin >> nFrameNumber;

    // ask whether random frame extraction desired

    std::cout << "Random frame extraction desired? 1: Yes, 0: No" << std::endl;

    std::cin >> bRandomFrames;

    if(bRandomFrames == 0)
    {
        std::cout << "Extracting frames chronologically starting at first frame of sequence..." << std::endl;

        // extract single frames from frame sequence in vector, starting at first frame and taking subsequent frames in order of recording
        while(vidObj.read(pCurrentFrame))
        {
            if(nFrameCounter >= nFrameNumber)
            {
                break;
            }

            vExtractedFrames.push_back(pCurrentFrame.clone());
            nFrameCounter++;
        }

    }
    // if random frame extraction desired
    else
    {
        std::cout << "Extracting frames randomly..." << std::endl;

        // extract single frames from frame sequence in vector, random frame index selection

        // seed random number generator
        std::srand(std::time(nullptr));

        // try to read next frame from sequence
        while(vidObj.read(pCurrentFrame))
        {
            // increment absolute index of current frame in frame sequence
            nAbsoluteFrameIdx++;

            // if enough frames extracted
            if(nFrameCounter >= nFrameNumber)
            {
                break;
            }

            // get random number of frames to be skipped (between 1 and 30)
            nNumberOfSkippedFrames = (std::rand() % 10)*3 + 1;

            // if more frames must be skipped
            if(nCurrentFrameIdx < nNumberOfSkippedFrames)
            {
                nCurrentFrameIdx++;
                continue;
            }

            vExtractedFrames.push_back(pCurrentFrame.clone());
            nFrameCounter++;

            std::cout << "Absolute index of randomly extracted frame no. " << nFrameCounter << ": " << nAbsoluteFrameIdx << std::endl;

            nCurrentFrameIdx = 0;
        }
    }

    vidObj.release();

    // save extracted frames in folder "ExtractedFrames"

    std::cout << "Saving extracted frames..." << std::endl;

    saveOrShowImage(vExtractedFrames, "ExtractedFrames", "frame", false);

    std::cout << "Frames saved! Hit key to end program execution...";

    // wait for user confirmation
    cv::waitKey(0);

    return 0;
}
