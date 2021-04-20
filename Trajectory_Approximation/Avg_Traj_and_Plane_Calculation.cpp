// Jacob F. Fast
//
// 2020/2021
//
// Avg_Traj_and_Plane_Calculation.cpp
// Read file containing triangulated (spatial) droplet positions and associated time stamps.
// Identify global linear (DEPRECATED MODEL) and parabolic (STANDARD MODEL) trajectory approximations based on this set of droplet positions. Calculate trajectory fit plane PI. Store results in YML file. Store PI and trajectory models as point cloud files (PLY) for visualization.

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

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

// GLOBAL VARIABLES

std::string sProjectPath;
std::string sBasePath;
std::string sBasePath_extended;

// float fVisibleDistance;

int nMaxNumOfSamplingPoints = 0;

// number of sampling points taken from each individual experiment
int nSamplingPoints = 0;

// global number of sampling points taken from all experiments
int nGlobalSamplingPoints = 0;

int nNumberOfSkippedLines = 0;

int nPointCounter = 0;

std::vector<int> vNumberIndividualSamplingPoints(10, 0);

// declaration of vector of coordinates of current droplet position
cv::Vec4f vCurrentDropletParameters;

// flag for cone section identification method (not used)
bool b2DMethod = false;

// flag for parabola identification using projections of sampling points onto trajectory fit plane (not used)
bool bProjectedPoints = false;

// flag for first time stamp identification (DO NOT CHANGE)
bool bTimeStampsStart = false;

// flag for status of sample point set reduction (DO NOT CHANGE)
bool bReduced = false;

// variable declaration for average geometric distance between sampling points and linear fit function in mm (DEPRECATED)
double avg_dist_lin = 0.0;

// variable declaration for average geometric distance between sampling points and parabolical fit function in mm (STANDARD)
double avg_dist_para = 0.0;

// variable for average geometric distance between sampling points and trajectory fit plane PI
double avg_dist_PI = 0.0;

// FUNCTIONS

// calculate average geometric distance between sampling points and linear fit function (DEPRECATED MODEL)
double calculateAverageGeometricDistanceLin(cv::Point3f &suppVec, cv::Point3f &dirVec, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // temporary helper point
    cv::Point3f TempTriangulatedPoint = cv::Point3f(0.0,0.0,0.0);

    std::cout << "vTriangulatedPoints.size(): " << vTriangulatedPoints.size() << std::endl;

    std::string filename = "DistancesToLinearFit.csv";

    std::ofstream csvfile_lin;

    csvfile_lin.open(filename);

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

        // write current distance to CSV file
        csvfile_lin << cv::norm((TempTriangulatedPoint - suppVec).cross(dirVec))/cv::norm(dirVec) << "\n";
    }

    // calculate average geometric distance in mm
    avg_dist_lin /= (double)vTriangulatedPoints.size();

    // output average geometric distance between sampling points and linear fit function
    std::cout << "Average distance from sampling points to linear fit in mm: " << avg_dist_lin << std::endl;

    // write average distance value to CSV file
    csvfile_lin << "\n";
    csvfile_lin << "AVERAGE" << "\n";
    csvfile_lin << avg_dist_lin << "\n";

    csvfile_lin.close();

    // return average geometric distance between sampling points and linear fit function
    return avg_dist_lin;
}

// calculate average geometric distance between sampling points and parabolic fit function (STANDARD MODEL)
double calculateAverageGeometricDistancePara(cv::Mat &vecF, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // reset average geometric distance between sampling points and parabolic fit function
    avg_dist_para = 0.0;

    // variable for squared distance between sampling point and parabolic trajectory model
    double dist_para_squared = 0.0;

    // variable for number of found distance values
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

    std::string filename = "DistancesToParabolicFit.csv";

    std::ofstream csvfile_para;

    csvfile_para.open(filename);

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
        else    // no physically plausible root found
        {
            continue;
        }

        std::cout << "t_starred in ms: " << t_starred << std::endl;

        // calculate shortest distance between current point and parabolical trajectory model
        dist_para_squared = (double)(std::pow(vecF.at<float>(2,0) + vecF.at<float>(1,0)*t_starred+vecF.at<float>(0,0)*std::pow(t_starred,2) - TempTriangulatedPoint.x, 2) + std::pow(vecF.at<float>(5,0) + vecF.at<float>(4,0)*t_starred+vecF.at<float>(3,0)*std::pow(t_starred,2) - TempTriangulatedPoint.y,2) + std::pow(vecF.at<float>(8,0) + vecF.at<float>(7,0)*t_starred + vecF.at<float>(6,0)*std::pow(t_starred,2) - TempTriangulatedPoint.z,2));

        // add geometric distance of current point to fit line
        avg_dist_para += std::sqrt(dist_para_squared);

        std::cout << "Current geometric distance from sampling point to parabolic fit in mm: " << std::sqrt(dist_para_squared) << std::endl;

        // write current distance to CSV file
        csvfile_para << std::sqrt(dist_para_squared) << "\n";
    }

    // calculate average geometric distance
    avg_dist_para /= (double)nDistanceValues;

    // output average geometric distance between sampling points and parabolic fit function
    std::cout << "Average distance from sampling points to parabolic fit in mm: " << avg_dist_para << std::endl;

    // write average distance value to CSV file
    csvfile_para << "\n";
    csvfile_para << "AVERAGE" << "\n";
    csvfile_para << avg_dist_para << "\n";

    csvfile_para.close();

    // return average geometric distance between sampling points and parabolic fit function
    return avg_dist_para;
}

// calculate average geometric distance between sampling points and fit plane PI
double calculateAverageGeometricDistancePI(cv::Vec3f &vecN, double d, std::vector<cv::Vec4f> vTriangulatedPoints)
{
    // variable for current geometric distance between sampling points and fit plane PI
    double curr_dist_PI = 0.0;

    // temporary helper point
    cv::Point3f TempTriangulatedPoint = cv::Point3f(0.0,0.0,0.0);

    std::string filename = "DistancesToFitPlane.csv";

    std::ofstream csvfile_plane;

    csvfile_plane.open(filename);

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

        // write current distance to CSV file
        csvfile_plane << curr_dist_PI << "\n";
    }

    // calculate average orthogonal distance
    avg_dist_PI /= (double)vTriangulatedPoints.size();

    // output average geometric distance between sampling points and fit plane PI
    std::cout << "Average orthogonal distance from sampling point to fit plane PI in mm: " << avg_dist_PI << std::endl;

    // write average distance value to CSV file
    csvfile_plane << "\n";
    csvfile_plane << "AVERAGE" << "\n";
    csvfile_plane << avg_dist_PI << "\n";

    csvfile_plane.close();

    // return average geometric distance between sampling points and fit plane PI
    return avg_dist_PI;
}

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


// MAIN PROGRAM

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    cv::ocl::setUseOpenCL(true);

    // _____________________________________read list of triangulated droplet positions and associated time stamps from file_____________________________________

    // read current project path
    std::string pTempPath[256] = {argv[0]};
    std::string sExecPath;
    sExecPath = *pTempPath;

    // identify folder path
    int iPosOfChar = sExecPath.find_last_of("/");

    // extract project path from string

    sProjectPath = sExecPath.substr(0, iPosOfChar);

    // go back one level
    iPosOfChar = sProjectPath.find_last_of("/");
    sBasePath = sProjectPath.substr(0, iPosOfChar);

    std::cout  <<  sBasePath  <<  std::endl;

//    // set visibility limit of stereo laryngoscope in mm (NOT USED)
//    fVisibleDistance = 50.0;

    // set number of sampling points to be used for further calculations (USE ALL)
    nMaxNumOfSamplingPoints = 100000;

    // string for (x,y,z) coordinates
    std::string sLine;

    // string for time stamps
    std::string sLineTerm;

    // file reader for (x,y,z) coordinates in PLY files
    std::ifstream fileread;

    // file reader for time stamps in terminal output files
    std::ifstream fileread_terminal;

    // declaration of (4 x nSamplingPoints) matrix of sampling point parameters (x,y,z,t)
    std::vector<cv::Vec4f> vSamplingPoints;
    vSamplingPoints.resize(0);

    // open all files "triangulatedCentroids0.ply" and read all available sampling points from each file
    for (int i=1; i<11; i++)
    {
        std::cout << "i= " << i << std::endl;

        // common file path for (x,y,z) coordinates AND time stamps
        // adapt path to current experiment of interest
        sBasePath_extended = sBasePath + "/" + "Trajectory_Identification"  + "/" + "01_Trajectory_Results" + "/" +
                "02_FiberSystem" + "/" + "03_High" + "/" + "02_45deg"  + "/" + std::to_string(i);

        // open PLY file with (x,y,z) coordinates
        fileread.open(sBasePath_extended + "/" +
                       "Droplet Centroid Positions" + "/" + "triangulatedCentroids0.ply");

        // reset number of found sampling points of each droplet shooting experiment to zero
        nSamplingPoints = 0;

        if (fileread.is_open())
        {
            // read lines from file
            while(std::getline(fileread, sLine))
            {
                // std::cout << sLine << std::endl;

                // replace decimal separator (LOCALE-DEPENDENT!)
                std::replace(sLine.begin(), sLine.end(), '.', ',');

                // extract x, y, z coordinates from line
                std::istringstream iterator(sLine);
                std::vector<std::string> results((std::istream_iterator<std::string>(iterator)), std::istream_iterator<std::string>());

                if(results[0] != "ply" && results[0] != "format" && results[0] != "element" && results[0] != "property" && results[0] != "end_header")
                {
                    // store x, y, z coordinates in vector
                    vCurrentDropletParameters[0] = std::stof(results[0]);   // x coordinate value
                    vCurrentDropletParameters[1] = std::stof(results[1]);   // y coordinate value
                    vCurrentDropletParameters[2] = std::stof(results[2]);   // z coordinate value
                }
                else
                {
                    continue;
                }

                nSamplingPoints++;
                vNumberIndividualSamplingPoints[i-1]++;

                // add correct time stamp value from terminal output file

                // open plain file with time stamps
                fileread_terminal.open(sBasePath_extended + "/" + "Terminal_Output");

                if (fileread_terminal.is_open())
                {
                    // reset number of skipped lines
                    nNumberOfSkippedLines = 0;

                    // reset Boolean variable
                    bTimeStampsStart = false;

                    // read all lines
                    while(std::getline(fileread_terminal, sLineTerm))
                    {
                        // std::cout << sLineTerm << std::endl;

                        // if current line empty: continue
                        if(sLineTerm.empty())
                        {
                            continue;
                        }

                        // replace decimal separator (LOCALE-DEPENDENT!)
                        std::replace(sLineTerm.begin(), sLineTerm.end(), '.', ',');

                        std::istringstream iteratorTerm(sLineTerm);
                        std::vector<std::string> resultsTerm((std::istream_iterator<std::string>(iteratorTerm)), std::istream_iterator<std::string>());

                        // first time stamps value will be read at next iteration
                        if(resultsTerm[0] == "Valid" && resultsTerm[1] == "time")
                        {
                            bTimeStampsStart = true;
                            std::cout << "Line with first time stamp found!" << std::endl;
                            continue;
                        }

                        // first line with time stamp found
                        if(bTimeStampsStart == true)
                        {
                            // jump to correct line
                            if(nNumberOfSkippedLines < nSamplingPoints - 1)
                            {
                                nNumberOfSkippedLines++;

                                // std::cout << "Line skipped!" << std::endl;

                                continue;
                            }
                            else
                            {
                                vCurrentDropletParameters[3] = std::stof(resultsTerm[0]);   // add time stamp value to vector

                                vSamplingPoints.push_back(vCurrentDropletParameters);

                                std::cout << "vCurrentDropletParameters: " << vCurrentDropletParameters << std::endl;

                                break;
                            }
                        }
                    }
                    fileread_terminal.close();
                }
                else
                {
                   std::cout << "File not found!" << std::endl;
                }

                //            // check if z coordinate value of current point is in visibility range (set by fVisibleDistance)
                //            if(vCurrentDropletParameters[2] < fVisibleDistance)
                //            {
                //                vSamplingPoints.push_back(vCurrentDropletParameters);
                //                nSamplingPoints++;

                //                nGlobalSamplingPoints++;
                //            }
            }
            fileread.close();
        }
        else
        {
            std::cout << "File not found!" << std::endl;
        }
    }

    // show identified numbers of sampling points for each experiment
    std::cout << "Number of available sampling points for each individual droplet shooting event: " << std::endl;

    for(int i=0;i<10;i++)
    {
        std::cout << vNumberIndividualSamplingPoints[i] << std::endl;
    }

    // ________________________store reduced vector with first nMaxNumOfSamplingPoints sampling points from each experiment__________________________________

    // declaration of vector of coordinates of current droplet position
    std::vector<cv::Vec4f> vSamplingPointsReduced;

    // declaration of vector of coordinates of current droplet position
    std::vector<cv::Vec4f> vSamplingPointsReducedProjected;

    // iterator for current sampling point index
    int nSamplingPointIterator = 0;

    // lowest number of available sampling points per element
    int nMinNumOfSamplingPoints = *std::min_element(std::begin(vNumberIndividualSamplingPoints), std::end(vNumberIndividualSamplingPoints));

    std::cout << "nMinNumOfSamplingPoints: " << nMinNumOfSamplingPoints << std::endl;

    // if nMaxNumOfSamplingPoints sampling points available in each experiment: reduce vector of global sampling point set
    if(nMaxNumOfSamplingPoints < nMinNumOfSamplingPoints)
    {
        // for each individual experiment (total number of experiments here: 10)
        for (int i=0; i<10; i++)
        {
            // add first nMaxNumOfSamplingPoints sampling points of each individual experiment to vector vSamplingPointsReduced
            for (int j=0; j<nMaxNumOfSamplingPoints; j++)
            {
                vCurrentDropletParameters[0] = vSamplingPoints[nSamplingPointIterator + j][0];
                vCurrentDropletParameters[1] = vSamplingPoints[nSamplingPointIterator + j][1];
                vCurrentDropletParameters[2] = vSamplingPoints[nSamplingPointIterator + j][2];
                vCurrentDropletParameters[3] = vSamplingPoints[nSamplingPointIterator + j][3];

                vSamplingPointsReduced.push_back(vCurrentDropletParameters);
            }
            // jump to first sampling point of next experiment
            nSamplingPointIterator += vNumberIndividualSamplingPoints[i];
        }
        bReduced = true;
    }
    // if at least one experiment contains less than nMaxNumOfSamplingPoints sampling points
    else
    {
        // no reduction of global sampling point set
        vSamplingPointsReduced = vSamplingPoints;

        bReduced = false;
    }

    // __________________________________________calculate global linear trajectory approximation (DEPRECATED TRAJECTORY MODEL)_____________________________________________________________

    // non-shifted sampling points must be stored as 3D points
    std::vector<cv::Point3f> vSamplingPointsLin;

    vSamplingPointsLin.resize(0);

    std::cout << "vSamplingPointsReduced.size(): " << vSamplingPointsReduced.size() << std::endl;

    // temporary helper point
    cv::Point3f TempSamplingPoint = cv::Point3f(0.0,0.0,0.0);

    // read all (x,y,z) coordinates from vSamplingPointsReduced
    for (unsigned int i=0; i<vSamplingPointsReduced.size(); i++)
    {
        TempSamplingPoint.x = vSamplingPointsReduced[i][0];
        TempSamplingPoint.y = vSamplingPointsReduced[i][1];
        TempSamplingPoint.z = vSamplingPointsReduced[i][2];

        vSamplingPointsLin.push_back(TempSamplingPoint);

    }

    std::cout << "Vector of sampling points for global linear trajectory approximation: " << vSamplingPointsLin << std::endl;

    // parameter set of global linear trajectory approximation
    cv::Vec6f T_lin;

    cv::Point3f suppVecLin = Point3f(0.0,0.0,0.0);
    cv::Point3f dirVecLin = Point3f (0.0,0.0,0.0);

    // find global fit line parameters using target metric "CV_DIST_L2" (Euclidean distance)
    cv::fitLine(vSamplingPointsLin, T_lin, CV_DIST_L2, 0, 0.01, 0.01);

    // direction vector of global fit line
    dirVecLin.x = T_lin[0];
    dirVecLin.y = T_lin[1];
    dirVecLin.z = T_lin[2];

    // support vector of global fit line
    suppVecLin.x =  T_lin[3];
    suppVecLin.y =  T_lin[4];
    suppVecLin.z =  T_lin[5];

    // calculate and store distances from non-reduced set of sampling points to global linear trajectory approximation
    avg_dist_lin = calculateAverageGeometricDistanceLin(suppVecLin, dirVecLin, vSamplingPoints);

    // _____________________________________store global linear trajectory approximation in YML file_____________________________________

    // initialize file storage object
    FileStorage trajInfos;

    std::string filename = "TrajInfos.yml";

    trajInfos.open(filename, FileStorage::WRITE);

    // get current system time
    time_t rawtime;
    time(&rawtime);

    // store current system time
    trajInfos << "Date" << asctime(localtime(&rawtime));

    // store global linear approximation
    trajInfos << "suppVecLin" << suppVecLin;
    trajInfos << "dirVecLin" << dirVecLin;

    // store average distance from line to non-reduced set of sampling points
    trajInfos << "avg_dist_lin" << avg_dist_lin;

    // _____________________________________draw global linear trajectory approximation_____________________________________

    // declare vectors for fit functions and fit plane
    std::vector<cv::Mat> vTrajectory, vPlanes;

    vTrajectory.resize(2);
    vPlanes.resize(1);

    // declarations
    double fZMin = -10.0;   // minimum z coordinate value in mm
    double fZMax = 200.0;   // maximum z coordinate value in mm
    double fFactor = 0.0;   // factor for point density along trajectory
    int nPosition = 0;      // position iterator

    // calibration performed in mm
    int nCounter = -10000;

    // temporary point
    cv::Vec3d tempPoint;

    // initialize z coordinate
    tempPoint[2] = -20.0;

    vTrajectory[0] = cv::Mat::zeros(5000, 5000, CV_64FC3);

    // inject average support and direction vectors to save linear trajectory approximation

    //    suppVecLin.x = 3.5652444362640381e+00;
    //    suppVecLin.y = -7.9501438140869141e+00;
    //    suppVecLin.z = 5.6179321289062500e+01;

    //    dirVecLin.x = -4.2035675048828125e-01;
    //    dirVecLin.y = -1.3615272939205170e-01;
    //    dirVecLin.z = 8.9708566665649414e-01;

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

    // _________________________________________________calculate global fit plane_____________________________________________________________

    // declare vector of shifted sampling points (x,y,z)
    std::vector<cv::Vec3f> vSamplingPointsShifted(vSamplingPointsReduced.size());

    // declare centroid C of set of sampling points
    cv::Point3f C;

    // calculate centroid of set of sampling points
    C.x = (float)cv::mean(vSamplingPointsReduced)[0];  // mean of all x coordinates
    C.y = (float)cv::mean(vSamplingPointsReduced)[1];  // mean of all y coordinates
    C.z = (float)cv::mean(vSamplingPointsReduced)[2];  // mean of all z coordinates

    std::cout << "Centroid calculated." << std::endl;
    std::cout << "C.x: " << C.x << std::endl;
    std::cout << "C.y: " << C.y << std::endl;
    std::cout << "C.z: " << C.z << std::endl;

    // shift sampling points to origin of coordinate frame (CF)_C of left virtual camera

    // for each sampling point of set of avaiable sampling points from 10 droplet shooting experiments
    for (unsigned int i=0; i<vSamplingPointsReduced.size(); i++)
    {
        // for each set of coordinates (x,y,z)
        vSamplingPointsShifted[i][0] = vSamplingPointsReduced[i][0] - C.x;
        vSamplingPointsShifted[i][1] = vSamplingPointsReduced[i][1] - C.y;
        vSamplingPointsShifted[i][2] = vSamplingPointsReduced[i][2] - C.z;
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

    cv::Mat M = cv::Mat::zeros(3, vSamplingPointsReduced.size(), CV_32F);

    // for all rows of M (iterate over sampling point coordinates (x,y,z))
    for (int i=0; i<3 ;i++)
    {
        // for all columns of M (iterate over global set of sampling points)
        for (unsigned int j=0; j<vSamplingPointsReduced.size(); j++)
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

    // calculate scalar d of Hesse normal form
    double d;
    d = C.dot(n_PI_0);

    std::cout << "d = " << d << std::endl;

    std::cout << "Fit plane calculated. " << std::endl;

    // calculate mean distance from all sampling points to fit plane
    avg_dist_PI = calculateAverageGeometricDistancePI(n_PI_0, d, vSamplingPoints);


    // _____________________________________store global fit plane parameters in YML file_____________________________________

    trajInfos << "C" << C;
    trajInfos << "n_PI_0" << n_PI_0;
    trajInfos << "d" << d;

    trajInfos << "avg_dist_PI" << avg_dist_PI;


    // _____________________________________draw and save global fit plane____________________________________________________________


    // inject average plane parameters, if desired

    //    d = 3.8055939674377441e+00;

    //    n_PI_0[0] = 1.0853711515665054e-01;
    //    n_PI_0[1] = 9.7403359413146973e-01;
    //    n_PI_0[2] = 1.9869123399257660e-01;

    // reset values
    nPosition = 0;      // frame column iterator

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

    // color for fit plane
    std::vector<cv::Vec3i> vPlaneColors(1);

    vPlaneColors[0][0] = 255;
    vPlaneColors[0][1] = 255;
    vPlaneColors[0][2] = 0;

    // save fit plane PI
    savePointCloud("Global Fit Plane", "FitPlane", vPlaneColors, vPlanes);

    std::cout << "Global fit plane stored as point cloud!" << std::endl;

    // __________________________________________calculate global parabolical trajectory approximation (STANDARD MODEL)_____________________________________

    // perform identification of conic section in global fit plane PI (METHOD NOT USED, MAY CONTAIN ERRORS)
    if(b2DMethod == true)
    {
        // declare vector for CENTROID-FREE sampling points projected onto plane PI (3D points)
        std::vector<cv::Vec3f> vSamplingPointsShiftedProjected;

        // orthogonal distance between current sampling point and plane PI
        float fCurrOrthDistPI = 0.0;

        vSamplingPointsShiftedProjected.resize(vSamplingPointsReduced.size());

        // project all available, CENTROID-FREE sampling points (after centroid subtraction) onto plane PI

        std::cout << "Projected, CENTROID-FREE sampling points in (x, y, z) coordinates:" << std::endl;

        for(unsigned int i=0; i<vSamplingPointsReduced.size(); ++i)
        {
            // calculate orthogonal distance between current sampling point and plane PI
            fCurrOrthDistPI =  vSamplingPointsShifted[i][0] * n_PI_0[0] + vSamplingPointsShifted[i][1] * n_PI_0[1] + vSamplingPointsShifted[i][2] * n_PI_0[2];

            // calculate x coordinate of projected sampling point
            vSamplingPointsShiftedProjected[i][0] = vSamplingPointsShifted[i][0]-fCurrOrthDistPI * n_PI_0[0];

            // y coordinate of projected sampling point
            vSamplingPointsShiftedProjected[i][1] = vSamplingPointsShifted[i][1]-fCurrOrthDistPI * n_PI_0[1];

            // z coordinate of projected sampling point
            vSamplingPointsShiftedProjected[i][2] = vSamplingPointsShifted[i][2]-fCurrOrthDistPI * n_PI_0[2];

            std::cout << vSamplingPointsShiftedProjected[i] << std::endl;
        }

        // define new 2D coordinate system (x', y') in plane PI

        // unit vector u_y' parallel to global fit line direction vector
        cv::Vec3f vUnitVectorYPrime;

        vUnitVectorYPrime[0] = dirVecLin.x;
        vUnitVectorYPrime[1] = dirVecLin.y;
        vUnitVectorYPrime[2] = dirVecLin.z;

        std::cout << "Unit vector in direction of y': " << vUnitVectorYPrime << std::endl;

        // vector u_x' = u_y' x n_PI_0
        cv::Vec3f vVectorXPrime;

        vVectorXPrime = vUnitVectorYPrime.cross(n_PI_0);

        // transform into unit vector
        cv::Vec3f vUnitVectorXPrime;

        std::cout << "cv::norm(vVectorXPrime): " << cv::norm(vVectorXPrime) << std::endl;

        // make sure vector is unitary
        // vUnitVectorXPrime = vVectorXPrime / cv::norm(vVectorXPrime);
        vUnitVectorXPrime = vVectorXPrime;

        std::cout << "Unit vector in direction of x': " << vUnitVectorXPrime << std::endl;

        // define 3D rotation matrix between (x, y, z) and (x', y') coordinate system
        cv::Mat RotationMatrixXXPrime = cv::Mat::zeros(3, 3, CV_32F);

        // unit vector u_x expressed in (x', y') coordinates
        RotationMatrixXXPrime.at<float>(0,0) = vUnitVectorXPrime[0];
        RotationMatrixXXPrime.at<float>(1,0) = vUnitVectorYPrime[0];
        RotationMatrixXXPrime.at<float>(2,0) = n_PI_0[0];

        // unit vector u_y expressed in (x', y') coordinates
        RotationMatrixXXPrime.at<float>(0,1) = vUnitVectorXPrime[1];
        RotationMatrixXXPrime.at<float>(1,1) = vUnitVectorYPrime[1];
        RotationMatrixXXPrime.at<float>(2,1) = n_PI_0[1];

        // unit vector u_z expressed in (x', y') coordinates
        RotationMatrixXXPrime.at<float>(0,2) = vUnitVectorXPrime[2];
        RotationMatrixXXPrime.at<float>(1,2) = vUnitVectorYPrime[2];
        RotationMatrixXXPrime.at<float>(2,2) = n_PI_0[2];

        std::cout << "RotationMatrixXXPrime: " << RotationMatrixXXPrime << std::endl;

        double fDetRotationMatrixXXPrime = cv::determinant(RotationMatrixXXPrime);

        std::cout << "Determinant of RotationMatrixXXPrime (should be 1 for rotation matrix): " << fDetRotationMatrixXXPrime << std::endl;

        // vector of projected sampling points in (x', y') coordinates (2D points)
        std::vector<cv::Vec2f> vSamplingPointsShiftedProjected2D;
        vSamplingPointsShiftedProjected2D.resize(vSamplingPointsShiftedProjected.size());

        // transform coordinates of all projected points into 2D coordinate system (x', y')

        std::cout << "Sampling points in (x', y') coordinates:" << std::endl;

        for(unsigned int i=0; i<vSamplingPointsShiftedProjected.size(); ++i)
        {
            // x' coordinate
            vSamplingPointsShiftedProjected2D[i][0] = RotationMatrixXXPrime.at<float>(0,0) * vSamplingPointsShiftedProjected[i][0] +
                    RotationMatrixXXPrime.at<float>(0,1) * vSamplingPointsShiftedProjected[i][1] + RotationMatrixXXPrime.at<float>(0,2) * vSamplingPointsShiftedProjected[i][2];

            // y' coordinate
            vSamplingPointsShiftedProjected2D[i][1] = RotationMatrixXXPrime.at<float>(1,0) * vSamplingPointsShiftedProjected[i][0] +
                    RotationMatrixXXPrime.at<float>(1,1) * vSamplingPointsShiftedProjected[i][1] + RotationMatrixXXPrime.at<float>(1,2) * vSamplingPointsShiftedProjected[i][2];

            std::cout << vSamplingPointsShiftedProjected2D[i] << std::endl;
        }

        // find parameters of conic section which fits projected sampling points in (x', y') coordinate system

        // declare vector K of five unknown parameters (a, b, c, d, e) of best-fit conic section, set sixth parameter f to 1.0
        cv::Mat K = cv::Mat::zeros(5, 1, CV_32F);

        // construct matrix P_PRIME of known sampling points in (x', y') coordinates
        cv::Mat P_PRIME = cv::Mat::zeros(vSamplingPointsShiftedProjected2D.size(), 5, CV_32F);

        // fill matrix P_PRIME of known sampling points in (x', y') coordinates
        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2D.size(); ++i)
        {
                // x'²
                P_PRIME.at<float>(i,0) = vSamplingPointsShiftedProjected2D[i][0] * vSamplingPointsShiftedProjected2D[i][0];
                // x'y'
                P_PRIME.at<float>(i,1) = vSamplingPointsShiftedProjected2D[i][0] * vSamplingPointsShiftedProjected2D[i][1];
                // y'²
                P_PRIME.at<float>(i,2) = vSamplingPointsShiftedProjected2D[i][1] * vSamplingPointsShiftedProjected2D[i][1];
                // x'
                P_PRIME.at<float>(i,3) = vSamplingPointsShiftedProjected2D[i][0];
                // y'
                P_PRIME.at<float>(i,4) = vSamplingPointsShiftedProjected2D[i][1];
        }

        std::cout << "Matrix P_PRIME: " << std::endl << P_PRIME << std::endl;

        // declare vector VEC_RIGHT_SIDE with sixth parameter f held constant at 1.0
        cv::Mat VEC_RIGHT_SIDE = cv::Mat(vSamplingPointsShiftedProjected2D.size(), 1, CV_32F, -1.0);

        // solve over-determined system P_PRIME * K = VEC_ZERO using SVD
        cv::solve(P_PRIME, VEC_RIGHT_SIDE, K, DECOMP_SVD);

        std::cout << "Identified parameter vector of conic section: " << std::endl << K << std::endl;

        // construct (2x2) matrix A
        cv::Mat A = cv::Mat::zeros(2,2,CV_32FC1);

        A.at<float>(0,0) = K.at<float>(0,0);
        A.at<float>(0,1) = K.at<float>(0,1)/2.0;
        A.at<float>(1,0) = A.at<float>(0,1);
        A.at<float>(1,1) = K.at<float>(0,2);

        std::cout << "Matrix A: " << A << std::endl;

        std::cout << "Determinant of matrix A (should be close to 0 for parabolic result): " << cv::determinant(A) << std::endl;

        // identify eigenvalues and eigenvectors of matrix A
        // as conic section is close to parabolic shape, one eigenvector should be close to 0!
        std::vector<float> vEigenvalues;
        vEigenvalues.resize(2);

        // perform calculation
        cv::eigen(A, vEigenvalues);

        std::cout << "vEigenvalues[0]: " << vEigenvalues[0] << std::endl;
        std::cout << "vEigenvalues[1]: " << vEigenvalues[1] << std::endl;

        // store identified eigenvalues
        trajInfos << "vEigenvalues" << vEigenvalues;

        std::vector<cv::Vec2f> vEigenvectors;
        vEigenvectors.resize(2);

        // x' value of eigenvector to vEigenvalues[0]: vEigenvalues[0] - C
        vEigenvectors[0][0] = vEigenvalues[0] - A.at<float>(1,1);
        // y' value of eigenvector to vEigenvalues[0]: B/2
        vEigenvectors[0][1] = A.at<float>(0,1);

        std::cout << "cv::norm(vEigenvectors[0]) before normalization: " << cv::norm(vEigenvectors[0]) << std::endl;

        // normalize vector
        vEigenvectors[0][0] /= std::sqrt(std::pow(vEigenvalues[0] - A.at<float>(1,1), 2) + std::pow(A.at<float>(0,1), 2));
        vEigenvectors[0][1] /= std::sqrt(std::pow(vEigenvalues[0] - A.at<float>(1,1), 2) + std::pow(A.at<float>(0,1), 2));

        std::cout << "cv::norm(vEigenvectors[0]): " << cv::norm(vEigenvectors[0]) << std::endl;

        // x' value of eigenvector to vEigenvalues[1]: vEigenvalues[1] - C
        vEigenvectors[1][0] = vEigenvalues[1] - A.at<float>(1,1);
        // y' value of eigenvector to vEigenvalues[1]: B/2
        vEigenvectors[1][1] = A.at<float>(0,1);

        std::cout << "cv::norm(vEigenvectors[1]) before normalization: " << cv::norm(vEigenvectors[1]) << std::endl;

        // normalize vector
        vEigenvectors[1][0] /= std::sqrt(std::pow(vEigenvalues[1] - A.at<float>(1,1), 2) + std::pow(A.at<float>(0,1), 2));
        vEigenvectors[1][1] /= std::sqrt(std::pow(vEigenvalues[1] - A.at<float>(1,1), 2) + std::pow(A.at<float>(0,1), 2));

        std::cout << "cv::norm(vEigenvectors[1]): " << cv::norm(vEigenvectors[1]) << std::endl;

        // identify 2D rotation matrix to transform identified conic section into parabola without mixed term x'y' -> (xi, eta) coordinates
        cv::Mat RotationMatrix2D = cv::Mat::zeros(2,2,CV_32F);

        // coefficient of eta² should vanish -> set lambda_2 equal to the eigenvalue which is closer to zero

        // if vEigenvalues[0] closer to zero than vEigenvalues[1]
        if(std::abs(vEigenvalues[0]) < std::abs(vEigenvalues[1]))
        {
            // fill first column with eigenvector to vEigenvalues[1]
            RotationMatrix2D.at<float>(0,0) = vEigenvectors[1][0];
            RotationMatrix2D.at<float>(1,0) = vEigenvectors[1][1];

            // fill second column with eigenvector to vEigenvalues[0]
            RotationMatrix2D.at<float>(0,1) = vEigenvectors[0][0];
            RotationMatrix2D.at<float>(1,1) = vEigenvectors[0][1];
        }
        else
        {
            // fill first column with eigenvector to vEigenvalues[0]
            RotationMatrix2D.at<float>(0,0) = vEigenvectors[0][0];
            RotationMatrix2D.at<float>(1,0) = vEigenvectors[0][1];

            // fill second column with eigenvector to vEigenvalues[1]
            RotationMatrix2D.at<float>(0,1) = vEigenvectors[1][0];
            RotationMatrix2D.at<float>(1,1) = vEigenvectors[1][1];
        }

        // second column of rotation matrix from (xi, eta) to (x', y') coordinate system now contains eigenvector to eigenvalue which is closer to zero

        // check if right-handed coordinate system obtained after transformation of (x', y') by RotationMatrix2D
        // det(RotationMatrix2D) must be +1

        float fDetRotationMatrix2D = RotationMatrix2D.at<float>(0,0)*RotationMatrix2D.at<float>(1,1)-RotationMatrix2D.at<float>(1,0)*RotationMatrix2D.at<float>(0,1);

        std::cout << "Determinant of rotation matrix between (x', y') and (xi, eta): " << fDetRotationMatrix2D << std::endl;

        // invert direction of first column vector if det != +1
        if(fDetRotationMatrix2D < 0)
        {
            RotationMatrix2D.at<float>(0,0) = -RotationMatrix2D.at<float>(0,0);
            RotationMatrix2D.at<float>(1,0) = -RotationMatrix2D.at<float>(1,0);
            std::cout << "Rotation matrix now yields right-handed coordinate system (xi, eta)! " << std::endl;
        }

        std::cout << "RotationMatrix2D: " << RotationMatrix2D << std::endl;

        // store parameters of identified 2D rotation matrix
        trajInfos << "RotationMatrix2D" << RotationMatrix2D;

        std::cout << "Rotation angle between (x', y') and (xi, eta) coordinate system in degrees: " << std::acos(RotationMatrix2D.at<float>(0,0))* (180.0/3.14159) << std::endl;

        // vector of projected sampling points in (xi, eta) coordinates (2D points)
        std::vector<cv::Vec2f> vSamplingPointsShiftedProjected2DXi;
        vSamplingPointsShiftedProjected2DXi.resize(vSamplingPointsShiftedProjected.size());

        // transform coordinates of all projected points into 2D coordinate system (xi, eta)

        // coordinate transformation matrix (x, y) -> (xi, eta)
        cv::Mat RotationMatrix2DXPrimeXi = cv::Mat::zeros(2,2,CV_32F);

        cv::invert(RotationMatrix2D, RotationMatrix2DXPrimeXi);

        std::cout << "Sampling points in (xi, eta) coordinates:" << std::endl;

        for(unsigned int i=0; i<vSamplingPointsShiftedProjected.size(); ++i)
        {
            // xi coordinate
            vSamplingPointsShiftedProjected2DXi[i][0] = RotationMatrix2DXPrimeXi.at<float>(0,0) * vSamplingPointsShiftedProjected2D[i][0] +
                    RotationMatrix2DXPrimeXi.at<float>(0,1) * vSamplingPointsShiftedProjected2D[i][1];

            // eta coordinate
            vSamplingPointsShiftedProjected2DXi[i][1] = RotationMatrix2DXPrimeXi.at<float>(1,0) * vSamplingPointsShiftedProjected2D[i][0] +
                    RotationMatrix2DXPrimeXi.at<float>(1,1) * vSamplingPointsShiftedProjected2D[i][1];

            std::cout << vSamplingPointsShiftedProjected2DXi[i] << std::endl;
        }

        // identify parameters (k, l, m) of best-fit parabola eta = k xi² + l xi + m in rotated (xi, eta) coordinate system
        cv::Mat Parameters2DParabola = cv::Mat::zeros(3, 1, CV_32F);

        // if vEigenvalues[0] closer to zero than vEigenvalues[0]
        if(std::abs(vEigenvalues[0]) < std::abs(vEigenvalues[1]))
        {
            // coefficient of xi²
            Parameters2DParabola.at<float>(0,0) = -vEigenvalues[1] / (K.at<float>(0,3)*RotationMatrix2D.at<float>(0,1) + K.at<float>(0,4)*RotationMatrix2D.at<float>(1,1));
        }
        else
        {
            // coefficient of xi²
            Parameters2DParabola.at<float>(0,0) = -vEigenvalues[0] / (K.at<float>(0,3)*RotationMatrix2D.at<float>(0,1) + K.at<float>(0,4)*RotationMatrix2D.at<float>(1,1));
        }

        // coefficient of xi
        Parameters2DParabola.at<float>(1,0) = -(K.at<float>(0,3)*RotationMatrix2D.at<float>(0,0) + K.at<float>(0,4)*RotationMatrix2D.at<float>(1,0)) /
                (K.at<float>(0,3)*RotationMatrix2D.at<float>(0,1) + K.at<float>(0,4)*RotationMatrix2D.at<float>(1,1));
        // constant coefficient
        Parameters2DParabola.at<float>(2,0) = -1.0 / (K.at<float>(0,3)*RotationMatrix2D.at<float>(0,1) + K.at<float>(0,4)*RotationMatrix2D.at<float>(1,1));


        //        // construct overdetermined system of linear equations

        //        // matrix of xi values
        //        cv::Mat XI = cv::Mat::zeros(vSamplingPointsShiftedProjected2DXi.size(), 3, CV_32F);

        //        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2DXi.size(); ++i)
        //        {
        //            XI.at<float>(i,0) = std::pow(vSamplingPointsShiftedProjected2DXi[i][0], 2);
        //            XI.at<float>(i,1) = vSamplingPointsShiftedProjected2DXi[i][0];
        //            XI.at<float>(i,2) = 1.0;
        //        }

        //        // vector of eta values
        //        cv::Mat ETA = cv::Mat::zeros(vSamplingPointsShiftedProjected.size(), 1, CV_32F);
        //        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2DXi.size(); ++i)
        //        {
        //            ETA.at<float>(i,0) = vSamplingPointsShiftedProjected2DXi[i][1];
        //        }

        //        // solve over-determined system
        //        cv::solve(XI, ETA, Parameters2DParabola, DECOMP_SVD);

        std::cout << "Equation of identified parabola (no mixed terms): eta = " << Parameters2DParabola.at<float>(0,0) << " * xi_squared + " << Parameters2DParabola.at<float>(1,0)
                  << " * xi + " <<  Parameters2DParabola.at<float>(2,0) << std::endl;

        // store parameters of identified parabola in plane PI
        trajInfos << "Parameters2DParabola" << Parameters2DParabola;

        trajInfos.release();

        // ____________________________________________draw resulting parabola in 3D camera coordinate system______________________________________________

        // initialize structure for points on parabola
        vTrajectory[1] = cv::Mat::zeros(5000, 5000, CV_64FC3);

        // identify/set minimum xi coordinate of first sampling point set
        float fXiMin = vSamplingPointsShiftedProjected2DXi[0][0];

        // identify/set maximum xi coordinate of first sampling point set
        float fXiMax = vSamplingPointsShiftedProjected2DXi[vNumberIndividualSamplingPoints[0]-1][0];

        // ensure that fXiMin < fXiMax
        if(fXiMax < fXiMin)
        {
            float fBuffer = fXiMin;

            fXiMin = fXiMax;
            fXiMax = fBuffer;
        }

        // multiply both values by constant
        int nXiMinCounter = (int)(fXiMin*1000);
        int nXiMaxCounter = (int)(fXiMax*1000);

        std::cout << "nXiMinCounter: " << nXiMinCounter << std::endl;
        std::cout << "nXiMaxCounter: " << nXiMaxCounter << std::endl;

        // reset position counter
        nPosition = 0;

        cv::Vec3d tempPointXi, tempPointXPrime;

        cv::Mat RotationMatrixXPrimeX = cv::Mat::zeros(2,2,CV_32F);

        cv::invert(RotationMatrixXXPrime, RotationMatrixXPrimeX);

//        // swipe xi coordinate value range
//        for(int i=nXiMinCounter; i<nXiMaxCounter; ++i)
//        {
//            // set xi coordinate
//            tempPointXi[0] = (float)i/1000.0;

//            // calculate eta coordinate
//            tempPointXi[1] = Parameters2DParabola.at<float>(0,0) * tempPointXi[0] * tempPointXi[0] + Parameters2DParabola.at<float>(1,0) * tempPointXi[0] +
//                    Parameters2DParabola.at<float>(2,0);

//            // set third coordinate to zero
//            tempPointXi[2] = 0.0;

//            // represent current point in coordinate system (x', y')

//            // calculate x' coordinate from (xi, eta) coordinates
//            tempPointXPrime[0] = RotationMatrix2D.at<float>(0,0) * tempPointXi[0] + RotationMatrix2D.at<float>(0,1) * tempPointXi[1];

//            // calculate y' coordinate from (xi, eta) coordinates
//            tempPointXPrime[1] = RotationMatrix2D.at<float>(1,0) * tempPointXi[0] + RotationMatrix2D.at<float>(1,1) * tempPointXi[1];

//            // set third coordinate to zero
//            tempPointXPrime[2] = 0.0;

//            // represent current point in camera coordinate system (x, y, z) and add centroid coordinates

//            // calculate x coordinate
//            tempPoint[0] = RotationMatrixXPrimeX.at<float>(0,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(0,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(0,2) * tempPointXPrime[2] + C.x;
//            // calculate y coordinate
//            tempPoint[1] = RotationMatrixXPrimeX.at<float>(1,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(1,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(1,2) * tempPointXPrime[2] + C.y;
//            // calculate z coordinate
//            tempPoint[2] = RotationMatrixXPrimeX.at<float>(2,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(2,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(2,2) * tempPointXPrime[2] + C.z;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


//        // save all projected sampling points in camera coordinate system
//        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2DXi.size(); ++i)
//        {
//            // calculate x' coordinate of current point
//            tempPointXPrime[0] = RotationMatrix2D.at<float>(0,0) * vSamplingPointsShiftedProjected2DXi[i][0] + RotationMatrix2D.at<float>(0,1) * vSamplingPointsShiftedProjected2DXi[i][1];
//            // calculate y' coordinate
//            tempPointXPrime[1] = RotationMatrix2D.at<float>(1,0) * vSamplingPointsShiftedProjected2DXi[i][0] + RotationMatrix2D.at<float>(1,1) * vSamplingPointsShiftedProjected2DXi[i][1];
//            // set third coordinate to zero
//            tempPointXPrime[2] = 0.0;

//            // represent current point in camera coordinate system (x, y, z) and add centroid coordinates

//            // calculate x coordinate
//            tempPoint[0] = RotationMatrixXPrimeX.at<float>(0,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(0,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(0,2) * tempPointXPrime[2] + C.x;
//            // calculate y coordinate
//            tempPoint[1] = RotationMatrixXPrimeX.at<float>(1,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(1,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(1,2) * tempPointXPrime[2] + C.y;
//            // calculate z coordinate
//            tempPoint[2] = RotationMatrixXPrimeX.at<float>(2,0) * tempPointXPrime[0] + RotationMatrixXPrimeX.at<float>(2,1) * tempPointXPrime[1] +
//                            RotationMatrixXPrimeX.at<float>(2,2) * tempPointXPrime[2] + C.z;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


//        std::cout << "Centroid-free points in camera coordinate system:" << std::endl;

//        // save all projected sampling points in (xi,eta) coordinate system
//        for(unsigned int i=0; i<vSamplingPointsShifted.size(); ++i)
//        {
//            // x coordinate of current point
//            tempPoint[0] = vSamplingPointsShifted[i][0];
//            // y coordinate
//            tempPoint[1] = vSamplingPointsShifted[i][1];
//            // z coordinate
//            tempPoint[2] = vSamplingPointsShifted[i][2];

//            std::cout << tempPoint[0] << " " <<  tempPoint[1] << " " << tempPoint[2] << std::endl;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


//        std::cout << "Centroid-free points PROJECTED on plane PI in camera coordinate system:" << std::endl;

//        // save all projected sampling points in (xi,eta) coordinate system
//        for(unsigned int i=0; i<vSamplingPointsShiftedProjected.size(); ++i)
//        {
//            // x coordinate of current point
//            tempPoint[0] = vSamplingPointsShiftedProjected[i][0];
//            // y coordinate
//            tempPoint[1] = vSamplingPointsShiftedProjected[i][1];
//            // z coordinate
//            tempPoint[2] = vSamplingPointsShiftedProjected[i][2];

//            std::cout << tempPoint[0] << " " <<  tempPoint[1] << " " << tempPoint[2] << std::endl;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


//        std::cout << "Centroid-free points PROJECTED on plane PI in (x',y') coordinate system:" << std::endl;

//        // save all projected sampling points in (x',y') coordinate system
//        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2D.size(); ++i)
//        {
//            // x' coordinate of current point
//            tempPoint[0] = vSamplingPointsShiftedProjected2D[i][0];
//            // y' coordinate
//            tempPoint[1] = vSamplingPointsShiftedProjected2D[i][1];
//            // set third coordinate to zero
//            tempPoint[2] = 0.0;

//            std::cout << tempPoint[0] << " " <<  tempPoint[1] << " " << tempPoint[2] << std::endl;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


//        std::cout << "Centroid-free points PROJECTED on plane PI in (xi,eta) coordinate system:" << std::endl;

//        // save all projected sampling points in (xi,eta) coordinate system
//        for(unsigned int i=0; i<vSamplingPointsShiftedProjected2DXi.size(); ++i)
//        {
//            // x' coordinate of current point
//            tempPoint[0] = vSamplingPointsShiftedProjected2DXi[i][0];
//            // y' coordinate
//            tempPoint[1] = vSamplingPointsShiftedProjected2DXi[i][1];
//            // set third coordinate to zero
//            tempPoint[2] = 0.0;

//            std::cout << tempPoint[0] << " " <<  tempPoint[1] << " " << tempPoint[2] << std::endl;

//            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

//            nPosition++;
//        }


        // save linear and parabolic trajectory approximation as point cloud
        std::vector<cv::Vec3i> vTrajColors(2);

        // color for linear fit
        vTrajColors[0][0] = 255;
        vTrajColors[0][1] = 0;
        vTrajColors[0][2] = 0;

        // color for parabolical fit
        vTrajColors[1][0] = 0;
        vTrajColors[1][1] = 0;
        vTrajColors[1][2] = 255;

        // save linear and parabolical trajectory approximations
        savePointCloud("FitFunctions", "FitFunction", vTrajColors, vTrajectory);

        std::cout << "Linear and parabolical (2D method) trajectory approximation stored as point clouds!" << std::endl;
    }
    // perform parabolic trajectory approximation in camera coordinate system by averaging all available sampling points (requires time stamps)
    // (STANDARD METHOD)
    else
    {
        // if parabola identification using sampling points projected onto plane PI desired:
        // calculate orthogonal projections of vSamplingPointsReduced onto plane PI
        // (NOT USED)
        if(bProjectedPoints)
        {
            // orthogonal distance between current CENTROID-FREE sampling point and plane PI
            float fCurrOrthDistPI = 0.0;

            vSamplingPointsReducedProjected.resize(vSamplingPointsReduced.size());

            // project all CENTROID-FREE sampling points onto plane PI
            for(unsigned int k=0; k<vSamplingPointsReduced.size(); ++k)
            {
                // calculate orthogonal distance between current CENTROID-FREE sampling point and plane PI
                fCurrOrthDistPI =  vSamplingPointsShifted[k][0] * n_PI_0[0] + vSamplingPointsShifted[k][1] * n_PI_0[1] + vSamplingPointsShifted[k][2] * n_PI_0[2];

                // calculate x coordinate of projected sampling point (add centroid coordinate)
                vSamplingPointsReducedProjected[k][0] = vSamplingPointsShifted[k][0]-fCurrOrthDistPI * n_PI_0[0] + C.x;

                // y coordinate of projected sampling point (add centroid coordinate)
                vSamplingPointsReducedProjected[k][1] = vSamplingPointsShifted[k][1]-fCurrOrthDistPI * n_PI_0[1] + C.y;

                // z coordinate of projected sampling point (add centroid coordinate)
                vSamplingPointsReducedProjected[k][2] = vSamplingPointsShifted[k][2]-fCurrOrthDistPI * n_PI_0[2] + C.z;

            }

        }

        // construct (3*vSamplingPointsReduced.size() x 9) matrix T of sampling time points
        cv::Mat T = cv::Mat::zeros(3*vSamplingPointsReduced.size(), 9, CV_32F);

        // fill T matrix with time stamps of individual sampling points in 3D coordinate system of left (virtual) camera

        // reset counter of inserted sampling points
        nPointCounter = 0;

        // declare variable for maximum number of valid points of current experiment
        int nMax;

        // for all droplet shooting experiments
        for (int i=0; i<10; i++)
        {
            // update maximum number of valid points of current experiment
            nMax = (bReduced == true) ? nMaxNumOfSamplingPoints : vNumberIndividualSamplingPoints[i];

            std::cout << "nMax: " << nMax << std::endl;

            // insert time stamp values of current experiment into matrix T
            for (int j=nPointCounter; j<nPointCounter + nMax; ++j)
            {
                // first three columns
                T.at<float>(j,0) = vSamplingPointsReduced[j][3]*vSamplingPointsReduced[j][3];
                T.at<float>(j,1) = vSamplingPointsReduced[j][3];
                T.at<float>(j,2) = 1.0;

                // middle three columns
                T.at<float>(j+vSamplingPointsReduced.size(),3) = vSamplingPointsReduced[j][3]*vSamplingPointsReduced[j][3];
                T.at<float>(j+vSamplingPointsReduced.size(),4) = vSamplingPointsReduced[j][3];
                T.at<float>(j+vSamplingPointsReduced.size(),5) = 1.0;

                // last three columns
                T.at<float>(j+2*vSamplingPointsReduced.size(),6) = vSamplingPointsReduced[j][3]*vSamplingPointsReduced[j][3];
                T.at<float>(j+2*vSamplingPointsReduced.size(),7) = vSamplingPointsReduced[j][3];
                T.at<float>(j+2*vSamplingPointsReduced.size(),8) = 1.0;
            }

            // increment counter of successfully inserted sampling points
            nPointCounter += nMax;
            std::cout << "nPointCounter: " << nPointCounter << std::endl;
        }

        // construct (3*vSamplingPointsReduced.size() x 1) column vector P of sampling point coordinates (x,y,z)
        cv::Mat P = cv::Mat::zeros(3*vSamplingPointsReduced.size(), 1, CV_32F);

        // fill P vector with sampling point coordinates

        // reset counter of inserted sampling points
        nPointCounter = 0;

        // for all droplet shooting events
        for (int i=0; i<10; i++)
        {
            // update maximum number of valid points of current experiment
            nMax = (bReduced == true) ? nMaxNumOfSamplingPoints : vNumberIndividualSamplingPoints[i];

            // for all sampling points of current droplet shooting event
            for (int j=nPointCounter; j<nPointCounter + nMax; ++j)
            {
                if(bProjectedPoints)
                {
                    // x coordinates
                    P.at<float>(j,0) = vSamplingPointsReducedProjected[j][0];

                    // y coordinates
                    P.at<float>(j+vSamplingPointsReducedProjected.size(),0) = vSamplingPointsReducedProjected[j][1];

                    // z coordinates
                    P.at<float>(j+2*vSamplingPointsReducedProjected.size(),0) = vSamplingPointsReducedProjected[j][2];
                }
                else
                {
                    // x coordinates
                    P.at<float>(j,0) = vSamplingPointsReduced[j][0];

                    // y coordinates
                    P.at<float>(j+vSamplingPointsReduced.size(),0) = vSamplingPointsReduced[j][1];

                    // z coordinates
                    P.at<float>(j+2*vSamplingPointsReduced.size(),0) = vSamplingPointsReduced[j][2];
                }
            }

            // increment counter of successfully inserted sampling points
            nPointCounter += nMax;
            std::cout << "nPointCounter: " << nPointCounter << std::endl;
        }

        std::cout << "Global time stamp matrix T: " << T << std::endl;

        std::cout << "Global sampling point coordinate vector P: " << P << std::endl;

        // construct (9x1) column coordinate vector F of trajectory-defining vectors to be identified
        cv::Mat F = cv::Mat::zeros(9, 1, CV_32F);

        // solve over-determined system T*F = P using SVD method
        cv::solve(T, P, F, DECOMP_SVD);

        std::cout << "Identified global parameter vector F: " << F << std::endl;

        // _____________________________________store F vector of global parabolical trajectory approximation in YML file_____________________________________

        trajInfos << "F" << F;

        // _____________________________________calculate and store orthogonal distance of all sampling points to global parabolical trajectory approximation____

        // inject different F vector components, if desired

        //        F.at<float>(0,0) = -3.482653479e-03;
        //        F.at<float>(1,0) = 3.05369177737669e-01;
        //        F.at<float>(2,0) = 1.07582581917396e+01;
        //        F.at<float>(3,0) = -5.693464657e-04;
        //        F.at<float>(4,0) = -3.27590433445617e-02;
        //        F.at<float>(5,0) = 1.21808960476477e+00;
        //        F.at<float>(6,0) = 3.302525334e-03;
        //        F.at<float>(7,0) = 3.18414039243754e-01;
        //        F.at<float>(8,0) = -1.38108789793898e+01;

        avg_dist_para = calculateAverageGeometricDistancePara(F, vSamplingPoints);

        trajInfos << "avg_dist_para" << avg_dist_para;

        trajInfos.release();

        std::cout << "Total number of available sampling points (without reduction): " << vSamplingPoints.size() << std::endl;

        // _________________________________________________draw parabola as point cloud_________________________________________________________________________

        // reset values
        fZMin = 0.0;        // minimum z coordinate value
        fZMax = 200.0;      // maximum z coordinate value
        nPosition = 0;      // frame column iterator

        // current time point during plotting of spatial parabola
        double fParabolaDrawingTimePoint = 0.0;

        // variable to store last z value
        double fLastZValue = 0.0;

        // initialize z coordinate of temporary point
        tempPoint[2] = -20.0;

        vTrajectory[1] = cv::Mat::zeros(5000, 5000, CV_64FC3);

        while((tempPoint[2] < fZMax))
        {
            tempPoint = cv::Vec3d(F.at<float>(2,0)+fParabolaDrawingTimePoint*F.at<float>(1,0)+fParabolaDrawingTimePoint*fParabolaDrawingTimePoint*F.at<float>(0,0),
                              F.at<float>(5,0)+fParabolaDrawingTimePoint*F.at<float>(4,0)+fParabolaDrawingTimePoint*fParabolaDrawingTimePoint*F.at<float>(3,0),
                              F.at<float>(8,0)+fParabolaDrawingTimePoint*F.at<float>(7,0)+fParabolaDrawingTimePoint*fParabolaDrawingTimePoint*F.at<float>(6,0));

            // avoid faulty points
            if ((tempPoint[2] < fZMin))
            {
                continue;
            }

            // increment current time point
            fParabolaDrawingTimePoint += 0.01;

            // if vertex of parabola was reached
            if(tempPoint[2] < fLastZValue)
            {
                break;
            }

            vTrajectory[1].at<cv::Vec3d>(0,nPosition) = tempPoint;

            nPosition++;

            // std::cout << "nPosition: " << nPosition << std::endl;

            fLastZValue = tempPoint[2];
        }

        // save linear and parabolic trajectory approximation as point cloud

        std::vector<cv::Vec3i> vTrajColors(2);

        // linear fit
        vTrajColors[0][0] = 255;
        vTrajColors[0][1] = 0;
        vTrajColors[0][2] = 0;

        // parabolical fit
        vTrajColors[1][0] = 0;
        vTrajColors[1][1] = 0;
        vTrajColors[1][2] = 255;

        // save linear and parabolical trajectory approximations
        savePointCloud("Global Fit Functions", "FitFunction", vTrajColors, vTrajectory);

        std::cout << "Linear and parabolical trajectory approximation stored as point clouds!" << std::endl;
    }

    std::cout << "Hit key to end program!" << std::endl;

    cv::waitKey(0);

    return 0;
}
