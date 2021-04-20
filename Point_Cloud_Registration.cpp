// Contributors: Hardik R. Dava/Jacob F. Fast
// 2020/2021
//
// Point_Cloud_Registration.cpp
//
// Read two point clouds and marker point coordinates of corresponding markers in both point clouds.
// Calculate rigid transformation between marker positions, visualize and save registered point clouds.

#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

#include <iostream>
#include <pcl/io/ply_io.h>
#include <ctime>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>

using namespace std;
using namespace pcl;

// FUNCTIONS

// split each line with "space" change
void splitPoints(const std::string& inputStr, std::vector<std::string>& splitVec)
{
    std::size_t pos = 0, found;

    while((found = inputStr.find_first_of(' ', pos)) != std::string::npos)
    {
        splitVec.push_back(inputStr.substr(pos, found - pos));
        pos = found+1;
    }

    splitVec.push_back(inputStr.substr(pos));
}

// load point cloud from file
void loadPoints(std::string &filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::ifstream pointsFile;
    pointsFile.open(filename);

    std::string line;
    std::vector<std::string> tmpLineCont;
    std::vector<std::vector<float>> pts;

    if (pointsFile.is_open())
    {
        while ( std::getline (pointsFile,line) )
        {
            tmpLineCont.clear();
            splitPoints(line, tmpLineCont);
            if ( tmpLineCont.size() == 3)
            {
                float strToDouble0 = std::atof(tmpLineCont[0].c_str());
                float strToDouble1 = std::atof(tmpLineCont[1].c_str());
                float strToDouble2 = std::atof(tmpLineCont[2].c_str());

                std::vector<float> tmpDoubleVec;
                tmpDoubleVec.push_back(strToDouble0);
                tmpDoubleVec.push_back(strToDouble1);
                tmpDoubleVec.push_back(strToDouble2);

                pts.push_back(tmpDoubleVec);
            }
        }
        pointsFile.close();
    }

    pcl::PointXYZ tmpPoint;

    for (unsigned int i = 0; i < pts.size(); ++i)
    {

        tmpPoint.x = pts[i][0];
        tmpPoint.y = pts[i][1];
        tmpPoint.z = pts[i][2];
        cloud->points.push_back(tmpPoint);
    }
}

// declarations for source and target point clouds
pcl::PointCloud<pcl::PointXYZ>::Ptr sourcemain (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr targetmain (new pcl::PointCloud<pcl::PointXYZ>);

// declarations for source and target markers
pcl::PointCloud<pcl::PointXYZ>::Ptr SGBMMarkers (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr LaserScanMarkers (new pcl::PointCloud<pcl::PointXYZ>);

// declarations for OPTIONAL cropped source point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr sourcemain_cropped (new pcl::PointCloud<pcl::PointXYZ>);

// flag for point cloud cropping procedure
bool bCropSourceCloud = true;

// flag for fine registration using ICP (DO NOT CHANGE)
bool bPerformAdditionalICP = true;

int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    std::cout << "Installed PCL version: " << PCL_VERSION_PRETTY << std::endl;

    // load source and target point clouds

    pcl::io::loadPLYFile("SGBM.ply", *sourcemain);
    pcl::io::loadPLYFile("GT.ply", *targetmain);

    // declarations of txt files for marker positions

    std::string markers1 = "SGBM_markers.txt";
    std::string markers2 = "GT_markers.txt";

    // load 3D coordinates of fiducial markers from txt files
    loadPoints(markers1, SGBMMarkers);
    loadPoints(markers2, LaserScanMarkers);

    // OPTIONAL STEP: crop source point cloud using spherical template
    if (bCropSourceCloud)
    {
        // define center of sphere

        // rod lens system
        // pcl::PointXYZ sphereCenter(3.368935, -15.6925, 61.9570995);

        // fiberoptic system
        pcl::PointXYZ sphereCenter(2.88224, -4.583455, 60.6016505);

        // define radius of sphere (here: in mm)
        float fSphereRadius = 16.0;

        for(unsigned int nIndex = 0; nIndex < sourcemain->points.size(); nIndex++)
        {
            if(std::sqrt(std::pow(sourcemain->points[nIndex].x - sphereCenter.x,2) + std::pow(sourcemain->points[nIndex].y - sphereCenter.y,2) + std::pow(sourcemain->points[nIndex].z - sphereCenter.z,2)) < fSphereRadius)
            {
                // add point to "sourcemain_cropped" point cloud
                sourcemain_cropped->points.push_back(sourcemain->points[nIndex]);
            }
        }
    }

    // perform SVD-based identification of the optimum rigid transformation between marker positions

    // declarations
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ> SVD;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ>::Matrix4 transformation;

    // return registration transformation as (4x4) homogeneous transformation matrix
    SVD.estimateRigidTransformation(*SGBMMarkers, *LaserScanMarkers, transformation);

    // apply rigid transformation on source point cloud as well as source fiducial markers
    if (!bCropSourceCloud)
    {
        pcl::transformPointCloud (*sourcemain, *sourcemain, transformation);
    }
    else
    {
        pcl::transformPointCloud (*sourcemain_cropped, *sourcemain_cropped, transformation);
    }

    pcl::transformPointCloud (*SGBMMarkers, *SGBMMarkers, transformation);

    // OPTIONAL STEP: apply ICP-based fine-registration on pre-registered point clouds
    if(bPerformAdditionalICP)
    {
        pcl::IterativeClosestPoint<PointXYZ, PointXYZ> icp;

        // set the input and target point cloud
        if (!bCropSourceCloud)
        {
            icp.setInputSource(sourcemain);
        }
        else
        {
            icp.setInputSource(sourcemain_cropped);
        }

        icp.setInputTarget(targetmain);

        // set the maximum distance between correspondences to 2 cm (e.g., correspondences with higher distances will be ignored)
        // icp.setMaxCorrespondenceDistance(0.02);

        // set the transformation epsilon
        // icp.setTransformationEpsilon(1e-9);

        // set Euclidean distance difference epsilon
        // icp.setEuclideanFitnessEpsilon(1);

        // set the maximum number of iterations
        icp.setMaximumIterations(100);

        // declaration for registered point cloud after ICP
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_registered_ICP (new pcl::PointCloud<pcl::PointXYZ>);

        // perform ICP alignment
        icp.align(*cloud_source_registered_ICP);

        if(icp.hasConverged())
        {
            std::cout << "ICP has converged! " << std::endl;
        }
        else
        {
            std::cout << "ICP has NOT converged! " << std::endl;
        }

        std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;

        // obtain the transformation that aligns source point cloud to target point cloud after ICP
        Eigen::Matrix4f transformation_ICP = icp.getFinalTransformation();

        // apply rigid transformation on pre-registered source point cloud as well as source fiducial markers
        if (!bCropSourceCloud)
        {
            pcl::transformPointCloud (*sourcemain, *sourcemain, transformation_ICP);
        }
        else
        {
            pcl::transformPointCloud (*sourcemain_cropped, *sourcemain_cropped, transformation_ICP);
        }

        pcl::transformPointCloud (*SGBMMarkers, *SGBMMarkers, transformation_ICP);
    }

    // visualize the point clouds after registration
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("Point Cloud Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(targetmain, 120, 120, 120);
    viewer->addPointCloud<pcl::PointXYZ>(targetmain, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target" );

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> targetmarker_color (LaserScanMarkers, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (LaserScanMarkers, targetmarker_color, "targetmarker");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "targetmarker");

    if (!bCropSourceCloud)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color (sourcemain, 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZ> (sourcemain, source_color, "source" );
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
    }
    else
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color (sourcemain_cropped, 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZ> (sourcemain_cropped, source_color, "source");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
    }

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sourcemarker_color (SGBMMarkers, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (SGBMMarkers, sourcemarker_color, "sourcemarker");
    viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "sourcemarker");

    viewer->resetCamera();
    viewer->spin();

    viewer->removePointCloud ("sourcemarker");
    viewer->removePointCloud ("targetmarker");
    viewer->removePointCloud ("target");
    viewer->removePointCloud ("source");

    // save registered point clouds

    pcl::io::savePLYFile("GT_registered.ply", *targetmain);

    if (!bCropSourceCloud)
    {
        pcl::io::savePLYFile("SGBM_registered.ply", *sourcemain);
    }
    else
    {
        pcl::io::savePLYFile("SGBM_registered.ply", *sourcemain_cropped);
    }

    return 0;
}
