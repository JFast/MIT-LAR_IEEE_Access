# MIT-LAR_IEEE_Access
This repository contains source code for the targeted, droplet-based stimulation of the laryngeal adductor reflex (LAR) using a stereoscopic laryngoscope.

# Table of Contents
* [Requirements](#requirements)
* [How to Cite this Repository](#how-to-cite-this-repository)
* [Descriptions of Source Files](#descriptions-of-source-files)
* [Notes](#notes)

# Requirements

Required packages: *OpenCV* (tested with version 3.3.1), *Point Cloud Library* (tested with version 1.11.1-dev).

All code executable on a standard desktop PC running the Linux distribution *Ubuntu* (version 18.04.3 LTS).

Code developed using Qt Creator (version 4.5.2).

# How to Cite this Repository

```BibTeX
@misc{
     FastIEEE.2021, 
     author={Jacob F. Fast and Hardik R. Dava and Adrian K. R\"uppel and Dennis Kundrat and Maurice Krauth and Max-Heinrich Laves and Svenja Spindeldreier and L\"uder A. Kahrs and Martin Ptok}, 
     year={2021},
     title={Code repository associated with the contribution "Stereo laryngoscopic impact site prediction for targeted, droplet-based stimulation of the laryngeal adductor reflex for latency measurements"}, 
     DOI={},
     publisher={GitHub}
     }
```

# Descriptions of Source Files

## Calibration

### Calibration_Procedure.cpp

Read calibration images and save XML file with stereo calibration results.

## (Droplet) Trajectory Approximation

### Trajectory_Identification.cpp

Read raw stereolaryngoscopic frame sequence showing droplet flight and corresponding camera calibration parameters. Perform spatial triangulation of detected droplet centroid positions (using blob instead of Hough circle detection) and identify linear and parabolic trajectory approximations. Used for the analysis of individual droplet flight recordings.

### Avg_Traj_and_Plane_Calculation.cpp

Read 3D droplet centroid positions and associated time stamps from 10 available droplet shooting events obtained by execution of Trajectory_Identification.cpp. Calculate global fit plane, global linear (DEPRECATED MODEL) droplet trajectory approximation and global parabolical (STANDARD MODEL) droplet trajectory approximation. Return CSV files with individual distances of centroid positions to fit plane, fit line and fit parabola. Return fit plane, linear approximation and parabolical approximation as point clouds. Return defining parameters of fit plane, line and parabola in YML file. Used for global analysis of a set of droplet flight recordings acquired at identical system conditions.

## Stereo Reconstruction

### Stereo_Reconstruction_and_Prediction.cpp

Read raw stereolaryngoscopic single frame (or frame sequence) and corresponding camera calibration parameters and perform spatial stereo reconstruction. If desired, calculate droplet impact site based on spatial stereo reconstruction of target and available trajectory parameter file.

## Live Application

### Live_Application.cpp

...

## Stereo Reconstruction Accuracy Assessment

### Point_Cloud_Registration.cpp

Read two point clouds and marker point coordinates of corresponding markers in both point clouds. Optionally crop source point cloud to points contained in a sphere with given center and radius. Calculate rigid transformation between marker positions. Optionally refine transformation by additional ICP step. Visualize and save registered point clouds.

### Congruence_Evaluation.cpp

Read two previously registered point clouds and calculate congruence between them using nearest neighbor Euclidean distance between reconstructed (source) and ground truth (target) point cloud.

### Frame_Extraction.cpp

Read stereo laryngoscopic frame sequence and extract subset of frames for further processing. This procedure is used to obtain input frames for subsequent evaluation of stereo reconstruction accuracy using a 3D phantom of the human larynx.

# Notes

All distance values given in mm for numerical stability at small horopter volumes.

All temporal values given in ms for numerical stability at small time scales.
