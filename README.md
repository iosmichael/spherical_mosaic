# Spherical Mosaic - Computational Photography Final Project
Libraries we used in this project:
- **OpenCV** (3.4.2) for feature detection/matching
- **Eigen3** for ceres compatibility
- **Ceres** for large-scale non-linear optimization
- **Boost** for file management system, sorting and random algorithms


## OPENCV VERSION 3.4.2 (w. SIFT/SURF Support)
Download the opencv git repo and switch to the version 3.4.2 (for SIFT/SURF feature detection support)

```bash
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/3.4.2 -b opencv3.4.2
```

Download the opencv_contrib git repo and switch to the version 3.4.2
```bash
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout tags/3.4.2 -b opencv_contrib3.4.2
```

Cmake build and install opencv 3.4.2 project and ensure that EXTRA MODULE and ENABLE NONFREE is set

```bash
cd opencv
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=<path_to_directory>/opencv_contrib/modules -DOPENCV_ENABLE_NONFREE=True -DBUILD_opencv_rgbd=OFF ..

sudo make install -j8
```

To test the version of your OPENCV, you want to find the OpenCVConfig.cmake. To find out this, use:
```bash
sudo updatedb
locate OpenCVConfig.cmake
```
The command will give you the path for OpenCVConfig.cmake (normally under the folder /usr/local/share/OpenCV/), open the file and find the line "SET(OpenCV_VERSION *.*.*)". Make sure the number match with 3.4.2.

## Eigen3 INCLUDE ISSUE

When we use **CERES** library, the include header will often gives warning on "Eigen/Core" not found in the system path, in spite of including Eigen3 exclusively in our CMAKELIST.txt. To resolve this issue, we need to create a soft-link in the eigen3 directory.

The path to eigen3 directory can be found by printing out the CMAKE variable $EIGEN3_INCLUDE_DIRS

```bash
cd <path-directory-to-eigen3>
sudo ln -sf eigen3/Eigen Eigen
```

## Screenshots of the Performance

**Undistorted Image Display**

![rectified image](../demo/undistort.png)

**SIFT Feature Detection and KNN Matching**

![SIFT feature detection and matching](../demo/featureDetectionAndMatching.png =400x)

**Outlier Rejection**

![before outlier rejection](../demo/featureMatchWithoutOutlier.png =400x)

![after outlier rejection](../demo/featureMatchingWithOutlier.png =400x)

**Reprojection Error Visualization**

![reprojection](../demo/reprojectionError.png)

**Bundle Adjustment**

                             |        Original       |           Reduced
Parameter blocks              |           6080         |            6080
Parameters                     |         18240          |          18240
Residual blocks                 |        16701           |         16701
Residuals                        |       33402            |        33402


iter    |  cost   |   cost_change | \|gradient\| |  \|step\|  |  tr_ratio | tr_radius | ls_iter | iter_time | total_time
   0 | 7.257846e+03  |  0.00e+00  |  4.10e+04  | 0.00e+00 |  0.00e+00 | 1.00e+04  |      0  |  6.52e-01  |  6.81e-01
   1 | 6.879847e+03  |  3.78e+02  |  |1.18e+02 | 2.52e-01 |  1.00e+00 | 3.00e+04  |    1  |  7.45e-01  |  1.43e+00
   2 | 6.879630e+03  |  2.17e-01  |  3.59e+00  | 3.44e-02 |  1.00e+00 | 9.00e+04  |      1 |   6.48e-01 |   2.07e+00

Cost:
Initial              |            7.257846e+03
Final                 |           6.879630e+03
Change                 |          3.782157e+02

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 9.642726e-08 <= 1.000000e-06)

Total difference SSE on Angle-Axis Representation: **0.0988294**

**Spherical Mosaic Reconstruction**

![mosaic0](../demo/mosaic0.png)
![mosaic1](../demo/mosaic1.png)
![mosaic2](../demo/mosaic2.png)
![mosaic3](../demo/mosaic4.png)