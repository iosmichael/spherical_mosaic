## ENABLE OPENCV VERSION 3.4.2
---
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
cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=<directory path to opencv_contrib>/modules ..
make -j8
make install
```

To test the version of your OPENCV, you want to find the OpenCVConfig.cmake. To find out this, use:
```bash
sudo updatedb
locate OpenCVConfig.cmake
```
The command will give you the path for OpenCVConfig.cmake (normally under the folder /usr/local/share/OpenCV/), open the file and find the line "SET(OpenCV_VERSION *.*.*)". Make sure the number match with 3.4.2.