#!/bin/bash

# Copyright (c) 2018 Intel Corporation.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Install the dependencies
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install python3-pip
sudo pip3 install numpy jupyter

BASE_DIR=`pwd`

#Download the video
cd resources
wget -O one-by-one-person-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
wget -O people-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
wget -O bottle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4

#Download the model
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name mobilenet-ssd

#Optimize the model
cd /opt/intel/openvino/deployment_tools/model_optimizer/
./mo_caffe.py --input_model /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/public/mobilenet-ssd/mobilenet-ssd.caffemodel  -o $BASE_DIR/resources/FP32 --data_type FP32 --scale 256 --mean_values [127,127,127]
./mo_caffe.py --input_model /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/public/mobilenet-ssd/mobilenet-ssd.caffemodel  -o $BASE_DIR/resources/FP16 --data_type FP16 --scale 256 --mean_values [127,127,127]

