#!/bin/bash
#
# Copyright (C) 2018-2019 Intel Corporation
#

#Install the dependencies
sudo apt update
sudo apt install ffmpeg
sudo apt-get update
sudo apt-get install python3-pip python3-numpy
sudo pip3 install jupyter

BASE_DIR=`pwd`

#Download the video
cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4

#Download the model
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name mobilenet-ssd

#Optimize the model
cd /opt/intel/openvino/deployment_tools/model_optimizer/
./mo_caffe.py --input_model /opt/intel/openvino/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o $BASE_DIR/resources/FP32 --data_type FP32 --scale 256 --mean_values [127,127,127]
./mo_caffe.py --input_model /opt/intel/openvino/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o $BASE_DIR/resources/FP16 --data_type FP16 --scale 256 --mean_values [127,127,127]

