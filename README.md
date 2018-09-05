# IoT Reference Implementation: Store Traffic Monitor in Python\*

| Details            |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  Python\* |
| Time to Complete:    |  50-70min     |

![store-traffic-monitor](docs/images/store-traffic-monitor-image.png)

An application capable of detecting objects on any number of screens.

## What it Does
This application is one of a series of IoT reference implementations aimed at instructing users on how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This reference implementation monitors the activity of people inside and outside a facility, as well as counts product inventory.

## How it Works
The counter uses the Inference Engine included in the OpenVINO™ toolkit. A trained neural network detects objects within a designated area by displaying a green bounding box over them. This reference implementation identifies multiple intruding objects entering the frame and identifies their class, count, and time entered. <br>

## Requirements
### Hardware
* 6th Generation Intel® Core™ processor with Intel® Iris® Pro graphics and Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)
*Note*: You must be running kernel version 4.7+ to use this software. We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:

```
uname -a
```
* OpenCL™ Runtime Package
* OpenVINO™ toolkit

## Setup

### Install OpenVINO™ Toolkit
Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the OpenVINO™ toolkit.

You will need the OpenCL™ Runtime package if you plan to run inference on the GPU as shown by the
instructions below. It is not mandatory for CPU inference.

### Ffmpeg
ffmpeg is installed separately from the Ubuntu repositories:
```
sudo apt update
sudo apt install ffmpeg
```

## Configure the Application

All configurations are written to resources/conf.txt
* 1st line: `path/to/model.xml`  
   This is the path to the model topology on the local system.  
   The model topology file is the .xml file that the model optimizer produces, containing the IR of the model's topology.

* 2nd line: `path/to/model.bin`  
   This is the path to the model weights on the local system.  
   The model weights file is the .bin file that the model optimizer produces, containing the IR of the model's weights.

* 3rd line: `path/to/labels`  
   This is a path to the labels file on the local system.  
   The labels file is a text file containing, all the classes/labels that the model can recognize, in the order that it was trained to recognize them (one class per line). All detection models work with integer labels and not string labels (e.g. for the SSD model, the number 15 represents the class "person").   
   For the ssd300 model, we provide the class file labels.txt in the [resources folder](./application/resources/labels.txt).
   
Each of the following lines contain the `path/to/video` followed by the label to be detected on that video, e.g.:
```
videos/video1.mp4 person
```
The `path/to/video` is the path, on the local system, to a video to use as input. The labels used must coincide with the labels from the labels file.

### What Model to Use
The application works with any object-detection model, provided it has the same input and output format of the SSD model.  
The model can be any object detection model:
* that is provided by OpenVINO™ toolkit.  
   These can be found in the `deployment_tools/intel_models` folder.
* downloaded using the **model downloader**.   
   These can be found in the `deployment_tools/model_downloader/object_detection` folder.
* created by the user

For first-use, we recommend using the ssd300 model provided by the **model downloader**.

### What Input Video to Use
The application works with any input video.
Sample videos for object detection are provided [here](https://github.com/intel-iot-devkit/sample-videos/).  


For first-use, we recommend using the [people-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/people-detection.mp4), [one-by-one-person-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/one-by-one-person-detection.mp4), [bottle-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/bottle-detection.mp4) videos.   
E.g.:
```
sample-videos/people-detection.mp4 person
sample-videos/one-by-one-person-detection.mp4 person
sample-videos/bottle-detection.mp4 bottle
```
These videos can be downloaded directly, via the `video_downloader` python script provided. The script works with both python2 and python3. Run the following command: 
```
python video_downloader.py
```
The videos are automatically downloaded to the `resources/` folder.

### Having the Input Video Loop
By default, the application reads the input videos only once, and ends when the videos end.
In order to not have the sample videos end, thereby ending the application, the option to continously loop the videos is provided.   
This is done by setting the existing LOOP_VIDEO global variable to True. E.g.:
```
LOOP_VIDEO = True
```

### Using Camera Stream Instead of Video File
Replace `path/to/video` with the camera ID, where the ID is taken from yout video device (the number X in /dev/videoX).
On Ubuntu, to list all available video devices use the following command:
```
ls /dev/video*
```

## Run the Application

Setup the environment variables required to run OpenVINO™ toolkit applications:
```
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```
Run the Python script:

```
./store-traffic-monitor.py
```

## Using the browser UI

The default application uses a simple user interface created with OpenCV.
A web based UI, with more features is also provided [here](./UI).
In order for the application to work with the browser UI, you have to set the existing UI_OUTPUT global variable to TRUE. E.g.:
```
UI_OUTPUT = True
```
