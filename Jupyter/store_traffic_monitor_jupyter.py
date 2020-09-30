#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

##########################################################
# INCLUDES
##########################################################

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy
import time
import datetime
import collections
import threading
import queue
import json
import datetime
import numpy as np
from inference import Network

##########################################################
# CONSTANTS
##########################################################

CONFIG_FILE = '../resources/config.json'
CONF_VIDEODIR = '../UI/resources/video_frames/'
CONF_DATAJSON_FILE = '../UI/resources/video_data/data.json'
CONF_VIDEOJSON_FILE = '../UI/resources/video_data/videolist.json'
CPU_EXTENSION = ''
TARGET_DEVICE = 'CPU'
STATS_WINDOW_NAME = 'Statistics'
CAM_WINDOW_NAME_TEMPLATE = 'Video {}'
PROB_THRESHOLD = 0.145
FRAME_THRESHOLD = 5
WINDOW_COLUMNS = 3
LOOP_VIDEO = False
UI_OUTPUT = False
is_async_mode = True

##########################################################
# GLOBALS
##########################################################

model_xml = ''
model_bin = ''
labels_file = ''
videoCaps = []
videoCapsJson = []
display_lock = threading.Lock()
log_lock = threading.Lock()
frames = 0
frameNames = []
numVids = 20000
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']

##########################################################
# CLASSES
##########################################################


class FrameInfo:
    def __init__(self, frameNo=None, count=None, timestamp=None):
        self.frameNo = frameNo
        self.count = count
        self.timestamp = timestamp


class VideoCap:
    def __init__(self, cap, req_label, cap_name, is_cam):
        self.cap = cap
        self.req_label = req_label
        self.cap_name = cap_name
        self.is_cam = is_cam
        self.cur_frame = numpy.array([], dtype='uint8')
        self.initial_w = 0
        self.initial_h = 0
        self.frames = 0
        self.cur_frame_count = 0
        self.total_count = 0
        self.last_correct_count = 0
        self.candidate_count = 0
        self.candidate_confidence = 0
        self.closed = False
        self.countAtFrame = []
        self.video = None
        self.frame = None

        if not is_cam:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = 0

        self.videoName = cap_name + "_inferred.mp4"

    def init_vw(self, h, w, fps):
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video = cv2.VideoWriter(os.path.join('../resources', self.videoName),
                                     self.fourcc, fps, (w, h), True)
        if not self.video.isOpened():
            print("Could not open for write" + self.videoName)
            sys.exit(1)


##########################################################
# FUNCTIONS
##########################################################

def env_parser():
    global TARGET_DEVICE, model_xml, labels_file, CPU_EXTENSION, UI_OUTPUT, numVids, LOOP_VIDEO, is_async_mode
    if 'DEVICE' in os.environ:
        TARGET_DEVICE = os.environ['DEVICE']
    if 'MODEL' in os.environ:
        model_xml = os.environ['MODEL']
    if 'LABELS' in os.environ:
        labels_file = os.environ['LABELS']
    if 'DEVICE' in os.environ:
        TARGET_DEVICE = os.environ['DEVICE']
    if 'CPU_EXTENSION' in os.environ:
        CPU_EXTENSION = os.environ['CPU_EXTENSION']
    if 'UI' in os.environ:
        ui = os.environ['UI']
        if ui == "true":
            UI_OUTPUT = True

    if 'LOOP' in os.environ:
        lp = os.environ['LOOP']
        if lp == "true":
            LOOP_VIDEO = True
        if lp == "false":
            LOOP_VIDEO = False
    if 'NUM_VIDEOS' in os.environ:
        numVids = int(os.environ['NUM_VIDEOS'])
    if 'FLAG' in os.environ:
        async_mode = os.environ['FLAG']
        if async_mode == "async":
            is_async_mode = True
        else:
            is_async_mode = False


def check_args():
    # ArgumentParser checks model and labels by default right now
    if model_xml == '':
        print("You need to specify the path to the .xml file")
        print("Use -m MODEL or --model MODEL")
        sys.exit(11)
    if labels_file == '':
        print("You need to specify the path to the labels file")
        print("Use -l LABELS or --labels LABELS")
        sys.exit(12)

    global TARGET_DEVICE
    if 'MULTI' not in TARGET_DEVICE and TARGET_DEVICE not in accepted_devices:
        print("Unsupported device: " + TARGET_DEVICE)
        sys.exit(13)
    elif 'MULTI' in TARGET_DEVICE:
        target_devices = TARGET_DEVICE.split(':')[1].split(',')
        for multi_device in target_devices:
            if multi_device not in accepted_devices:
                print("Unsupported device: " + TARGET_DEVICE)
                sys.exit(13)

    if numVids < 1:
        print("Please set NUM_VIDEOS to at least 1")
        sys.exit(14)


def parse_conf_file():
    """
        Parses the configuration file.
        Reads videoCaps
    """
    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    cnt = 0
    for idx, item in enumerate(config['inputs']):
        if cnt < numVids:
            if item['video'].isdigit():
                videoCap = VideoCap(cv2.VideoCapture(int(item['video'])),
                                    item['label'],
                                    CAM_WINDOW_NAME_TEMPLATE.format(idx),
                                    True)
            else:
                if os.path.isfile(item['video']):
                    videoCap = VideoCap(cv2.VideoCapture(item['video']),
                                        item['label'],
                                        CAM_WINDOW_NAME_TEMPLATE.format(idx),
                                        False)
                else:
                    print("Couldn't find " + item['video'])
                    sys.exit(3)
            videoCaps.append(videoCap)
            cnt += 1
        else:
            break

    for vc in videoCaps:
        if not vc.cap.isOpened():
            print("Could not open for reading " + vc.cap_name)
            sys.exit(2)


def arrange_windows(width, height):
    """
        Arranges the windows so they are not overlapping
        Also starts the display threads
    """
    spacer = 25
    cols = 0
    rows = 0

    # Arrange video windows
    for idx in range(len(videoCaps)):
        if cols == WINDOW_COLUMNS:
            cols = 0
            rows += 1
        cv2.namedWindow(CAM_WINDOW_NAME_TEMPLATE.format(idx),
                        cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(CAM_WINDOW_NAME_TEMPLATE.format(idx),
                       (spacer + width) * cols, (spacer + height) * rows)
        cols += 1

    # Arrange statistics window
    if cols == WINDOW_COLUMNS:
        cols = 0
        rows += 1
    cv2.namedWindow(STATS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(STATS_WINDOW_NAME, (spacer + width) * cols,
                   (spacer + height) * rows)


def saveJSON():
    """
        This JSON contains info about current and total object count
    """
    global videoCapsJson
    if UI_OUTPUT:
        dataJSON = open(CONF_DATAJSON_FILE, "w")
        if not os.access(CONF_DATAJSON_FILE, os.W_OK):
            print("Could not open dataJSON file for writing")
            return 5

        videoJSON = open(CONF_VIDEOJSON_FILE, "w")
        if not os.access(CONF_VIDEOJSON_FILE, os.W_OK):
            print("Could not open videoJSON file for writing")
            return 5

        dataJSON.write("{\n")
        videoJSON.write("{\n")
        vsz = len(videoCapsJson)
        for i in range(vsz):
            if len(videoCapsJson[i].countAtFrame) > 0:
                dataJSON.write("\t\"Video_" + str(i + 1) + "\": {\n")
                fsz = len(videoCapsJson[i].countAtFrame) - 1
                for j in range(fsz):
                    strt = "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"" \
                           "time\":\"%s\"\n\t\t},\n" % \
                           (videoCapsJson[i].countAtFrame[j].frameNo,
                            videoCapsJson[i].countAtFrame[j].count,
                            videoCapsJson[i].countAtFrame[j].timestamp)
                    dataJSON.write(strt)
                strt = "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"" \
                       "time\":\"%s\"\n\t\t}\n" % \
                       (videoCapsJson[i].countAtFrame[fsz].frameNo,
                        videoCapsJson[i].countAtFrame[fsz].count,
                        videoCapsJson[i].countAtFrame[fsz].timestamp)
                dataJSON.write(strt)
                dataJSON.write("\t},\n")

        dataJSON.write("\t\"totals\": {\n")
        for i in range(vsz - 1):
            dataJSON.write("\t\t\"Video_" + str(i + 1) + "\": \"" +
                           str(videoCapsJson[i].total_count) + "\",\n")

        i = vsz - 1
        dataJSON.write("\t\t\"Video_" + str(i + 1) + "\": \"" +
                       str(videoCapsJson[i].total_count) + "\"\n")
        dataJSON.write("\t}\n")
        dataJSON.write("}")
        dataJSON.close()

        sz = len(frameNames) - 1
        for i in range(sz):
            videoJSON.write(
                "\t\"" + str(i + 1) + "\":\"" + str(frameNames[i]) + "\",\n")

        i = sz
        videoJSON.write(
            "\t\"" + str(i + 1) + "\":\"" + str(frameNames[i]) + "\"\n")
        videoJSON.write("}")
        videoJSON.close()

        return 0


def main():
    # Plugin initialization for specified device and load extensions library
    global rolling_log
    global TARGET_DEVICE
    global videoCapsJson
    global is_async_mode

    env_parser()
    check_args()
    parse_conf_file()

    # Initialize the class
    infer_network = Network()
    # Load the network to IE Plugin
    n, c, h, w = infer_network.load_model(model_xml, TARGET_DEVICE, 1, 1, 2,
                                          CPU_EXTENSION)[1]
    minFPS = min([i.cap.get(cv2.CAP_PROP_FPS) for i in videoCaps])
    for vc in videoCaps:
        vc.init_vw(h, w, minFPS)

    statsWidth = w if w > 345 else 345
    statsHeight = h if h > (len(videoCaps) * 20 + 15) else (
                len(videoCaps) * 20 + 15)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    statsVideo = cv2.VideoWriter(os.path.join('../resources', 'Statistics.mp4'),
                                 fourcc, minFPS, (statsWidth, statsHeight),
                                 True)
    if not statsVideo.isOpened():
        print("Couldn't open stats video for writing")
        sys.exit(4)

    # Read the labels file
    if labels_file:
        with open(labels_file, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # Init a rolling log to store events
    rolling_log_size = int((h - 15) / 20)
    rolling_log = collections.deque(maxlen=rolling_log_size)

    # Init inference request IDs
    cur_request_id = 0
    next_request_id = 1

    if not UI_OUTPUT:
        # Arrange windows so they are not overlapping
        arrange_windows(w, h)
        print("To stop the execution press Esc button")

    for idx, vc in enumerate(videoCaps):
        vc.start_time = datetime.datetime.now()
        vc.pos = idx

    if UI_OUTPUT:
        videoCapsJson = videoCaps.copy()

    if is_async_mode:
        print("Application running in async mode...")
    else:
        print("Application running in sync mode...")

    while True:

        # If all video captures are closed stop the loop
        no_more_data = [videoCap.closed for videoCap in videoCaps]
        # loop over all video captures
        for idx, videoCapInfer in enumerate(videoCaps):

            # read the next frame
            vfps = int(round(videoCapInfer.cap.get(cv2.CAP_PROP_FPS)))
            for i in range(0, int(round(vfps / minFPS))):
                ret, frame = videoCapInfer.cap.read()
                videoCapInfer.cur_frame_count += 1
                # If the read failed close the program
                if not ret:
                    no_more_data[idx] = True
                    break

            if no_more_data[idx]:
                if UI_OUTPUT:
                    videoCaps.pop(idx)
                    continue
                else:
                    stream_end_frame = np.zeros((h, w, 1), dtype='uint8')
                    cv2.putText(stream_end_frame,
                                "Input file {} has ended".format(
                                    videoCapInfer.cap_name),
                                (20, 150),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                                1)
                    cv2.imshow(videoCapInfer.cap_name, stream_end_frame)
                    cv2.waitKey(1)
                    videoCaps.pop(idx)
                    continue
            # Copy the current frame for later use
            videoCapInfer.cur_frame = frame.copy()
            videoCapInfer.initial_w = videoCapInfer.cap.get(3)
            videoCapInfer.initial_h = videoCapInfer.cap.get(4)
            # Resize and change the data layout so it is compatible
            in_frame = cv2.resize(videoCapInfer.cur_frame, (w, h))
            in_frame = in_frame.transpose(
                (2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))

            infer_start = datetime.datetime.now()
            if is_async_mode:
                # Async enabled and only one video capture
                infer_network.exec_net(next_request_id, in_frame)
                if len(videoCaps) == 1:
                    videoCapResult = videoCapInfer
                # Async enabled and more than one video capture
                else:
                    # Get previous index
                    videoCapResult = videoCaps[
                        idx - 1 if idx - 1 >= 0 else len(videoCaps) - 1]
            else:
                # Async disabled
                infer_network.exec_net(next_request_id, in_frame)
                videoCapResult = videoCapInfer

            if infer_network.wait(cur_request_id) == 0:
                infer_end = datetime.datetime.now()
                res = infer_network.get_output(cur_request_id)
                infer_duration = infer_end - infer_start
                current_count = 0
                # Parse detection results of the current request
                for obj in res[0][0]:
                    class_id = int(obj[1])
                    # Draw only objects when probability more than specified threshold
                    if (obj[2] > PROB_THRESHOLD and
                            videoCapResult.req_label in labels_map and
                            labels_map.index(
                                videoCapResult.req_label) == class_id - 1):
                        current_count += 1
                        xmin = int(obj[3] * videoCapResult.initial_w)
                        ymin = int(obj[4] * videoCapResult.initial_h)
                        xmax = int(obj[5] * videoCapResult.initial_w)
                        ymax = int(obj[6] * videoCapResult.initial_h)
                        # Draw box
                        cv2.rectangle(videoCapResult.cur_frame, (xmin, ymin),
                                      (xmax, ymax), (0, 255, 0), 4, 16)

                if videoCapResult.candidate_count is current_count:
                    videoCapResult.candidate_confidence += 1
                else:
                    videoCapResult.candidate_confidence = 0
                    videoCapResult.candidate_count = current_count

                if videoCapResult.candidate_confidence is FRAME_THRESHOLD:
                    videoCapResult.candidate_confidence = 0
                    if current_count > videoCapResult.last_correct_count:
                        videoCapResult.total_count += current_count - videoCapResult.last_correct_count

                    if current_count is not videoCapResult.last_correct_count:
                        if UI_OUTPUT:
                            currtime = datetime.datetime.now().strftime(
                                "%H:%M:%S")
                            fr = FrameInfo(videoCapResult.frames, current_count,
                                           currtime)
                            videoCapResult.countAtFrame.append(fr)

                        new_objects = current_count - videoCapResult.last_correct_count
                        for _ in range(new_objects):
                            string = "{} - {} detected on {}". \
                                format(time.strftime("%H:%M:%S"),
                                       videoCapResult.req_label,
                                       videoCapResult.cap_name)
                            rolling_log.append(string)

                    videoCapResult.frames += 1
                    videoCapResult.last_correct_count = current_count
                else:
                    videoCapResult.frames += 1

                videoCapResult.cur_frame = cv2.resize(videoCapResult.cur_frame,
                                                      (w, h))

                if UI_OUTPUT:
                    imgName = videoCapResult.cap_name
                    imgName = imgName.split()[0] + "_" + chr(
                        ord(imgName.split()[1]) + 1)
                    imgName += "_" + str(videoCapResult.frames)
                    frameNames.append(imgName)
                    imgName = CONF_VIDEODIR + imgName + ".jpg"
                    cv2.imwrite(imgName, videoCapResult.cur_frame)
                    videoCapsJson[videoCapResult.pos].countAtFrame = videoCapResult.countAtFrame
                    a = saveJSON()
                    if a:
                        return a
                if not UI_OUTPUT:
                    # Add log text to each frame
                    log_message = "Async mode is on." if is_async_mode else \
                        "Async mode is off."
                    cv2.putText(videoCapResult.cur_frame, log_message, (15, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)
                    log_message = "Total {} count: {}" \
                        .format(videoCapResult.req_label,
                                videoCapResult.total_count)
                    cv2.putText(videoCapResult.cur_frame, log_message,
                                (10, h - 10)
                                , cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    log_message = "Current {} count: {}" \
                        .format(videoCapResult.req_label,
                                videoCapResult.last_correct_count)
                    cv2.putText(videoCapResult.cur_frame, log_message,
                                (10, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)
                    cv2.putText(videoCapResult.cur_frame,
                                'Infer wait: %0.3fs' % (
                                    infer_duration.total_seconds()),
                                (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

                    # Display inferred frame and stats
                    stats = numpy.zeros((statsHeight, statsWidth, 1),
                                        dtype='uint8')
                    for i, log in enumerate(rolling_log):
                        cv2.putText(stats, log, (10, i * 20 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                    cv2.imshow(STATS_WINDOW_NAME, stats)
                    if idx == 0:
                        stats = cv2.cvtColor(stats, cv2.COLOR_GRAY2BGR)
                        statsVideo.write(stats)
                    end_time = datetime.datetime.now()
                    cv2.putText(videoCapResult.cur_frame, 'FPS: %0.2fs'
                                % (1 / (
                                end_time - videoCapResult.start_time).total_seconds()),
                                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.imshow(videoCapResult.cap_name,
                               videoCapResult.cur_frame)
                    videoCapResult.start_time = datetime.datetime.now()
                    videoCapResult.video.write(videoCapResult.cur_frame)

            # Wait if necessary for the required time
            key = cv2.waitKey(1)

            # Esc key pressed
            if key == 27:
                cv2.destroyAllWindows()
                infer_network.clean()
                print("Finished")
                return
            # Tab key pressed
            if key == 9:
                is_async_mode = not is_async_mode
                print("Switched to {} mode".format(
                    "async" if is_async_mode else "sync"))

            if is_async_mode:
                # Swap infer request IDs
                cur_request_id, next_request_id = next_request_id, cur_request_id

            # Loop video if LOOP_VIDEO = True and input isn't live from USB camera
            if LOOP_VIDEO and not videoCapInfer.is_cam:
                vfps = int(round(videoCapInfer.cap.get(cv2.CAP_PROP_FPS)))
                # If a video capture has ended restart it
                if (videoCapInfer.cur_frame_count >
                        videoCapInfer.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        - int(round(vfps / minFPS))):
                    videoCapInfer.cur_frame_count = 0
                    videoCapInfer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if False not in no_more_data:
            break

    infer_network.clean()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
