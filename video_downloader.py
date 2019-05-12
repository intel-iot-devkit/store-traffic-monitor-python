#!/usr/bin/python3
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

import sys
import os

def progress(blockNum, blockSize, totalSize):
    read = blockNum * blockSize
    if totalSize > 0:
        percent = read * 100 / totalSize
        s = "\r%5.0f%% " % (percent)
        sys.stderr.write(s)
        if read >= totalSize:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("read %d\n" % (read,))

urls = ['https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4',
        'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4',
        'https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bottle-detection.mp4']

for url in urls:
    filename = url[url.rfind("/") + 1: ]
    print ('Downloading ' + filename)
    if sys.version_info.major > 2:
        import urllib.request
        urllib.request.urlretrieve(url, os.path.join("resources", filename), progress)
    else:
        import urllib
        urllib.urlretrieve(url, os.path.join("resources", filename), progress)
