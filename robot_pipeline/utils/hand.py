# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on deep-high-resolution-net.pytorch.
# (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import argparse
import cv2
import sys
import numpy as np

sys.path.append('/home/fanfan/Documents/yolo-hand-detection/')

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("/home/fanfan/Documents/yolo-hand-detection/models/cross-hands.cfg",
                "/home/fanfan/Documents/yolo-hand-detection/models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-tiny-prn.cfg",
                "/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-yolov4-tiny.cfg",
                "/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-tiny.cfg",
                "/home/fanfan/Documents/yolo-hand-detection/models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)


def hand_predict(self):
    conf_sum = 0
    detection_count = 0

    mat = self.cv_image
    width, height, inference_time, self.results = yolo.inference(mat)

    print(np.array(self.results).shape)
    print(self.results)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 848, 640)

    ind = 0

    for detection in self.results:
        if ind >= 0:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            conf_sum += confidence
            detection_count += 1

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, color, 1)

            print("%s with %s confidence" % (name, round(confidence, 2)))

        ind += 1

    # show the output image
    cv2.imshow('image', mat)
    cv2.waitKey(1)
    print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
