from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2
import uuid

import boto3
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

from python_settings import settings
import settings as my_local_settings

#Collect config settings:
settings.configure(my_local_settings) # configure() receives a python module
assert settings.configured # now you are set
DETECTOR=settings.DETECTOR
POSEMODEL=settings.POSEMODEL
POSEMODEL_SHA=settings.POSEMODEL_SHA
SERVER_SECRET_KEY = settings.AWS_SERVER_SECRET_KEY
SERVER_PUBLIC_KEY = settings.AWS_SERVER_PUBLIC_KEY
REGION_NAME = settings.REGION_NAME

#Initialize dynamodb
dynamodb = boto3.client('dynamodb',
                        aws_access_key_id=SERVER_PUBLIC_KEY,
                        aws_secret_access_key=SERVER_SECRET_KEY,
                        region_name=REGION_NAME
                        )

session = None  # Session id is null because no person is present unless camera detects person.


# This is the original lambda handler func. - not touched
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    print("hello world")
    return


def getNewSession():
    return str(uuid.uuid4())

def addMessage(img, correct=True):
    # Using cv2.imread() method
    fontcolor = None
    text = None
    if(correct==True):
        fontColor=(0,0,255)
        text="Please adjust pose"
    else:
        fontColor = (255, 255, 255)
        text = "Great Job !!!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, 100)
    fontScale = 2
    thickness = 3
    lineType = 2
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)




def keypoint_detection(img, detector, pose_net, ctx=mx.cpu()):

    return img

if __name__ == '__main__':
    ctx = mx.cpu()
    detector_name = "ssd_512_mobilenet1.0_coco"
    DETECTOR = get_model(detector_name, pretrained=True, ctx=ctx)
    DETECTOR.reset_class(classes=['person'], reuse_weights={'person': 'person'})
    net = get_model(POSEMODEL, pretrained=POSEMODEL_SHA, ctx=ctx)

    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus

    while(True): #Main loop
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
        x = x.as_in_context(ctx)
        class_IDs, scores, bounding_boxs = DETECTOR(x)

        pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs,
                                                           output_shape=(128, 96), ctx=ctx)
        if len(upscale_bbox) > 0:
            predicted_heatmap = net(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

            scale = 1.0 * img.shape[0] / scaled_img.shape[0]
            img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                    box_thresh=1, keypoint_thresh=0.3, scale=scale)
            addMessage(img, True)
        cv_plot_image(img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()