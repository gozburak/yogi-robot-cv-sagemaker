from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2
from time import time
import uuid
import boto3
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
import greengrasssdk
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model, model_zoo
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from dynamodb import Dynamodb
from awscrt import mqtt
import sys
import threading
from uuid import uuid4
import json
from AvaMQTTHelper  import AvaMQTTHelper
from python_settings import settings
import settings as my_local_settings
from posture_analysis import PostureAnalysis
#Collect config settings:
settings.configure(my_local_settings) # configure() receivesge a python module
assert settings.configured # now you are set
DETECTOR=settings.DETECTOR
POSEMODEL=settings.POSEMODEL
POSEMODEL_SHA=settings.POSEMODEL_SHA
SERVER_SECRET_KEY = os.getenv("SECRET")
SERVER_PUBLIC_KEY = os.getenv("ACCESSKEY")
REGION_NAME = os.getenv('REGION_NAME')

received_count = 0
received_all_event = threading.Event()
#is_ci = cmdUtils.get_command("is_ci", None) != None

# Callback when connection is accidentally lost.
def on_connection_interrupted(connection, error, **kwargs):
    print("Connection interrupted. error: {}".format(error))


# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()

def sendMessage(message):
    avaMQTTHelper.publishMessage(message)


dynamodb = Dynamodb()
avaMQTTHelper = AvaMQTTHelper()
session = None  # Session id is null because no person is present unless camera detects person.
postureAnalysis = PostureAnalysis()

# This is the original lambda handler func. - not touched
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    print("hello world")
    return


def getNewSession():
    return str(uuid.uuid4())

def count_people(class_ids, scores, bounding_boxes,threshold=0.5):
    num_people = 0
    scores = scores.asnumpy().squeeze().astype(float)
    class_ids = class_ids.asnumpy().squeeze().astype(int)
    for index,i in enumerate(class_ids):
        if scores[index]>=threshold:
            num_people+=1
    return num_people

def find_wheelchair(chair_detector, x_chair, img_chair):
    class_IDs_chair, scores_chair, bounding_boxs_chair = chair_detector(x_chair)
    chair_score = max(scores_chair[0].reshape(-1))
    chair_detected = False
    confidence_chair = None
    if chair_score != -1 and chair_score > 0.01:
        chair_detected = True
    return chair_detected

def addFeedback(img, correct=True):
    fontcolor = None
    text = None
    if(correct==False):
        fontColor=(0,0,255)
        text="Please adjust pose"
    else:
        fontColor = (255, 255, 255)
        text = "Great Job !!!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, 50)
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
    return img

def tooManyPeople(img):
    fontColor=(0,255,0)
    text="Too many people"
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 50)
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
    return img
imageCounter = 0
def initialize(imageIndex):
    ImageArray = ["./images/yoga1.jpg", "./images/yoga2.jpg", "./images/yoga3.jpg"]
    count = len(ImageArray)
    index = imageIndex%count
    img = cv2.imread(ImageArray[index])
    img = mx.nd.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype('uint8')
    return img

def get_Image(cap,testImage= False):
    if testImage:
        img = cv2.imread('./images/testImage.jpg')
        return img
    else:
        ret, frame = cap.read()
        return frame


if __name__ == '__main__':
    lastActivityTime = time()-10
    img = initialize(imageCounter)
    imageCounter+=1
    cv_plot_image(img)
    ctx = mx.cpu()
    imageIndex =0
    #detector_name = "ssd_512_mobilenet1.0_coco"
    detector = get_model(DETECTOR, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
    
    #chair detector
    chair_detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    chair_detector.reset_class(["bicycle"], reuse_weights=['bicycle'])
    
    net = get_model(POSEMODEL, pretrained=POSEMODEL_SHA, ctx=ctx)
    session = None
    cap = cv2.VideoCapture(0)

    IDLE_SECONDS=4
    while(True): #Main loop
        currentposture = dynamodb.getPosture()
        frame = get_Image(cap, False)
        img = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
        x = x.as_in_context(ctx)
        class_IDs, scores, bounding_boxs = detector(x)

        #Wheelchair logic commented for now
        #Wheelchair detector
        #wheelchair_Flag = find_wheelchair(chair_detector, x_chair, img_chair)
        #if wheelchair_Flag:
        #    print("Wheelchair detected!")
        
        #count number of people
        peoplecount = count_people(class_IDs,scores,bounding_boxs)
        if (peoplecount ==0):
            print(time()-lastActivityTime)
            if (time()-lastActivityTime) > IDLE_SECONDS:
                img = initialize(imageCounter)
                imageCounter += 1
                session = None
                dynamodb.setStatus('False', session)
                dynamodb.setTooManyPeople('False')
            else:
                print("Waiting for Idle")


        if(peoplecount > 1):
            img = tooManyPeople(frame)
            dynamodb.setTooManyPeople('True')

        if(peoplecount==1):
            if (session == None):
                session = getNewSession()
            lastActivityTime = time()
            item = '{"statuskey":"personpresent", "statusvalue": { "presence": "True", "sessionID": "'+session+'" }}'
            dynamodb.setStatusek(json.loads(item))
            dynamodb.setTooManyPeople('False')
            pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs,
                                                               output_shape=(128, 96), ctx=ctx)
            if len(upscale_bbox) > 0:
                predicted_heatmap = net(pose_input)
                pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

                scale = 1.0 * img.shape[0] / scaled_img.shape[0]
                img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                        box_thresh=1, keypoint_thresh=0.3, scale=scale)
                poseCorrect, booleon, result_json = postureAnalysis.create_json(pred_coords, confidence, bounding_boxs, scores,
                                                          session,currentposture)
                cloud_output = '{"out":' + result_json + '}'
                #print(cloud_output)
                sendMessage(cloud_output)
                if poseCorrect:
                    addFeedback(img, True)
                else:
                    addFeedback(img, False)
        cv_plot_image(img)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()