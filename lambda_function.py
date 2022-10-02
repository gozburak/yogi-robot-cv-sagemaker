import awscam
import json
import cv2
import greengrasssdk
import os
from local_display import LocalDisplay
import numpy as np
from time import time
# from matplotlib import pyplot as plt
# Pls see requirements.txt
from mxnet import nd
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from math import atan2, degrees
import datetime
import boto3
import csv

s3 = boto3.client("s3")
S3_BUCKET = '<YOUR_S3_BUCKET_NAME>'
DATA_BUFFER = 60 # Not used but kept for future uses.

# The below are the main definitions for our human boday parts
keys_Joints = ['Nose', 'Right Eye', 'Left Eye', 'Right Ear','Left Ear','Right Shoulder',
         'Left Shoulder','Right Elbow','Left Elbow','Right Wrist','Left Wrist','Right Hip',
         'Left Hip','Right Knee','Left Knee','Right Ankle','Left Ankle']

BodyForm = {
    'Right Lower Arm': ['Right Shoulder','Right Elbow'],
    'Left Lower Arm': ['Left Shoulder','Left Elbow'],
    'Right Upper Arm': ['Right Elbow', 'Right Wrist'],
    'Left Upper Arm': ['Left Elbow', 'Left Wrist'],
    'Right Thigh': ['Right Hip', 'Right Knee'],
    'Left Thigh': ['Left Hip', 'Left Knee'],
    'Hips': ['Left Hip', 'Right Hip'],
    'Shoulders': ['Left Shoulder', 'Right Shoulder'],
    }

# This is the original lambda handler func. - not touched
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    print("hello world")
    return


# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase
class Joint():
    def __init__(self, name, x,y,conf):
        self.name = name
        self.coords = (float(x),float(y))
        self.conf = float(conf)
class Bodypart():
    def __init__(self, name, jointA, jointB):
        self.jointA = jointA
        self.jointB = jointB
        self.name = name
    def get_metrics(self):
        self.vec = [b - a for a,b in zip(self.jointA.coords, self.jointB.coords)]
        self.length = float(math.hypot(self.vec[1], self.vec[0]))
        if self.vec[0] != 0:
            self.orient = float(math.degrees(math.atan(self.vec[1]/self.vec[0])))
        else:
            self.orient = 90.0
        self.conf = float(self.jointA.conf * self.jointA.conf)
        
# This calculates the locations of the joints
def calculate_joints(allJoints):
    paramsJoints = []
    for allJoints_pp in allJoints:
        paramsJoints_pp = {}
        for joint_name in keys_Joints:
            joint = allJoints_pp[joint_name]
            paramsJoints_pp.update({joint_name:{"Coords":joint.coords, "Conf":joint.conf}})
        paramsJoints.append(paramsJoints_pp)
    return paramsJoints

# This calculates the angles of the body parts
def calculate_bodyparts(allBodyparts):
    paramsBodyparts = []
    for allBodyparts_pp in allBodyparts:
        paramsBodyparts_pp = {}
        for bodypart_name in BodyForm:
            body = allBodyparts_pp[bodypart_name]
            paramsBodyparts_pp.update({bodypart_name:{"Angle":body.orient, "Conf":body.conf}})
        paramsBodyparts.append(paramsBodyparts_pp)
    return paramsBodyparts
        
# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase.
#... 'create_json' replaces the original 'update_state_json' function
def create_json(pred_coords, confidence, bboxes, scores, client, iot_topic)

    # numpy is needed for better calculation of metrics
    pred_coords_clean = pred_coords.asnumpy()
    confidence_clean = confidence.asnumpy()
    bounding_boxs_clean = bboxes.asnumpy()
    scores_clean = scores.asnumpy()
    
    # The following identifies the joints and body part dictionaries for the picture
    allJoints = [{name:Joint(name,coord[0],coord[1],conf[0]) for name,coord,conf in 
                  zip(keys_Joints,coord_per_person,conf_per_person)} for coord_per_person, conf_per_person in 
                 zip(pred_coords_clean,confidence_clean)]
    allBodyParts = [{bodypart_name:Bodypart(bodypart_name, joint[joint_names[0]], joint[joint_names[1]]) 
                 for bodypart_name, joint_names in BodyForm.items()} for joint in allJoints]
       
    # We also transfer the bounding box
    keys_BoundingBox = ["X0","Y0", "Width", "Height"]
    resBoundingBox = [{"BoundingBox":{"Coords":{key:float(value) for key,value in 
                        zip(keys_BoundingBox,Boundingbox_per_person[0])},"Confidence":float(conf_per_person[0][0])}}
                          for Boundingbox_per_person, conf_per_person in zip(bounding_boxs_clean,scores_clean)]

    # Let's calculate the joint parts and body angles
    paramsJoints = calculate_joints(allJoints)       
    paramsBodyparts = calculate_bodyparts(allBodyparts)
    
    #Time stamp is added to the output
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # This is the schema for our JSON
    res = [{"Timestamp":time_now,"PersonID":person,**Bbox, "Joints":Joint, "Bodyparts":Body} for person,(Bbox,Joint, Body)
       in enumerate(zip(resBoundingBox, paramsJoints, paramsBodyparts))]
    
    return json.dumps(res)
    
    

# This is the original function
def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    # Create an IoT client for sending to messages to the cloud.
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

    # Create a local display instance that will dump the image bytes to a FIFO
    # file that the image can be rendered locally.
    local_display = LocalDisplay('480p')
    local_display.start()

    MODEL_PATH = '/opt/awscam/artifacts/'

    # Load the models here
    people_detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root=MODEL_PATH)
    # pose_net = model_zoo.get_model('simple_pose_resnet18_v1b')
    # pose_net.load_parameters(MODEL_PATH+'simple_pose_resnet18_v1b-0000.params')
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True, root=MODEL_PATH)
    people_detector.reset_class(["person"], reuse_weights=['person'])
    sec_mark = 1
    while True:
        # Get a frame from the video stream
        start = time()
        _, frame = awscam.getLastFrame()
        frame = cv2.resize(frame, (380, 672))
        print('Load frame: {}s'.format(time() - start))
        print(frame.shape)
        start = time()
        x, img = data.transforms.presets.ssd.transform_test(nd.array(frame), short=256)
        print('Shape of pre-processed image:', x.shape)

        print('Load image: {}s'.format(time() - start))

        start = time()
        class_ids, scores, bboxes = people_detector(x)

        print('Detection: {}s'.format(time() - start))

        pose_input, upscale_bbox = detector_to_simple_pose(img, class_ids, scores, bboxes)

        if pose_input is None:
            print("no person detected")
            continue

        print(pose_input.shape)

        start = time()
        predicted_heatmap = pose_net(pose_input)
        
        
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        print('Pose Detect: {}s'.format(time() - start))

        # Creating JSON
        print("creating json")
        json = create_json(pred_coords, confidence, bboxes, scores, client, iot_topic)
        

infinite_infer_run()