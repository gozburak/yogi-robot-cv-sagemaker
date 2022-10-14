import awscam
import json
import cv2
import greengrasssdk
import os
import math
from local_display import LocalDisplay
import numpy as np
from time import time
from mxnet import nd
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from math import atan2, degrees
import datetime
import boto3
from boto3.dynamodb.conditions import Key
import uuid

# The below are the main definitions for our human boday parts
keys_Joints = ['Nose', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Right Shoulder',
               'Left Shoulder', 'Right Elbow', 'Left Elbow', 'Right Wrist', 'Left Wrist', 'Right Hip',
               'Left Hip', 'Right Knee', 'Left Knee', 'Right Ankle', 'Left Ankle']

BodyForm = {
    'Right Lower Arm': ['Right Shoulder', 'Right Elbow'],
    'Left Lower Arm': ['Left Shoulder', 'Left Elbow'],
    'Right Upper Arm': ['Right Elbow', 'Right Wrist'],
    'Left Upper Arm': ['Left Elbow', 'Left Wrist'],
    'Right Thigh': ['Right Hip', 'Right Knee'],
    'Left Thigh': ['Left Hip', 'Left Knee'],
    'Hips': ['Left Hip', 'Right Hip'],
    'Shoulders': ['Left Shoulder', 'Right Shoulder'],
}
session =None# Session id is null because no person is present unless camera detects person.
# WE NEED TO REPLACE THSI WITH A DATABASE
ground_truth_angles = {'Right Lower Arm': {'GT Angle': -45.0},
                       'Left Lower Arm': {'GT Angle': -45.0},
                       'Right Upper Arm': {'GT Angle': -45.0},
                       'Left Upper Arm': {'GT Angle': -45.0},
                       'Right Thigh': {'GT Angle': 45.0},
                       'Left Thigh': {'GT Angle': 45.0},
                       'Hips': {'GT Angle': 0.0},
                       'Shoulders': {'GT Angle': 00.0}}


# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase
class Joint():
    def __init__(self, name, x, y, conf):
        self.name = name
        self.coords = (float(x), float(y))
        self.conf = float(conf)


class Bodypart():
    def __init__(self, name, jointA, jointB):
        self.jointA = jointA
        self.jointB = jointB
        self.name = name

    def get_metrics(self):
        self.vec = [b - a for a, b in zip(self.jointA.coords, self.jointB.coords)]
        self.length = float(math.hypot(self.vec[1], self.vec[0]))
        if self.vec[0] != 0:
            self.orient = float(math.degrees(math.atan(self.vec[1] / self.vec[0])))
        else:
            self.orient = 90.0
        self.conf = float(self.jointA.conf * self.jointA.conf)


# This build the joints object
def get_joints_params(allJoints):
    paramsJoints = []
    for allJoints_pp in allJoints:
        paramsJoints_pp = {}
        for joint_name in keys_Joints:
            joint = allJoints_pp[joint_name]
            paramsJoints_pp.update({joint_name: {"Coords": joint.coords, "Conf": joint.conf}})
        paramsJoints.append(paramsJoints_pp)
    return paramsJoints


# This builds the body parts
def get_bodyparts_params(allBodyparts):
    paramsBodyparts = []
    for allBodyparts_pp in allBodyparts:
        paramsBodyparts_pp = {}
        for bodypart_name in BodyForm:
            body = allBodyparts_pp[bodypart_name]
            paramsBodyparts_pp.update({bodypart_name: {"Angle": body.orient, "Conf": body.conf}})
        paramsBodyparts.append(paramsBodyparts_pp)
    return paramsBodyparts


# This calculates orientations
def build_body_from_joints(allJoints):
    allBodyparts = []
    for joint in allJoints:
        iter_body = {}
        for bodypart_name, joint_names in BodyForm.items():
            body = Bodypart(bodypart_name, joint[joint_names[0]], joint[joint_names[1]])
            body.get_metrics()
            iter_body.update({bodypart_name: body})
        allBodyparts.append(iter_body)
    return allBodyparts


# This calculates deviations from the ground truth
def calculate_deviations(paramsBodyparts):
    deviations = []
    for paramsBodyparts_pp in paramsBodyparts:
        deviations_pp = {}
        for bodypart_name, data in paramsBodyparts_pp.items():
            diff = data['Angle'] - ground_truth_angles[bodypart_name]['GT Angle']
            deviations_pp.update({bodypart_name: {'Diff': diff}})
        deviations.append(deviations_pp)
    return deviations


# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase.
# ... 'create_json' replaces the original 'update_state_json' function
def create_json(pred_coords, confidence, bboxes, scores, client, iot_topic, session):
    # numpy is needed for better calculation of metrics
    pred_coords_clean = pred_coords.asnumpy()
    confidence_clean = confidence.asnumpy()
    bounding_boxs_clean = bboxes.asnumpy()
    scores_clean = scores.asnumpy()

    # The following identifies the joints and body part dictionaries for the picture
    allJoints = [{name: Joint(name, coord[0], coord[1], conf[0]) for name, coord, conf in
                  zip(keys_Joints, coord_per_person, conf_per_person)} for coord_per_person, conf_per_person in
                 zip(pred_coords_clean, confidence_clean)]
    allBodyParts = build_body_from_joints(allJoints)

    # We also transfer the bounding box
    keys_BoundingBox = ["X0", "Y0", "Width", "Height"]
    resBoundingBox = [{"BoundingBox": {"Coords": {key: float(value) for key, value in
                                                  zip(keys_BoundingBox, Boundingbox_per_person[0])},
                                       "Confidence": float(conf_per_person[0][0])}}
                      for Boundingbox_per_person, conf_per_person in zip(bounding_boxs_clean, scores_clean)]

    # Let's calculate the joint parts and body angles
    paramsJoints = get_joints_params(allJoints)
    paramsBodyparts = get_bodyparts_params(allBodyParts)

    # Time stamp is added to the output
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculating deviations
    deviations = calculate_deviations(paramsBodyparts)

    # This is the schema for our JSON
    res = [{"SessionID": session, "Timestamp": time_now, "PersonID": person, **Bbox, "Joints": Joint, "Bodyparts": Body,
            "Deviations": Devi}
           for person, (Bbox, Joint, Body, Devi) in
           enumerate(zip(resBoundingBox, paramsJoints, paramsBodyparts, deviations))]

    return json.dumps(res)


# This is the original lambda handler func. - not touched
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    print("hello world")
    return

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
statustable = dynamodb.Table('statustable')

def getStatus():
    response = statustable.get_item(
        Key={
            'statuskey': 'personpresent'
        }
    )
    statusvalue = response['Item']['statusvalue']
    return statusvalue


def setStatus(value):
    statustable.update_item(
        Key={
            'statuskey': 'personpresent'
        },
        UpdateExpression='SET statusvalue = :statusvalue',
        ExpressionAttributeValues={
            ':statusvalue': value
        }
    )


def getPosture():
    response = statustable.get_item(
        Key={
            'statuskey': 'currentposture'
        }
    )
    statusvalue = response['Item']['statusvalue']
    return statusvalue


def setPosture(value):
    statustable.update_item(
        Key={
            'statuskey': 'currentposture',
        },
        UpdateExpression='SET statusvalue = :statusvalue',
        ExpressionAttributeValues={
            ':statusvalue': value
        }
    )
def getDeviationExceeded():
    response = statustable.get_item(
        Key={
            'statuskey': 'deviationexceeded'
        }
    )
    statusvalue = response['Item']['statusvalue']
    return statusvalue


def setDeviationExceeded(value):
    statustable.update_item(
        Key={
            'statuskey': 'deviationexceeded',
        },
        UpdateExpression='SET statusvalue = :statusvalue',
        ExpressionAttributeValues={
            ':statusvalue': value
        }
    )


def getNewSession():
    return str(uuid.uuid4())


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
    local_display.reset_frame_data()

    MODEL_PATH = '/opt/awscam/artifacts/'

    # Load the models here
    people_detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root=MODEL_PATH)
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True, root=MODEL_PATH)
    people_detector.reset_class(["person"], reuse_weights=['person'])
    currentlyInSession = False #Assume we start with no person in front of DeepLens

    while True:
        print('===========================sarting new loop:')
        loopstart = time()
        # Get a frame from the video stream
        start = time()
        ret, frame = awscam.getLastFrame()
        # frame = cv2.resize(frame, (380, 672))
        posture = getPosture()#get current posture from dynamodb
        # cv2.putText(image, posture, org, font,
        #           fontScale, color, thickness, cv2.LINE_AA)
        print('---------------Load frame: {}s'.format(time() - start))

        start = time()
        x, img = data.transforms.presets.ssd.transform_test(nd.array(frame), short=256)
        print('---------------transform_test: {}s'.format(time() - start))

        start = time()
        class_ids, scores, bboxes = people_detector(x)
        print('---------------people_detector: {}s'.format(time() - start))

        start = time()
        pose_input, upscale_bbox = detector_to_simple_pose(img, class_ids, scores, bboxes)
        print('---------------detector_to_simple_pose: {}s'.format(time() - start))

        if pose_input is None:
            print("no person detected")
            setStatus('False')
            currentlyInSession = False
            session = None
            local_display.reset_frame_data()
            continue # do not process further
        #if it comes here, then a valid person has been detected
        print('person detected)')
        if (currentlyInSession == False):
            currentlyInSession = True
            session = getNewSession()
            print("New session created" + session)
            setStatus('True') # update dynamodb

        start = time()
        predicted_heatmap = pose_net(pose_input)
        print('---------------heatmap: {}s'.format(time() - start))

        start = time()
        coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        print('--------------Coords from heatmap: {}s'.format(time() - start))

        start = time()
        ax = utils.viz.cv_plot_keypoints(img, coords, confidence, class_ids, bboxes, scores, box_thresh=0.5,
                                         keypoint_thresh=0.2)
        print('--------------rendered plot: {}s'.format(time() - start))

        start = time()
        local_display.set_frame_data(ax)
        print('--------------updated local display: {}s'.format(time() - start))
        # Creating JSON
        start = time()
        result_json = create_json(coords, confidence, bboxes, scores, client, iot_topic, session)
        print('--------------Created JSON: {}s'.format(time() - start))

        # Now we publish the iot topic
        start = time()
        cloud_output = '{"out":' + result_json + '}'
        client.publish(topic=iot_topic, payload=cloud_output)
        print('--------------published to IOT topic: {}s'.format(time() - start))
        print('===========================.Entire loop took{}s'.format(time() - loopstart))


infinite_infer_run()