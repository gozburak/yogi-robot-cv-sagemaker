import datetime
import json
import math

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
session = None  # Session id is null because no person is present unless camera detects person.
ground_truth_angles = {'Right Lower Arm': {'GT Angle': 0.0},
                       'Left Lower Arm': {'GT Angle': 0.0},
                       'Right Upper Arm': {'GT Angle': -10.0},
                       'Left Upper Arm': {'GT Angle': 10.0},
                       'Right Thigh': {'GT Angle': 60},
                       'Left Thigh': {'GT Angle': -60},
                       'Hips': {'GT Angle': 0.0},
                       'Shoulders': {'GT Angle': 0.0}}


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


def get_joints_params(allJoints):
    paramsJoints = []
    for allJoints_pp in allJoints:
        paramsJoints_pp = {}
        for joint_name in keys_Joints:
            joint = allJoints_pp[joint_name]
            paramsJoints_pp.update({joint_name: {"Coords": joint.coords, "Conf": joint.conf}})
        paramsJoints.append(paramsJoints_pp)
    return paramsJoints


def get_bodyparts_params(allBodyparts):
    paramsBodyparts = []
    for allBodyparts_pp in allBodyparts:
        paramsBodyparts_pp = {}
        for bodypart_name in BodyForm:
            body = allBodyparts_pp[bodypart_name]
            paramsBodyparts_pp.update({bodypart_name: {"Angle": body.orient, "Conf": body.conf}})
        paramsBodyparts.append(paramsBodyparts_pp)
    return paramsBodyparts


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


def calculate_deviations(paramsBodyparts):
    deviations = []
    for paramsBodyparts_pp in paramsBodyparts:
        deviations_pp = {}
        for bodypart_name, data in paramsBodyparts_pp.items():
            diff = data['Angle'] - ground_truth_angles[bodypart_name]['GT Angle']
            deviations_pp.update({bodypart_name: {'Diff': diff}})
        deviations.append(deviations_pp)
    return deviations


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