import datetime
import json
import math
import numpy as np


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


class PostureAnalysis():
    # The below are the main definitions for our human boday parts
    keys_Joints = ['Nose', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Right Shoulder',
                   'Left Shoulder', 'Right Elbow', 'Left Elbow', 'Right Wrist', 'Left Wrist', 'Right Hip',
                   'Left Hip', 'Right Knee', 'Left Knee', 'Right Ankle', 'Left Ankle']
    
    Parts_needed = ['Lower Arm', 'Upper Arm', 'Thigh']
    Thresholds = {'Kink': 25,
                  'MinConf': 0.35,
                  'Diff': 0.15}
 

    BodyForm = {
        'Right Lower Arm': ['Right Shoulder', 'Right Elbow'],
        'Left Lower Arm': ['Left Shoulder', 'Left Elbow'],
        'Right Upper Arm': ['Right Elbow', 'Right Wrist'],
        'Left Upper Arm': ['Left Elbow', 'Left Wrist'],
        'Right Thigh': ['Right Hip', 'Right Knee'],
        'Left Thigh': ['Left Hip', 'Left Knee'],
        'Right Leg': ['Right Knee', 'Right Ankle'],
        'Left Leg': ['Left Knee', 'Left Ankle'],
        'Hips': ['Left Hip', 'Right Hip'],
        'Shoulders': ['Left Shoulder', 'Right Shoulder'],
    }

    # WE NEED TO REPLACE THSI WITH A DATABASE
    ground_truth_angles = {'chair': {
        'Right Lower Arm': {'GT Angle': -50.0, 'Threshold': 10},
        'Left Lower Arm': {'GT Angle': -50.0, 'Threshold': 10},
        'Right Upper Arm': {'GT Angle': -60.0, 'Threshold': 15},
        'Left Upper Arm': {'GT Angle': -60.0, 'Threshold': 15},
        'Right Thigh': {'GT Angle': 40.0, 'Threshold': 10},
        'Left Thigh': {'GT Angle': 40.0, 'Threshold': 10},
        'Left Leg': {'GT Angle': -65.0, 'Threshold': 35},
        'Right Leg': {'GT Angle': -65.0, 'Threshold': 35}}}

    # This build the joints object
    def get_joints_params(self, allJoints):
        paramsJoints = []
        for allJoints_pp in allJoints:
            paramsJoints_pp = {}
            for joint_name in self.keys_Joints:
                joint = allJoints_pp[joint_name]
                paramsJoints_pp.update({joint_name: {"Coords": joint.coords, "Conf": joint.conf}})
            paramsJoints.append(paramsJoints_pp)
        return paramsJoints

    # This builds the body parts
    def get_bodyparts_params(self, allBodyparts):
        paramsBodyparts = []
        for allBodyparts_pp in allBodyparts:
            paramsBodyparts_pp = {}
            for bodypart_name in PostureAnalysis.BodyForm:
                body = allBodyparts_pp[bodypart_name]
                paramsBodyparts_pp.update({bodypart_name: {"Angle": body.orient, "Conf": body.conf}})
            paramsBodyparts.append(paramsBodyparts_pp)
        return paramsBodyparts

    # This calculates orientations
    def build_body_from_joints(self, allJoints):
        allBodyparts = []
        for joint in allJoints:
            iter_body = {}
            for bodypart_name, joint_names in self.BodyForm.items():
                body = Bodypart(bodypart_name, joint[joint_names[0]], joint[joint_names[1]])
                body.get_metrics()
                iter_body.update({bodypart_name: body})
            allBodyparts.append(iter_body)
        return allBodyparts
    
    # This calculates deviations from the ground truth
    def calculate_deviations(self, pose, facing_direction, paramsBodyparts):
        deviations = []
        booleon = []
        excess_deviations = []
        confidence_threshold = self.Thresholds['MinConf']
        
        # Facing direction affects the ground truth
        if facing_direction == "Left Facing":
            adopter = -1
        else:
            adopter = 1
            
        # exception ahndling, our default pose is chair pose
        if pose == '':
            pose = 'chair'
        for paramsBodyparts_pp in paramsBodyparts:
            deviations_pp = {}
            booleon_pp = {}
            devi_excess_pp = {}
            for bodypart_name, GT_data in self.ground_truth_angles[pose].items():
                if paramsBodyparts_pp[bodypart_name]['Conf'] >= confidence_threshold:
                    diff = adopter*paramsBodyparts_pp[bodypart_name]['Angle'] - GT_data['GT Angle']
                    devi_excess = abs(diff) - GT_data['Threshold']
                    deviations_pp.update({bodypart_name: {'Diff': diff}})
                    if devi_excess < 0:
                        booleon_pp.update({bodypart_name: True})
                    else:
                        booleon_pp.update({bodypart_name: False})
                        directional_deviation = np.sign(diff)*devi_excess
                        devi_excess_pp.update({bodypart_name: directional_deviation})
            booleon.append(booleon_pp)
            deviations.append(deviations_pp)
            excess_deviations.append(devi_excess_pp)
        return deviations, excess_deviations, booleon

    
    #get direction of the face - orientation
    def get_facing_direction(self, allJoints):
        diff_conf_threshold = self.Thresholds['Diff']
        for allJoints_pp in allJoints:
            right_joint_list = [allJoints_pp["Right Elbow"]['Conf'],allJoints_pp["Right Wrist"]['Conf'],
              allJoints_pp["Right Knee"]['Conf'], allJoints_pp["Right Ankle"]['Conf'],  allJoints_pp["Right Ear"]['Conf']]
            left_joint_list = [allJoints_pp["Left Elbow"]['Conf'],allJoints_pp["Left Wrist"]['Conf'],
              allJoints_pp["Left Knee"]['Conf'], allJoints_pp["Left Ankle"]['Conf'],  allJoints_pp["Left Ear"]['Conf']]
            avg_right_joint_conf = sum(right_joint_list)/len(right_joint_list)
            avg_left_joint_conf = sum(left_joint_list)/len(left_joint_list)
            difference = avg_right_joint_conf - avg_left_joint_conf

            if difference > diff_conf_threshold:
                side = "Left Facing"
            elif difference < -diff_conf_threshold:
                side = "Right Facing"
            else:
                side = "Probably Front"
        return side
    
    # Analysis of difference
    def analyze_result(self,booleon):
        # The below confidence
        all_check = []
        for Bodypart_required in self.Parts_needed:
            check = False
            for Bodypart_detected, value in booleon[0].items():
                if Bodypart_required in Bodypart_detected:
                    check = value
                    break
            all_check.append(check)
        
        return all(all_check)
    
    # Get directions
    def get_direction(self,excess_deviations):
        kink_threshold = self.Thresholds['Kink']
        directions_dict = {}
        for bodypart, excess in excess_deviations[0].items():
            direction = None
            if 'Arm' in bodypart and excess < 0:
                direction = 'Arms Down'
            elif 'Arm' in bodypart and excess > 0:
                direction = 'Arms Up'
            elif 'Thigh' in bodypart and excess < 0:
                direction = 'Sit Up'
            elif 'Thigh' in bodypart and excess > 0:
                direction = 'Sit Down'
            if direction:
                directions_dict.update({direction:abs(excess)})
            
        
        try:
            left_arm_kink = abs(excess_deviations[0]['Left Upper Arm'] - excess_deviations[0]['Left Lower Arm'])
        except:
            left_arm_kink = None
        try:
            right_arm_kink = abs(excess_deviations[0]['Right Upper Arm'] - excess_deviations[0]['Right Lower Arm'])
        except:
            right_arm_kink = None

        if (left_arm_kink and left_arm_kink > kink_threshold) or (right_arm_kink and right_arm_kink > kink_threshold):
            direction = 'Arms Straight'
            directions_dict.update({direction:sum(filter(None, [left_arm_kink,right_arm_kink]))})
        return directions_dict
    # These are the classes used to calculate angles, locations and other metrics specific to yoga usecase.
    
    def get_critical_direction(self, directions):
        max_direction_angle = 0
        critical_direction = ''
        for direction, angle in directions.items():
            if abs(angle) > max_direction_angle:
                max_direction_angle = abs(angle)
                critical_direction = direction
        return critical_direction
    
    
    # ... 'create_json' replaces the original 'update_state_json' function
    def create_json(self, pred_coords, confidence, bboxes, scores, session, pose):

        # numpy is needed for better calculation of metrics
        pred_coords_clean = pred_coords.asnumpy()
        confidence_clean = confidence.asnumpy()
        bounding_boxs_clean = bboxes.asnumpy()
        scores_clean = scores.asnumpy()
        # The following identifies the joints and body part dictionaries for the picture
        allJoints = [{name: Joint(name, coord[0], coord[1], conf[0]) for name, coord, conf in
                      zip(self.keys_Joints, coord_per_person, conf_per_person)} for coord_per_person, conf_per_person in zip(pred_coords_clean, confidence_clean)]
        allBodyParts = self.build_body_from_joints(allJoints)

        # We also transfer the bounding box
        keys_BoundingBox = ["X0", "Y0", "Width", "Height"]
        resBoundingBox = [{"BoundingBox": {"Coords": {key: float(value) for key, value in
                                                      zip(keys_BoundingBox, Boundingbox_per_person[0])},
                                           "Confidence": float(conf_per_person[0][0])}}
                          for Boundingbox_per_person, conf_per_person in zip(bounding_boxs_clean, scores_clean)]

        # Let's calculate the joint parts and body angles
        paramsJoints = self.get_joints_params(allJoints)
        paramsBodyparts = self.get_bodyparts_params(allBodyParts)

        #Let determine side
        facing_direction = self.get_facing_direction(paramsJoints)

        # Time stamp is added to the output
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculating deviations
        deviations, excess_deviations, booleon = self.calculate_deviations(pose, facing_direction, paramsBodyparts)
        
        # get all directions
        directions = self.get_direction(excess_deviations)
        
        #Pick the critical direction
        critical_direction = self.get_critical_direction(directions)
        
        # This is the schema for our JSON
        res = [{"SessionID": session, "Timestamp": time_now, "PersonID": person, **Bbox, "Joints": Joint,
                "Bodyparts": Body,
                "Deviations": Devi, 'Direction':critical_direction}
               for person, (Bbox, Joint, Body, Devi) in
               enumerate(zip(resBoundingBox, paramsJoints, paramsBodyparts, deviations))]
        
        final_boolean = self.analyze_result(booleon)
        
        return final_boolean, booleon, json.dumps(res)
