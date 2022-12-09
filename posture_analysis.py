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
    Thresholds = {'Kink': 45,
                  'Facing Conf': 0.13,
                  'NonFacing Conf': 0.7,
                  'Diff': 0.15}
 
    # NOTICE: THE SIDES OF BELOW BODY PARTS ARE NOW DEFINED FOR THE ATTENDEE SIDE (not viewer side!!!)
    BodyForm = {
        'Right Lower Arm': ['Left Shoulder', 'Left Elbow'],
        'Left Lower Arm': ['Right Shoulder', 'Right Elbow'],
        'Right Upper Arm': ['Left Elbow', 'Left Wrist'],
        'Left Upper Arm': ['Right Elbow', 'Right Wrist'],
        'Right Thigh': ['Left Hip', 'Left Knee'],
        'Left Thigh': ['Right Hip', 'Right Knee'],
        'Hips': ['Left Hip', 'Right Hip']
    }

    ground_truth_angles = {'chair': {
        'Left Facing': {
        'Left Lower Arm': {'GT Angle': 40.0, 'Threshold': 25},
        'Right Lower Arm': {'GT Angle': 40.0, 'Threshold': 25},
        'Left Upper Arm': {'GT Angle': 50.0, 'Threshold': 25},
        'Right Upper Arm': {'GT Angle': 50.0, 'Threshold': 25},
        'Left Thigh': {'GT Angle': 60.0, 'Threshold': 20},
        'Right Thigh': {'GT Angle': 60.0, 'Threshold': 20}}}}

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
                paramsBodyparts_pp.update({bodypart_name: {"Angle": abs(body.orient), "Conf": body.conf}})
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

        facing_side = facing_direction.split()[0]

        # exception ahndling, our default pose is chair pose
        if pose == '':
            pose = 'chair'
        for paramsBodyparts_pp in paramsBodyparts:
            deviations_pp = {}
            booleon_pp = {}
            devi_excess_pp = {}
            for bodypart_name, GT_data in self.ground_truth_angles[pose][facing_direction].items():
                if facing_side in bodypart_name:
                    confidence_threshold = self.Thresholds['NonFacing Conf']
                else:
                    confidence_threshold = self.Thresholds['Facing Conf']
                if paramsBodyparts_pp[bodypart_name]['Conf'] >= confidence_threshold:
                    diff = paramsBodyparts_pp[bodypart_name]['Angle'] - GT_data['GT Angle']
                    devi_excess = abs(diff) - GT_data['Threshold']
                    deviations_pp.update({bodypart_name: {'Diff': diff}})
                    if devi_excess <= 0:
                        booleon_pp.update({bodypart_name: True})
                    else:
                        booleon_pp.update({bodypart_name: False})
                        directional_deviation = np.sign(diff)*devi_excess
                        devi_excess_pp.update({bodypart_name: directional_deviation})
                    # print(f"Partname: {bodypart_name}, Conf Thres: {confidence_threshold}, Angle: {paramsBodyparts_pp[bodypart_name]['Angle']}, Diff: {diff} ")
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
                direction = 'Arms Up'
            elif 'Arm' in bodypart and excess > 0:
                direction = 'Arms Down'
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
    
    def get_critical_directions(self, directions):
        max_upper_angle = 0
        upper_direction = ''
        max_lower_angle = 0
        lower_direction = ''
        for direction, angle in directions.items():
            if (abs(angle) > max_upper_angle) and ('Arms' in direction):
                max_upper_angle = abs(angle)
                upper_direction = direction
            elif (abs(angle) > max_lower_angle) and ('Sit' in direction):
                max_lower_angle = abs(angle)
                lower_direction = direction

        if max_upper_angle > max_lower_angle:
            body_direction = upper_direction
        else:
            body_direction = lower_direction

        critical_directions = {'CompleteBody':body_direction, 'TopBody': upper_direction, 'LowBody': lower_direction}
        return critical_directions
    
    
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
        #facing_direction = self.get_facing_direction(paramsJoints)
        facing_direction = "Left Facing" #CHEATING
        # Time stamp is added to the output
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculating deviations
        deviations, excess_deviations, booleon = self.calculate_deviations(pose, facing_direction, paramsBodyparts)

        # print(excess_deviations)
        # get all directions
        directions = self.get_direction(excess_deviations)
        # print(directions)

        #Pick the critical direction
        critical_directions = self.get_critical_directions(directions)
        
        critical_directions.update({'Boolean': booleon[0]})
        
        # This is the schema for our JSON
        res = [{"SessionID": session, "Timestamp": time_now, "PersonID": person, **Bbox, 
                "Bodyparts": Body,
                "Deviations": Devi, 'Direction':critical_directions}
               for person, (Bbox, Body, Devi) in
               enumerate(zip(resBoundingBox, paramsBodyparts, deviations))]


        # DEBUGGING!!!!
        #del res[0]['SessionID']
        #del res[0]['PersonID']
        #del res[0]['BoundingBox']
        #del res[0]['Joints']
        #res[0].update({"Facing":facing_direction})

        final_boolean = self.analyze_result(booleon)
        #if final_boolean:
            #print('YOU MADE IT BELOW!')

        return final_boolean, booleon, json.dumps(res)
