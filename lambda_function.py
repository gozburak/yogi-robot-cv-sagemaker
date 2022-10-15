import awscam
import cv2
import greengrasssdk
import os
from local_display import LocalDisplay
from dynamodb import Dynamodb
from time import time
from mxnet import nd
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import boto3
import uuid

# The below are the main definitions for our human boday parts

# WE NEED TO REPLACE THSI WITH A DATABASE


# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase


# This build the joints object


# This builds the body parts


# This calculates orientations


# This calculates deviations from the ground truth


# These are the classes used to calculate angles, locations and other metrics specific to yoga usecase.
# ... 'create_json' replaces the original 'update_state_json' function
from posture_analysis import create_json


# This is the original lambda handler func. - not touched
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    print("hello world")
    return



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

    dynamodb = Dynamodb()

    MODEL_PATH = '/opt/awscam/artifacts/'

    # Load the models here
    people_detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root=MODEL_PATH)
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True, root=MODEL_PATH)
    people_detector.reset_class(["person"], reuse_weights=['person'])
    currentlyInSession = False  # Assume we start with no person in front of DeepLens
    ############ Detect Person from Object Detection
    # This object detection model is implemented as single shot detector (ssd), since
    # the number of labels is small we create a dictionary that will help us convert
    # the machine labels to human readable labels.
    model_type = 'ssd'
    output_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
                  7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dinning table',
                  12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
                  16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train',
                  20: 'tvmonitor'}
    # The sample projects come with optimized artifacts, hence only the artifact
    # path is required.
    model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
    # Load the model onto the GPU.
    model = awscam.Model(model_path, {'GPU': 1})
    # Set the threshold for detection
    detection_threshold = 0.25
    # The height and width of the training set images
    input_height = 300
    input_width = 300
    person = 15
    detection_threshold = 0.5
    # Do inference until the lambda is killed.
    while True:
        print('===========================sarting new loop:')
        loopstart = time()
        # Get a frame from the video stream
        start = time()
        ret, frame = awscam.getLastFrame()
        ##### code to detect person
        frame_resize = cv2.resize(frame, (input_height, input_width))
        # Run the images through the inference engine and parse the results using
        # the parser API, note it is possible to get the output of doInference
        # and do the parsing manually, but since it is a ssd model,
        # a simple API is provided.
        parsed_inference_results = model.parseResult(model_type, model.doInference(frame_resize))
        personcount = 0
        for obj in parsed_inference_results[model_type]:
            if (((obj['label']) == person)and(obj['prob'] > detection_threshold)):
                personcount += 1
        if (personcount == 0):
            print("No person detected")
            continue
        print("Number of people detected: " + str(personcount))

        ##### end code to detect person
        # frame = cv2.resize(frame, (380, 672))
        posture = dynamodb.getPosture()  # get current posture from dynamodb
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
            dynamodb.setStatus('False')
            currentlyInSession = False
            session = None
            local_display.reset_frame_data()
            continue  # do not process further
        # if it comes here, then a valid person has been detected
        print('person detected)')
        if (currentlyInSession == False):
            currentlyInSession = True
            session = getNewSession()
            print("New session created" + session)
            dynamodb.setStatus('True')  # update dynamodb

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