# first get all includes
from local_display import LocalDisplay
from dynamodb import Dynamodb
from posture_analysis import PostureAnalysis
# next import statements for deeplens
import awscam
import greengrasssdk
import os
from time import time
import uuid
# mx, numpy
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
# gluoncv


import gluoncv as gcv

gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints, plot_image
import cv2
import mo

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

# This is the original function
def infinite_infer_run():
    #Set Environment variables:
    os.environ['MODEL_PATH'] = '/tmp'
    os.environ['MPLCONFIGDIR'] = '/tmp'
    os.environ['MXNET_HOME'] = '/tmp'
    # Create an IoT client for sending to messages to the cloud.
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

    # Create a local display instance that will dump the image bytes to a FIFO
    # file that the image can be rendered locally.
    local_display = LocalDisplay('480p')
    local_display.start()
    local_display.reset_frame_data()

    dynamodb = Dynamodb()
    postureAnalysis = PostureAnalysis()

    input_height = 480
    input_width = 640
    # Load the models here
    print('ctx = mx.cpu')
    ctx = mx.cpu()
    detector_name = "ssd_512_mobilenet1.0_coco"
    #print(detector_name)
    #print("About to do get_model")
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    #print('detector get_model')
    #print('Before mo.optimize')
    #error, model_path = mo.optimize(detector_name, input_width, input_height, platform="MXNet", aux_inputs=aux_inputs)
    #print("model path is :" + model_path)
    #print("model error is : {s}".format(error))

    # detector=awscam.Model(model_path, {"GPU": 1})
    print("loaded detector")
    detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
    detector.hybridize()
    print('detector reset class to person')

    estimator_name = 'simple_pose_resnet18_v1b'
    estimator_sha = 'ccd24037'
    estimators = get_model(estimator_name, pretrained=estimator_sha, ctx=ctx)
    # error, model_path = mo.optimize(estimator_name,input_width,input_height,platform="MXNet",aux_inputs=aux_inputs)
    # estimators=awscam.Model(model_path, {"GPU": 1})
    # estimators.hybridize()
    currentlyInSession = False  # Assume we start with no person in front of DeepLens
    ############ Detect Person from Object Detection
    # This object detection model is implemented as single shot detector (ssd), since
    # the number of labels is small we create a dictionary that will help us convert
    # the machine labels to human readable labels.
    model_type = 'ssd'
    # The sample projects come with optimized artifacts, hence only the artifact
    # path is required.
    model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
    # Load the model onto the GPU.
    model = awscam.Model(model_path, {'GPU': 1})
    # Set the threshold for detection
    detection_threshold = 0.55
    # The height and width of the training set images
    input_height = 300
    input_width = 300
    person = 15

    # Do inference until the lambda is killed.
    while True:
        print('===========================sarting new loop:')
        loopstart = time()
        # Get a frame from the video stream
        start = time()
        ret, frame = awscam.getLastFrame()
        print('-------------got last frame: {}s'.format(time() - start))

        ##### code to detect person
        start = time()
        frame_resize = cv2.resize(frame, (input_height, input_width))
        print('--------------cv2.resize: {}s'.format(time() - start))
        # Run the images through the inference engine and parse the results using
        # the parser API, note it is possible to get the output of doInference
        # and do the parsing manually, but since it is a ssd model,
        # a simple API is provided.

        start = time()
        parsed_inference_results = model.parseResult(model_type, model.doInference(frame_resize))
        print('--------------model.parseResult->model.doinference: {}s'.format(time() - start))

        personcount = 0
        for obj in parsed_inference_results[model_type]:
            if (((obj['label']) == person) and (obj['prob'] > detection_threshold)):
                personcount += 1
        print("Number of people detected: " + str(personcount))
        if (personcount == 0):
            print("No person detected")
            dynamodb.setStatus('False')
            currentlyInSession = False
            session = None
            local_display.reset_frame_data()
            continue  # do not process further
        if (personcount > 1):
            print("too many persons detected")
            dynamodb.setStatus('False')
            currentlyInSession = False
            session = None
            local_display.reset_frame_data()
            continue  # do not process further
        # if it comes here, exactly one person is detected
        print('Person detected)')
        if (currentlyInSession == False):
            currentlyInSession = True
            session = getNewSession()
            print("New session created" + session)
            dynamodb.setStatus('True')  # update dynamodb
        ##### end code to detect person
        posture = dynamodb.getPosture()  # get current posture from dynamodb
        # cv2.putText(image, posture, org, font,
        #           fontScale, color, thickness, cv2.LINE_AA)

        start = time()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        print('--------------cv2.cvtColor: {}s'.format(time() - start))

        start = time()
        x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=350)
        print('--------------transforms.presets: {}s'.format(time() - start))

        start = time()
        x = x.as_in_context(ctx)
        print('--------------x.as_incontext: {}s'.format(time() - start))

        start = time()
        class_IDs, scores, bounding_boxs = detector(x)
        print('--------------detector(s): {}s'.format(time() - start))

        start = time()
        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                           output_shape=(128, 96), ctx=ctx)
        print('--------------detector_to_simple_pose: {}s'.format(time() - start))
        # Should never happen - but...
        if pose_input is None:
            continue

        start = time()
        predicted_heatmap = estimators(pose_input)
        print('--------------predicted heatmap: {}s'.format(time() - start))

        start = time()
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        print('--------------heatmap_to_coord: {}s'.format(time() - start))
        scale = 1.0 * frame.shape[0] / scaled_img.shape[0]
        print("Scale is: {}s".format(scale))

        start = time()
        img = cv_plot_keypoints(frame.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.25, scale=scale)
        print('--------------rendered plot: {}s'.format(time() - start))

        start = time()
        local_display.set_frame_data(img)
        print('--------------updated local display: {}s'.format(time() - start))

        # Creating JSON
        start = time()
        result_json = postureAnalysis.create_json(pred_coords, confidence, bounding_boxs, scores, client, iot_topic,
                                                  session)
        print('--------------Created JSON: {}s'.format(time() - start))

        start = time()
        deviationExceeded = True
        addMessage(img,deviationExceeded)
        local_display.set_frame_data(img)
        print('--------------Add Message and Render: {}s'.format(time() - start))

        # Now we publish the iot topic
        start = time()
        result_json = postureAnalysis.create_json(pred_coords, confidence, bounding_boxs, scores, client, iot_topic,
                                                  session)
        cloud_output = '{"out":' + result_json + '}'
        client.publish(topic=iot_topic, payload=cloud_output)
        print('--------------published to IOT topic: {}s'.format(time() - start))
        print('===========================.Entire loop took{}s'.format(time() - loopstart))


infinite_infer_run()