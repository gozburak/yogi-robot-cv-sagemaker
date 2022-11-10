# first get all includes
from posture_analysis import PostureAnalysis
# next import statements for deeplens
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

#Set Environment variables:
os.environ['MODEL_PATH'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['MXNET_HOME'] = '/tmp'
# Create an IoT client for sending to messages to the cloud.



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

# Do inference until the lambda is killed.

print('===========================sarting new loop:')
loopstart = time()
frame = cv2.imread('Chair-Pose.png')
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
                        box_thresh=0.5, keypoint_thresh=0.20, scale=scale)
print('--------------rendered plot: {}s'.format(time() - start))

start = time()
cv2.imshow( 'image', img)
print('--------------updated local display: {}s'.format(time() - start))


start = time()
deviationExceeded = True
addMessage(img,deviationExceeded)
cv2.imshow( 'image', img)
print('--------------Add Message and Render: {}s'.format(time() - start))

# Now we publish the iot topic
print('===========================.Entire loop took{}s'.format(time() - loopstart))

cv2.waitKey(0)
