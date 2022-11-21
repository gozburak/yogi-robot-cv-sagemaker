DETECTOR='ssd_512_mobilenet1.0_coco'
POSEMODEL='alpha_pose_resnet101_v1b_coco'
# POSEMODEL_SHA='ccd24037' <--- Use for simple pose detector
POSEMODEL_SHA= True #Default

ENDPOINT = 'a2h8id2qn57my5-ats.iot.us-east-1.amazonaws.com'
KEY = './connectionproperties/privKey.key'
CERT = './connectionproperties/thingCert.crt'
ROOTCERT = './connectionproperties/rootCA.pem'
TOPIC = 'yogabot/stream'
CLIENT_ID = 'buraklaptop'
