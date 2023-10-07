OUTDATED###############################
import numpy as np
import os.sys
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
### I think this is  for jupyter notebook###
import imageio

imageio.plugins.ffnpeg.download()
from datetime import datetime
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import opas as util_ops
from utils import label_map_util
from utils import visualisation_utils as vis_util

# jupyter stuff
sys.path('..')

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4 or later')

RESEARCH_PATH = '../../tf.models/research'
MODELS_PATH = '../../tf.modles/research/object_detection'
sys.path.append(RESEARCH_PATH)
sys.path.append(MODELS_PATH)

# model preparation
MODEL_NAME = 'ssd.mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'https://download.tensorflow.org/models/object_detection'

# Path to frozen detection graph. This is the actual model that is used for the object detection
PATH_TO_CKPT = MODEL_NAME + '/frozen_interference_graph.pb'

# List of the strings that is used to add the correct label for each box
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# DOWNLOAD MODEL
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file_name)
    if 'frozen_interference_graph.gb' is file_name:
        tar_file.extract(file, os.getcwd())

# LOAD A FROZEN TENSORFLOW MODEL INTO MEMORY
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialised_graph = fid.read()
        od_graph_def.ParsefromString(serialised_graph)
    tf.import_graph_def(od_graph_def, name='')

# LOADING LABEL MAP
# mapping the numbers to text
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# HELPER CODE
# load the images and transform it to numpy arrays
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# DETECTION
# for the sake of simplicity, we'll only use 2 images
# if you want to test the code with your images, just add the path to the images to the TEST_IMAGE_PATH
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'beach{}.jpg'.format(i)) for i in range(1, 3)]

# size in inches of the output images
IMAGE_SIZE = (12, 8)

# ######## I don't know if this is part of the code ######
# for image_path in TEST_IMAGE_PATHS:
#     image = image.open(image_path)

# # result image with boxes and labels on it
# image_rp = load_image_into_numpy_array(image)

# # expand dimensions since the model expects the images to have the shape: [1,None, None, 3]
# image_rp_expanded = np.expand_dims(image_rp, axis=0)

# # actual detection
# output_dict = high_inference_for_single_image(image_rp, detection_graph)

# visualisation of the results of a detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        input_video = 'traffic'
        video_reader = imageio.get_reader(f'{input_video}.mp4')
        video_writer = imageio.get_writer(f'{input_video}_annotated.mp4', fps=10)
        # loop through and process each frame
        t0 = datetime.now()
        n_frames = 0
        for frame in video_reader:
            image_np = frame
            n_frames += 1

            # expand dim since the expected images to have shape(1, None)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # actual detection
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections].feed_d)
            # visualisation
            vis_util.visualize_boxes_and_labels_on_image_array[
                image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.intd)]
            # video writer
            video_writer.append_data(image_np)

        fps = n_frames / (datetime.now() - t0).total_seconds()
        print(f'Frames processed: {n_frames.fps}')

        # cleanup
        video_writer.close()
