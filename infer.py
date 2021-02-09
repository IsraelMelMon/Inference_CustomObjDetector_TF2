# -*- coding: utf-8 -*-
"""infer.py

Program that runs a TF HUB pretrained object detection model 
(MELI - packages over a conveyor belt, using mobilenetv2-224-10)
 with a pipeline config, a checkpoint folder and a label_map.pbtxt file 
over a video camera stream. 

Example of usage with a webcam:
`python infer.py --video_cam 0 `

Example of usage with a video:
`python infer.py --input_path video.mov --output_path videoOuput.mov`

Made by: Israel Melendez Montoya
"""

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import argparse

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#%matplotlib inline


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

def main(input_path, output_path, config_path, labels_path):

    #recover our saved model
    pipeline_config = config_path
    #generally you want to put the last ckpt from training in here
    model_dir = '/content/fine_tuned_model/checkpoint/ckpt-10'
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join('/content/fine_tuned_model/checkpoint/ckpt-10'))


    detect_fn = get_model_detection_function(detection_model)

    #map labels for inference decoding
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        help="Path of the video to perform object detection inferences",
        default=None)
    
    parser.add_argument(
        "--video_cam",
        help="Path of the video to perform object detection inferences",
        default=0)

    parser.add_argument(
        "--output_path",
        help="Path of the output video",
        default=None)

    parser.add_argument(
        "--config_path",
        help="Path of the initial pipeline configuration of the \
            neural network (indicates the base architecture)",
        default="ml_aug_model/pipeline_file.config")
    
    parser.add_argument(
        "--checkpoint_path",
        help="Checkpoint path that must contain: checkpoint, \
            ckpt-X.data-Y, ckpt-X.index" ,
        default="/checkpoint/")

    parser.add_argument(
        "--labels_path",
        help="Labels map path that indicates different classes" ,
        default="label_map.pbtxt")

    args = parser.parse_args()

    # Parse all the arguments

    video_cam = args.video_cam
    input_path = args.input_path
    output_path = args.output_path

    ckpt_path = args.checkpoint_path
    config_path = args.config_path
    labels_path = args.labels_path

    #Show the arguments
    print()
    print("[INFO]: Input path is {}".format(input_path))
    print("[INFO]: Output path is {}".format(output_path))
    print("[INFO]: Video Camera is {}".format(video_cam))
    print("[INFO]: Config path is {}".format(config_path))
    print("[INFO]: Checkpoint path is {}".format(ckpt_path))
    print("[INFO]: Labels path is {}".format(labels_path))
    print()
    quit()
    # Conditional to open camera for inferences
    if input_path is None:
        print("[INFO]: Opening camera {1}, output video is {1}:".format(\
            video_cam, output_path))
            
        main(video_cam, output_path, config_path, labels_path)
    else:
        print("[INFO]: Input path video is {0}, output video is {1}".format(\
            input_path, output_path))

        main(input_path, output_path, config_path, labels_path)






