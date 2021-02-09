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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib
import matplotlib.pyplot as plt

import io, os, argparse, glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import random
import cv2
from imutils.video import FPS

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


def main(input_path, output_path, config_path, ckpt_path, labels_path):
    # we recover our saved model here

    cwd = os.path.abspath(os.getcwd())
    # gets the last ckpt from the ckpt folder automatically,
    # gets full paths for ckpt and pipeline files
    ckpt_name = sorted(os.listdir(ckpt_path))[1].split(".")[0]
    model_dir = ckpt_path + ckpt_name
    config_path = cwd+"/"+config_path
    model_dir = cwd+"/"+model_dir
    labels_path = cwd+"/"+labels_path

    print("[INFO]: Last checkpoint is:", model_dir)
    print()
    print("[INFO]: Config path is:", config_path)
    print()
    
    configs = config_util.get_configs_from_pipeline_file(config_path)
    print(configs)
    print()
    model_config = configs["model"]

    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)

    ckpt.restore(model_dir)
    print("[INFO]: Done restoring model...")
    detect_fn = get_model_detection_function(detection_model)

    #map labels for inference decoding
    label_map_path = configs['eval_input_config'].label_map_path
    #label_map_path = labels_path
    print(label_map_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    print("[INFO]: Done")
    quit()
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


        #run detector on test image
    #it takes a little longer on the first run and then runs at normal speed. 
    print("[INFO]: Loaded labels...")
    print()
    quit()
    #input video for object detection inference
    vid = cv2.VideoCapture(0) # here goes the video path
    ret, im = vid.read()
    imshape = im.shape
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    #print("same shapes?",im.shape, (im.shape[1],im.shape[0]))


    #output video name
    videoOut = cv2.VideoWriter('output_video.avi',fourcc, 30.0, (im.shape[1],im.shape[0]))

    print("[INFO] loading model...")
    print("[INFO] starting video play...")
    fps = FPS().start()
    quit()
    counter = 1
    while True:

        ret, frame = vid.read()

        if ret:
            

            #image_path = random.choice(TEST_IMAGE_PATHS)
            (im_width, im_height) = (frame.shape[1],frame.shape[0])
            #print(im_width, im_height)

            
            image_np = np.array(frame).reshape((im_height, im_width, 3)).astype(np.uint8)#load_image_into_numpy_array(image_path)
            #print(image_np)
            
            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()

            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)-1,
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.5,
                agnostic_mode=False,
            )

            #plt.figure(figsize=(12,16))
            #plt.imshow(image_np_with_detections)
            #plt.show()
            #cv2.imwrite("check{0}.jpg".format(counter), cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))
            videoOut.write(image_np_with_detections)
            fps.update()
            counter = counter + 1
        else:
            break
        fps.stop()

        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        videoOut.release()

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
        default="checkpoint/")
    parser.add_argument(
        "--labels_path",
        help="Labels map path that indicates different classes" ,
        default="/Users/israel/Inf_ObjDet_TF2HUB/cool-label_map.pbtxt")

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
    
    # Conditional to open camera for inferences
    if input_path == None:
        print("[INFO]: Opening camera {0}, output video is {1}:".format(\
            video_cam, output_path))

        main(video_cam, output_path, config_path, ckpt_path, labels_path)
    else:
        print("[INFO]: Input path video is {0}, output video is {1}".format(\
            input_path, output_path))

        main(input_path, output_path, config_path, ckpt_path, labels_path)






