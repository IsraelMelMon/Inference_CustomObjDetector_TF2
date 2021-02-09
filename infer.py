# -*- coding: utf-8 -*-
"""infer.py

Program that runs a TF HUB pretrained object detection model 
(MELI - packages over a conveyor belt, using mobilenetv2-224-10)
 with a pipeline config, a checkpoint folder and a label_map.pbtxt file 
over a video camera stream. 

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
import time
from imutils.video import FPS, FileVideoStream, WebcamVideoStream
import imutils

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

Class VideoPredictor:
    def __init__(self, config_path="ml_aug_model/pipeline_file.config", ckpt_path="checkpoint"):
        # get current working directory
        #cwd = os.path.abspath(os.getcwd())
        ckpt_name = sorted(os.listdir(ckpt_path))[1].split(".")[0]
        model_dir = ckpt_path + ckpt_name
        #config_path = cwd+"/"+config_path
        #model_dir = cwd+"/"+model_dir

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
        label_map = label_map_util.load_labelmap(label_map_path)
        #print("[INFO]: Done")
        
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        

        #run detector on test image
        #it takes a little longer on the first run and then runs at normal speed. 
        print("[INFO]: Done loading labels...")
        print()
        
    def get_model_detection_function(self,model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(self,image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


    def start(self, input_path=0, output_path=None):
        # we recover our saved model here

        # gets the last ckpt from the ckpt folder automatically,
        # gets full paths for ckpt and pipeline files

        #input video for object detection inference
        if not isinstance(input_path,int):
            vid = WebcamVideoStream(src=0).start() # run another while function
        else:
            vid = FileVideoStream(input_path).start() # run another while in a function
        time.sleep(1.0)

        #output video name
        if output_path != None:

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            videoOut = cv2.VideoWriter(output_path,fourcc, 30.0, (im.shape[1],im.shape[0]))

        print("[INFO] loading model...")
        print("[INFO] starting video play...")
        fps = FPS().start()
        
        while True:

            frame = vid.read()
            frame = imutils.resize(frame, width=450)

            (im_width, im_height) = (frame.shape[1],frame.shape[0])

            
            image_np = np.array(frame).reshape((im_height, im_width, 3)).astype(np.uint8)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=.5,
                agnostic_mode=False,)

            cv2.imshow("frame",image_np_with_detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if output_path != None:
                videoOut.write(image_np_with_detections)

            fps.update()

        fps.stop()

        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    
        cv2.destroyAllWindows()
        vid.stop()

        if output_path != None:
                videoOut.release()
        

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        help="Path of the video to perform object detection inferences",
        default="9410828E-0960-45B6-8595-61B439AF4764.mov")
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
    args = parser.parse_args()

    # Parse all the arguments

    input_path = args.input_path
    output_path = args.output_path

    ckpt_path = args.checkpoint_path
    config_path = args.config_path

    #Show the arguments
    print()
    print("[INFO]: Input path is {}".format(input_path))
    print("[INFO]: Output path is {}".format(output_path))
    print("[INFO]: Config path is {}".format(config_path))
    print("[INFO]: Checkpoint path is {}".format(ckpt_path))
    print()
    
    # run the main
    main(input_path, output_path, config_path, ckpt_path)





