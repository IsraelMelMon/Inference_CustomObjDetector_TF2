# Inference_CustomObjDetector_TF2


 To run demo:

# Inference_Object_Detection_MELI_Tensorflow_2.X

Tested using Python 3.6.5


# MELI-Inference
Video inferences for MELI  conveyor products (webcam and video files supported)


##  To run demo:

First, get the TF Object detection API

`git clone --depth 1 https://github.com/tensorflow/models`

`cd models/research/`

`protoc object_detection/protos/*.proto --python_out=. `

`cp object_detection/packages/tf2/setup.py . `

And we install it with
`python -m pip install . `
## Now we are ready to do inferences
`git clone https://github.com/IsraelMelMon/Inference_CustomObjDetector_TF2.git`

`cd Inference_CustomObjDetector_TF2`

`python infer.py`

## To run in a webcam:

`python infer.py --input_path 0`



## To run in a different video other than demo:


`python infer.py --input_path /path/to/video`




## To save inference results as a video (only available for Video Files):


`python infer.py --input_path /path/to/video --output_path /path/to/output.mov`

