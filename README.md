# Computer Pointer Controller

## Overview
Computer Pointer Controller uses the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.The workflow is achieved by running
multiple models and coordinating the flow of data between those models.

The project relies on the following auxiliary networks from openvino pretrained models to accomplish mouse pointer position movement,

* Face Detection model - [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* Head Pose Estimation model - [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* Facial Landmarks Detection model - [Facial Landmarks](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* Gaze Estimation Model - [Gaze Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The project repository is developed and tested under the following environment,

| Environment item        | Description           | 
| ------------- |:-------------:| 
| Operation system      | Windows 10 Professional | 
| Configuration      | Intel      |   
| python version | 3.6      |
| Video input | attached demo videos with single face detection      |
| Device | CPU      |


## Project Set Up and Installation

### Step 1: Download and install OpenVino Tookit
Follow the instructions specified [here](https://docs.openvinotoolkit.org/latest/index.html) to install the OpenVino tool kit based on the operating system requirements.The project respository was developed and 
tested based on the OpenVino version 2020.1.

### Step 2: Setup the repository
Clone this repository and perform the below steps to create the Python Virtual Environment,
* create a virtual environment using the following command,
```
python3 -m venv <env-name>
```

* Use the following command to activate the virtual environment,
```
source env-name/bin/activate
```

* Required project dependencies can be installed based on the following command,
```
pip3 install -r requirements.txt
```

### Step 3: Download OpenVino pretrained model
The project repository contains the required pretrained model already downloaded and can be directly used. If required to download manually, follow the below command,
*TODO:*
 
 > make sure the model is converted to the Inference Engine format (*.xml + *.bin) 

## Demo
To run the application use the following command,
```
python3 src/main.py -i bin/demo.mp4
```


## Documentation
Run the application with the -h option to get the required parameter details,
```
python3 src/main.py -h
```
Overview about the commandline parameters is provided here,

Mandatory Parameters - application execution related:

| Parameter name        | Description           | 
| ------------- |:-------------:| 
| -i, --input      | Input file location or 'CAM' to use the webcamera | 
| -d, --device      | To specify the location of face detection model .xml file|   
| -pt, --prob_threshold      | To specify the location of head pose estimation model .xml file      |

Mandatory Parameters - model related:

| Parameter name        | Description           | 
| ------------- |:-------------:| 
| -i, --input      | Input file location or 'CAM' to use the webcamera | 
| -mf, --model_fd      | To specify the location of face detection model .xml file|   
| -mf, --model_fd      | To specify the location of head pose estimation model .xml file      |
| -mf, --model_fd      | To specify the location of facial landmarks detection model .xml file      |
| -mf, --model_fd      | To specify the location of gaze estimation model .xml file      |

Optional Parameters - visualization related:
To control the behavior of visualization for the model detection,

| Parameter name        | Description           | 
| ------------- |:-------------:| 
| -v_mf, --visualization_model_fd      | To specify the location of face detection model .xml file|   
| -mf, --model_fd      | To specify the location of head pose estimation model .xml file      |
| -mf, --model_fd      | To specify the location of facial landmarks detection model .xml file      |
| -mf, --model_fd      | To specify the location of gaze estimation model .xml file      |


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
#### Models load time (in secs):
<p align="center">
<img src="analysis/Model_Load_Time.png" width=500px height=450px/>
</p>
<br>

#### Models Inference time (in secs):
<p align="center">
<img src="analysis/Model_Inference_Time.png" width=500px height=450px/>
</p>


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## References
https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
https://sebastian-hoeffner.de/uni/mthesis/HoeffnerGaze.pdf
https://esc.fnwi.uva.nl/thesis/centraal/files/f1317295686.pdf
https://stackoverflow.com/questions/10365087/gaze-estimation-from-an-image-of-an-eye


### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
