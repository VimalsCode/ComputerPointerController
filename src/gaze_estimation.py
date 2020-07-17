import math

import cv2
from openvino.inference_engine.ie_api import IECore

from util.network_loader_helper import create_network, load_network, check_network, get_network_input_shape, \
    get_network_output_shape


class GazeEstimationModel:
    """
    The GazeEstimationModel class used to load the model, apply frame transformation, predict the gaze vector and
    extract required output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        To initialize the GazeEstimationModel class
        :param model_name: name of the model
        :param device: device to load the network
        :param extensions: extensions to use, if any
        """

        self.model_bin = model_name + ".bin"
        self.model_xml = model_name + ".xml"
        self.device = device
        self.extensions = extensions
        self.plugin = IECore()
        # Read the IR as a IENetwork
        self.network = create_network(self.model_xml, self.model_bin)
        # Get the input layer
        self.output_blob = next(iter(self.network.outputs))
        self.exec_network = None
        # info about network input & output
        get_network_input_shape(self.network, 'gaze')
        get_network_output_shape(self.network, 'gaze')

    def load_model(self):
        """
        To load the gaze estimation model to the specified hardware
        :return: None
        """
        # check model for unsupported layers
        self.check_model()
        # Load the network into the Inference Engine
        self.exec_network = load_network(self.plugin, self.network, self.device)

    def predict(self, left_eye_image, right_eye_image, head_pose_estimation_output):
        """
        To perform gaze estimation
        :param left_eye_image: left eye image
        :param right_eye_image: right eye image
        :param head_pose_estimation_output: head pose estimation
        :return: x,y position and gaze vector
        """
        try:
            # preprocessing step
            left_eye_processed_image = self.preprocess_input(left_eye_image)
            right_eye_processed_image = self.preprocess_input(right_eye_image)
            net_input = {'left_eye_image': left_eye_processed_image, 'right_eye_image': right_eye_processed_image,
                         'head_pose_angles': head_pose_estimation_output}
            # make a infer request
            infer_request = self.exec_network.start_async(0, inputs=net_input)
            status = self.exec_network.requests[0].wait(-1)
            if status == 0:
                # get the result
                network_result = infer_request.outputs[self.output_blob]
                return self.preprocess_output(network_result, head_pose_estimation_output)
        except Exception as e:
            print(str(e))

    def check_model(self):
        """
        To check the model for unsupported layers and apply necessary extensions
        :return:None
        """
        # Check for supported layers
        check_network(self.plugin, self.network, self.device, self.extensions)

    def preprocess_input(self, image):
        """
        To preprocess the input for the gaze estimation model.
        Name: left_eye_image, shape: [1x3x60x60]
        Name: right_eye_image , shape: [1x3x60x60]
        Name: head_pose_angles , shape: [1x3]
        :param image: input frame
        :return:  transformed input frame
        """
        # Pre-process the frame
        image = cv2.resize(image, (60, 60))
        # Change format from HWC to CHW
        image_to_infer = image.transpose((2, 0, 1))
        # prepare according to face_detection model
        image_to_infer = image_to_infer.reshape(1, *image_to_infer.shape)
        return image_to_infer

    def preprocess_output(self, outputs, head_pose_estimation_output):
        """
        To preprocess the output from gaze estimation model.
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector.
        Please note that the output vector is not normalizes and has non-unit length.
        Reference : https://github.com/opencv/open_model_zoo/blob/master/demos/gaze_estimation_demo/src/gaze_estimator.cpp
        Output layer name in Inference Engine format: gaze_vector
        :param outputs: gaze estimation output
        :param head_pose_estimation_output: head pose estimation output
        :return: x,y position and gaze vector
        """
        predicted_gaze_output = outputs[0]
        detected_roll_value = head_pose_estimation_output[2]

        cos_theta = math.cos(detected_roll_value * math.pi / 180)
        sin_theta = math.sin(detected_roll_value * math.pi / 180)

        x = predicted_gaze_output[0] * cos_theta + predicted_gaze_output[1] * sin_theta
        y = - predicted_gaze_output[0] * sin_theta + predicted_gaze_output[1] * cos_theta

        return (x, y), predicted_gaze_output
