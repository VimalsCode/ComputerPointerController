import logging as log
import math

import cv2
from base_model import BaseModel


class GazeEstimationModel(BaseModel):
    """
    The GazeEstimationModel class used to load the model, apply frame transformation, predict the gaze vector and
    extract required output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        To initialize the GazeEstimationModel class
        :param model_name: path to the location where the model is available
        :param device: device to load the network
        :param extensions: extensions to use, if any
        """
        BaseModel.__init__(self, model_name, device, extensions)
        self.processed_image = None
        self.model_name = "Gaze estimation Model"
        self.left_eye_processed_image = None
        self.right_eye_processed_image = None
        self.head_pose_estimation_output = None

    def predict_gaze_estimation(self, left_eye_image, right_eye_image, head_pose_estimation_output):
        """
        To perform gaze estimation
        :param left_eye_image: left eye image
        :param right_eye_image: right eye image
        :param head_pose_estimation_output: head pose estimation list
        :return: x,y position and gaze vector
        """
        try:
            # preprocessing step
            #  Name: left_eye_image, shape: [1x3x60x60], Name: right_eye_image , shape: [1x3x60x60],
            #  Name: head_pose_angles , shape: [1x3]
            self.left_eye_processed_image = self.preprocess_input(left_eye_image)
            self.right_eye_processed_image = self.preprocess_input(right_eye_image)
            self.head_pose_estimation_output = head_pose_estimation_output
            # prepare network input
            self.set_net_input()
            # call predict
            self.predict()
            # wait for the results
            if self.wait() == 0:
                # get the result
                network_result = self.infer_request.outputs[self.output_blob]
            return self.preprocess_output(network_result, head_pose_estimation_output)
        except Exception as e:
            log.error("The gaze estimation request cannot be completed!")
            log.error("Exception message during gaze estimation : {}".format(e))

    def set_net_input(self):
        self.net_input = {'left_eye_image': self.left_eye_processed_image,
                          'right_eye_image': self.right_eye_processed_image,
                          'head_pose_angles': self.head_pose_estimation_output}

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

    def draw_gaze_estimation(self, eye_bounding_box, predicted_gaze_output, frame):
        """
        To draw gaze estimation output on the frame
        Reference : https://knowledge.udacity.com/questions/257811
        :param eye_bounding_box: face landmark detection output
        :param predicted_gaze_output: gaze vector
        :param frame: image frame
        :return:
        """
        left_eye_center1 = int(eye_bounding_box[0] + predicted_gaze_output[0] * 100)
        left_eye_center2 = int(eye_bounding_box[1] - predicted_gaze_output[1] * 100)
        self.draw_arrow(frame, (eye_bounding_box[0], eye_bounding_box[1]), (left_eye_center1, left_eye_center2))

        right_eye_center1 = int(eye_bounding_box[2] + predicted_gaze_output[0] * 100)
        right_eye_center2 = int(eye_bounding_box[3] - predicted_gaze_output[1] * 100)
        self.draw_arrow(frame, (eye_bounding_box[2], eye_bounding_box[3]), (right_eye_center1, right_eye_center2))
        return frame

    def draw_arrow(self, frame, point1, point2):
        """
        To draw the arrow line
        :param frame: image frame
        :param point1: bounding box1
        :param point2: bounding box1
        :return: frame with arrowedLine drawn
        """
        return cv2.arrowedLine(frame, point1, point2, (255, 0, 0), 2)
