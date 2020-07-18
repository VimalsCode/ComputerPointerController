import logging as log

import cv2
from BaseModel import BaseModel


def draw_head_pose_estimation(original_frame, head_pose_estimation_output):
    """
    To draw the output from head pose estimation model
    :param original_frame:input image
    :param head_pose_estimation_output:output
    :return:frame with the text information
    """
    frame_text = "head pose: (y={:.2f}, p={:.2f}, r={:.2f})".format(head_pose_estimation_output[0],
                                                                    head_pose_estimation_output[1],
                                                                    head_pose_estimation_output[2])
    return cv2.putText(original_frame, frame_text, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (255, 255, 255), 1)


class HeadPoseEstimationModel(BaseModel):
    """
    The HeadPoseEstimationModel class used to load the model, apply frame transformation, predict and extract output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        To initialize the HeadPoseEstimationModel class
        :param model_name: path to the location where the model is available
        :param device: device to load the network
        :param extensions: extensions to use, if any
        """
        BaseModel.__init__(self, model_name, device, extensions)
        self.processed_image = None
        self.model_name = "Head pose estimation Model"

    def predict_head_pose_estimation(self, original_frame, image, visualize=True):
        """
        To perform head pose estimation
        :param original_frame: input frame
        :param image: detected face image
        :param visualize: flag if visualization is required
        :return: list containing yaw, pitch, roll
        """
        try:
            # preprocessing step
            # input format: name: "data" , shape: [1x3x60x60] , format: BGR format
            self.processed_image = self.preprocess_input(image)
            # prepare network input
            self.set_net_input()
            # call predict
            self.predict()
            # wait for the results
            if self.wait() == 0:
                # get the result
                network_result = self.infer_request.outputs
                head_pose_estimation_output = self.preprocess_output(network_result)
                # head pose estimation visualization
                if visualize:
                    original_frame = draw_head_pose_estimation(original_frame, head_pose_estimation_output)
                return head_pose_estimation_output, original_frame
        except Exception as e:
            log.error("The head pose prediction request cannot be completed!")
            log.error("Exception message during head pose prediction detection prediction : {}".format(e))

    def set_net_input(self):
        self.net_input = {self.input_blob: self.processed_image}

    def preprocess_output(self, outputs):
        """
        To preprocess the output from head pose estimation model.
        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        :param outputs: model output
        :return: list containing yaw, pitch, roll
        """
        head_pose_estimation_output = []

        yaw = outputs["angle_y_fc"].tolist()[0][0]
        pitch = outputs["angle_p_fc"].tolist()[0][0]
        roll = outputs["angle_r_fc"].tolist()[0][0]
        head_pose_estimation_output.append(yaw)
        head_pose_estimation_output.append(pitch)
        head_pose_estimation_output.append(roll)
        return head_pose_estimation_output
