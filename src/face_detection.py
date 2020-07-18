import logging as log

import cv2
from BaseModel import BaseModel


class FaceDetectionModel(BaseModel):
    """
    The FaceDetectionModel class used to load the model, apply frame transformation, predict and extract output
    """

    def __init__(self, model_name, device='CPU', probs_threshold=0.5, extensions=None):
        """
        Face detection model initialization
        :param model_name: model path
        :param device: device to use
        :param probs_threshold: probability threshold
        :param extensions: specified extensions
        """
        BaseModel.__init__(self, model_name, device, extensions)
        self.processed_image = None
        self.probs_threshold = probs_threshold
        self.model_name = "Face Detection Model"

    def predict_face_detection(self, image, visualize=True):
        """
        To perform face detection
        :param image: input frame
        :param visualize: flag if visualization is required
        :return: bounding box list, detected face image
        """
        try:
            # preprocessing step
            # Name: input, shape: [1x3x384x672], format: BGR
            self.processed_image = self.preprocess_input(image)
            # prepare network input
            self.set_net_input()
            # call predict
            self.predict()
            # wait for the results
            if self.wait() == 0:
                # get the result
                network_result = self.infer_request.outputs[self.output_blob]
                detected_face_bounding = self.preprocess_output(network_result, image)
                # only one detection is considered
                box = detected_face_bounding[0]
                detected_face_image = image[box[1]:box[3], box[0]:box[2]]
                if visualize:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                                  (0, 55, 255),
                                  1)
            return detected_face_image, box
        except Exception as e:
            log.error("The face detection prediction request cannot be completed!")
            log.error("Exception message during face detection prediction : {}".format(e))

    def set_net_input(self):
        self.net_input = {self.input_blob: self.processed_image}

    def preprocess_output(self, outputs, image):
        """
        To preprocess the output from face detection model.
        prediction output format: [1, 1, N, 7] -  [image_id, label, conf, x_min, y_min, x_max, y_max]
        :param outputs: model output
        :param image: input frame
        :return: bounding box list
        """
        detected_face_bounding = []
        # Grab the shape of the input
        width = image.shape[1]
        height = image.shape[0]
        for obj in range(len(outputs[0][0])):
            output = outputs[0][0][obj]
            if output[2] >= self.probs_threshold:
                xmin = int(output[3] * width)
                ymin = int(output[4] * height)
                xmax = int(output[5] * width)
                ymax = int(output[6] * height)
                detected_face_bounding.append([xmin, ymin, xmax, ymax])
        return detected_face_bounding
