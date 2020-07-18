import cv2
from base_model import BaseModel

import logging as log


def draw_facial_landmarks_estimation(detected_face_image, left_eye_dimension, right_eye_dimension):
    """
    Reference : https://knowledge.udacity.com/questions/245775
    :param detected_face_image:frame
    :param left_eye_dimension:left eye dimension
    :param right_eye_dimension:right eye dimension
    :return:frame with eye dimension drawn
    """
    # draw bounding box on left eye
    cv2.rectangle(detected_face_image, (left_eye_dimension[0], left_eye_dimension[1]),
                  (left_eye_dimension[2], left_eye_dimension[3]),
                  (0, 55, 255), 1)

    # draw bounding box on right eye
    cv2.rectangle(detected_face_image, (right_eye_dimension[0], right_eye_dimension[1]),
                  (right_eye_dimension[2], right_eye_dimension[3]),
                  (0, 55, 255), 1)
    return detected_face_image


class FacialLandmarksDetectionModel(BaseModel):
    """
    The FacialLandmarkDetectionModel class used to load the model, apply frame transformation, predict the facial key
    points and extract required output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        FacialLandmarksDetectionModel initialization
        :param model_name: model path
        :param device: device to use
        :param extensions: specified extensions
        """
        BaseModel.__init__(self, model_name, device, extensions)
        self.processed_image = None
        self.outputs = None
        self.model_name = "Face Landmarks detection Model"

    def predict_facial_landmarks_detection(self, image, visualize=True):
        """
        To perform facial landmark detection
        :param image: detected face image
        :param visualize: flag if visualization is required
        :return: left and right eye extracted
        """
        try:
            # preprocessing step
            # Name: "data" , shape: [1x3x48x48] , format: BGR
            self.processed_image = self.preprocess_input(image)
            # prepare network input
            self.set_net_input()
            # call predict
            self.predict()
            # wait for the results
            if self.wait() == 0:
                # get the result
                network_result = self.infer_request.outputs[self.output_blob]
                left_eye_image, right_eye_image, left_eye_dimension, right_eye_dimension = self.preprocess_output(
                    network_result[0], image)
                # draw output
                if visualize:
                    draw_facial_landmarks_estimation(image, left_eye_dimension, right_eye_dimension)
                return left_eye_image, right_eye_image

        except Exception as e:
            log.error("The face detection prediction request cannot be completed!")
            log.error("Exception message during face detection prediction : {}".format(e))

    def set_net_input(self):
        self.net_input = {self.input_blob: self.processed_image}

    def preprocess_output(self, outputs, image):
        """
        To preprocess the output from facial landmarks detection model.
        Note:The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for
        five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be
        in range [0,1].(two eyes, nose, and two lip corners.)
        Reference : https://knowledge.udacity.com/questions/245775
        :param outputs: model output
        :param image: input frame
        :return: left eye coordinate list and right eye coordinate list and dimensions info for visualization
        """
        # image width and height
        height = image.shape[0]
        width = image.shape[1]

        left_eye_x = int(outputs[0] * width)
        left_eye_y = int(outputs[1] * height)
        right_eye_x = int(outputs[2] * width)
        right_eye_y = int(outputs[3] * height)

        # make box for left eye
        l_xmin = left_eye_x - 20
        l_ymin = left_eye_y - 20
        l_xmax = left_eye_x + 20
        l_ymax = left_eye_y + 20
        left_eye_dimension = [l_xmin, l_ymin, l_xmax, l_ymax]

        # get left eye image
        left_eye = image[l_ymin:l_ymax, l_xmin:l_xmax]

        # make box for right eye
        r_xmin = right_eye_x - 20
        r_ymin = right_eye_y - 20
        r_xmax = right_eye_x + 20
        r_ymax = right_eye_y + 20
        right_eye_dimension = [r_xmin, r_ymin, r_xmax, r_ymax]

        # get right eye image
        right_eye = image[r_ymin:r_ymax, r_xmin:r_xmax]
        # store the value for visualization
        self.outputs = outputs

        return left_eye, right_eye, left_eye_dimension, right_eye_dimension

    def get_outputs(self):
        return self.outputs
