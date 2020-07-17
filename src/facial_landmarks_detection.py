import cv2
from openvino.inference_engine.ie_api import IECore

from util.network_loader_helper import create_network, load_network, check_network, get_network_input_shape, \
    get_network_output_shape


class FacialLandmarksDetectionModel:
    """
    The FacialLandmarkDetectionModel class used to load the model, apply frame transformation, predict the facial key
    points and extract required output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        To initialize the FacialLandmarkDetectionModel class
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
        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        self.exec_network = None
        self.outputs = None
        # info about network input & output
        get_network_input_shape(self.network, 'Facial Landmark')
        get_network_output_shape(self.network, 'Facial Landmark')

    def load_model(self):
        """
        To load the facial landmark detection model to the specified hardware
        :return: None
        """
        # check model for unsupported layers
        self.check_model()
        # Load the network into the Inference Engine
        self.exec_network = load_network(self.plugin, self.network, self.device)

    def predict(self, image):
        """
        To perform facial landmarks detection
        :param image: input frame
        :return: list containing yaw, pitch, roll
        """
        try:
            # preprocessing step
            processed_image = self.preprocess_input(image)
            net_input = {self.input_blob: processed_image}
            # make a infer request
            infer_request = self.exec_network.start_async(
                0,
                inputs=net_input)
            status = self.exec_network.requests[0].wait(-1)
            if status == 0:
                # get the result
                network_result = infer_request.outputs[self.output_blob]
                return self.preprocess_output(network_result[0], image)
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
        To preprocess the input for the facial landmarks detection model.
        Name: "data" , shape: [1x3x48x48] , format: BGR
        :param image: input frame
        :return: preprocessed frame
        """
        # Pre-process the frame
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # Change format from HWC to CHW
        image_to_infer = image.transpose((2, 0, 1))
        # prepare according to face_detection model
        image_to_infer = image_to_infer.reshape(1, *image_to_infer.shape)
        return image_to_infer

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
