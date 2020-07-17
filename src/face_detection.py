import cv2
from openvino.inference_engine import IECore

from util.network_loader_helper import create_network, load_network, check_network, get_network_input_shape, \
    get_network_output_shape


class FaceDetectionModel:
    """
    The FaceDetectionModel class used to load the model, apply frame transformation, predict and extract output
    """

    def __init__(self, model_name, device='CPU', probs_threshold=0.5, extensions=None):
        """
        To initialize the FaceDetectionModel class
        :param model_name: name of the model
        :param device: device to load the network
        :param extensions: extensions to use, if any
        """
        self.model_bin = model_name + ".bin"
        self.model_xml = model_name + ".xml"
        self.probs_threshold = probs_threshold
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
        # info about network input & output
        get_network_input_shape(self.network, 'Face Detection')
        get_network_output_shape(self.network, 'Face Detection')

    def load_model(self, model_path=None):
        """
        To load the face detection model to the specified hardware
        :param model_path:
        :return: None
        """
        # check model for unsupported layers
        self.check_model()
        # Load the network into the Inference Engine
        self.exec_network = load_network(self.plugin, self.network, self.device)

    def predict(self, image):
        """
        To perform face detection prediction for the provided frame
        :param image: input frame
        :return: bounding box list and image with the detection drawn
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
                detected_face_bounding = self.preprocess_output(network_result, image)
                box = detected_face_bounding[0]
                detected_face_image = image[box[1]:box[3], box[0]:box[2]]
            return detected_face_image, box
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
        To preprocess the input for the face detection model.
        input format:Name: input, shape: [1x3x384x672], format: BGR
        :param image: input frame
        :return: transformed input frame
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
