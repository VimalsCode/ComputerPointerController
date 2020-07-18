import logging as log
from abc import abstractmethod

import cv2
from openvino.inference_engine.ie_api import IECore, IENetwork

"""
Base model class to be extended by other models
"""


class BaseModel:

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        Initialize
        :param model_name: model path
        :param device: device to use
        :param extensions: extensions
        """
        # model_weights
        self.model_bin = model_name + ".bin"
        # model_structure
        self.model_xml = model_name + ".xml"
        # device to use
        self.device = device
        self.plugin = IECore()
        self.network = None
        self.net_input = None
        # extensions to use
        self.extensions = extensions
        # Get the input layer
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        # name of the model extended
        self.model_name = None

    def load_model(self):
        """
        To load the model to the specified hardware
        :return:None
        """
        try:
            log.info("Loading {} IR to the plugin.".format(self.model_name))
            self.network = IENetwork(model=self.model_xml, weights=self.model_bin)
            self.input_blob = next(iter(self.network.inputs))
            self.input_shape = self.network.inputs[self.input_blob].shape
            self.output_blob = next(iter(self.network.outputs))
            # check model for unsupported layers
            self.check_model()
            # Load the network into the Inference Engine
            self.exec_network = self.plugin.load_network(self.network, self.device)
        except Exception as e:
            log.error("The loading of the model cannot be completed!".format(self.model_name))
            log.error("Exception message during {} load : {}".format(self.model_name, e))

    def check_model(self):
        """
        To check the model for unsupported layers and apply necessary extensions
        :return:None
        """
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.debug("Unsupported network layers found!")
            # Add necessary extensions
            self.plugin.add_extension(self.extensions, self.device)

    def preprocess_input(self, image):
        """
        To preprocess the input for the model.
        :param image: input frame
        :return: transformed input frame
        """
        # Pre-process the frame
        if self.model_name == "Gaze estimation Model":
            image = cv2.resize(image, (60, 60))
        else:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # Change format from HWC to CHW
        image_to_infer = image.transpose((2, 0, 1))
        # prepare according to face_detection model
        image_to_infer = image_to_infer.reshape(1, *image_to_infer.shape)
        return image_to_infer

    def predict(self):
        """
        Perform the inference request.
        :return:None
        """
        # make a infer request
        self.infer_request = self.exec_network.start_async(0, inputs=self.net_input)

    def wait(self):
        """
        Wait for the request to be complete.
        :return:status of the inference request
        """
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_network_input_shape(self):
        """
        To get the network input keys and corresponding shape
        :return:None
        """
        input_name = [i for i in self.network.inputs.keys()]
        log.debug("The network {} input name : {}".format(self.model_name, input_name))
        for name in input_name:
            input_shape = self.network.inputs[name].shape
            log.debug("The input shape for {} is {}".format(name, input_shape))

    def get_network_output_shape(self):
        """
        To get the network output keys
        :return:None
        """
        output_name = [i for i in self.network.outputs.keys()]
        log.debug("The network {} output name : {}".format(self.model_name, output_name))

    @abstractmethod
    def set_net_input(self):
        """
        Prepare the network input according to the model specification
        :return: None
        """
        pass
