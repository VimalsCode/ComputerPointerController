import cv2
from openvino.inference_engine.ie_api import IECore

from util.network_loader_helper import create_network, load_network, check_network, get_network_input_shape, \
    get_network_output_shape


class HeadPoseEstimationModel:
    """
    The HeadPoseEstimationModel class used to load the model, apply frame transformation, predict and extract output
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        To initialize the HeadPoseEstimationModel class
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
        # info about network input & output
        get_network_input_shape(self.network, 'Head Post Estimation')
        get_network_output_shape(self.network, 'Head Post Estimation')

    def load_model(self):
        """
        To load the head post estimation model to the specified hardware
        :return: None
        """
        # check model for unsupported layers
        self.check_model()
        # Load the network into the Inference Engine
        self.exec_network = load_network(self.plugin, self.network, self.device)

    def predict(self, image):
        """
        To perform head post estimation for the provided frame
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
                network_result = infer_request.outputs
                return self.preprocess_output(network_result)
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
        To preprocess the input for the head pose estimation model.
        input format: name: "data" , shape: [1x3x60x60] , format: BGR format
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
