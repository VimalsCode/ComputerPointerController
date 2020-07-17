import logging as log

from openvino.inference_engine import IENetwork

"""
Utility to handle network related API call
"""


def create_network(model_xml, model_bin):
    """
    To create network instance
    :param model_xml: network structure
    :param model_bin: network trained weights
    :return: created network based on structure and weights
    """
    return IENetwork(model=model_xml, weights=model_bin)


def load_network(plugin, network, device):
    """
    To load the network to the specified device
    :param plugin: plugin instance
    :param network: created network
    :param device: device to load the specified network
    :return: loaded network on to the device
    """
    return plugin.load_network(network, device)


def check_network(plugin, network, device, extensions):
    """
    check the network for any possible unsupported layers for the specified device
    :param plugin:plugin instance
    :param network:created network
    :param device: device to check for unsupported layers
    :param extensions:extensions to use
    :return:None
    """
    # Check for supported layers
    supported_layers = plugin.query_network(network=network, device_name=device)
    unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]

    if len(unsupported_layers) != 0:
        log.debug("Unsupported network layers found!")
        # Add necessary extensions
        plugin.add_extension(extensions, device)


def get_network_input_shape(network, type):
    """
    To get the network input keys and corresponding shape
    :param network:created network
    :param type:model type
    :return:None
    """
    input_name = [i for i in network.inputs.keys()]
    log.debug("The network {} input name : {}".format(type, input_name))
    for name in input_name:
        input_shape = network.inputs[name].shape
        log.debug("The input shape for {} is {}".format(name, input_shape))


def get_network_output_shape(network, type):
    """
    To get the network output keys
    :param network:created network
    :param type:model type
    :return:None
    """
    output_name = [i for i in network.outputs.keys()]
    log.debug("The network {} output name : {}".format(type, output_name))
