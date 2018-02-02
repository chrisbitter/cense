import nnFactory as nnF
from xml.etree import ElementTree as ET
import xml.dom.minidom as prettyXml
import os

input_size = (40, 40, 3)

abs_path = os.path.dirname(__file__)
abs_path = abs_path.replace('NeuralNetwork', 'Resources/nn-data/')
file_name = "Network.xml"

file_path = abs_path+file_name


def load_xml(filename):
    obj = ET.parse(filename)
    obj = obj.getroot()
    return obj


def load_models():
    actor_model = nnF.actor_network(input_size)  # input_size = (40, 40, 3)
    critic_model = nnF.critic_network(input_size)  # input_size = (40, 40, 3)
    return actor_model, critic_model


def build_xml():
    actor_model, critic_model = load_models()
    model = ET.Element('Model')
    actor = ET.SubElement(model, 'Network', {"Name": "Actor"})
    actor_layers = []
    for layer in actor_model.layers:
        if "Input" in str(type(layer)):
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Input', 'Size':
                '{}x{}'.format(layer.output_shape[1], layer.output_shape[2])}))
        if "Conv" in str(type(layer)):
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Convolutional', 'Shape':
                '{}x{}'.format(layer.output_shape[1], layer.output_shape[2]), 'Size':
                '{}'.format(layer.output_shape[3]), 'Kernel':
                '{}x{}'.format(layer.kernel_size[0], layer.kernel_size[1])}))
        if "Pooling" in str(type(layer)):
            kernel = [layer.input_shape[1]-layer.output_shape[1]+1,
                      layer.input_shape[2]-layer.output_shape[2]+1]
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Pooling', 'Kernel':
                '{}x{}'.format(kernel[0], kernel[1])}))
        if "Flatten" in str(type(layer)):
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Flatten', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if "Dropout" in str(type(layer)):
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Dropout', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if ("Dense" in str(type(layer)) or "Concat" in str(type(layer))) and len(layer._outbound_nodes) != 0:
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Fully-Connected', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if ("Dense" in str(type(layer)) or "Concat" in str(type(layer))) and len(layer._outbound_nodes) == 0:
            actor_layers.append(ET.SubElement(actor, 'Layer', {'Type': 'Output', 'Size':
                '{}'.format(layer.output_shape[1])}))
    critic = ET.SubElement(model, 'Network', {"Name": "Critic"})
    critic_layers = []
    for layer in critic_model.layers:
        if "Input" in str(type(layer)):
            if len(layer.output_shape) >= 3:
                critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Input', 'Size':
                    '{}x{}'.format(layer.output_shape[1], layer.output_shape[2])}))
            else:
                critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Input', 'Size':
                    '{}'.format(layer.output_shape[1])}))
        if "Conv" in str(type(layer)):
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Convolutional', 'Shape':
                '{}x{}'.format(layer.output_shape[1], layer.output_shape[2]), 'Size':
                                                                   '{}'.format(layer.output_shape[3]), 'Kernel':
                                                                   '{}x{}'.format(layer.kernel_size[0],
                                                                                  layer.kernel_size[1])}))
        if "Pooling" in str(type(layer)):
            kernel = [layer.input_shape[1] - layer.output_shape[1] + 1,
                      layer.input_shape[2] - layer.output_shape[2] + 1]
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Pooling', 'Kernel':
                '{}x{}'.format(kernel[0], kernel[1])}))
        if "Flatten" in str(type(layer)):
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Flatten', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if "Dropout" in str(type(layer)):
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Dropout', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if ("Dense" in str(type(layer)) or "Concat" in str(type(layer))) and len(layer._outbound_nodes) != 0:
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Fully-Connected', 'Size':
                '{}'.format(layer.output_shape[1])}))
        if ("Dense" in str(type(layer)) or "Concat" in str(type(layer))) and len(layer._outbound_nodes) == 0:
            critic_layers.append(ET.SubElement(critic, 'Layer', {'Type': 'Output', 'Size':
                '{}'.format(layer.output_shape[1])}))

    xml_string = ET.tostring(model)
    xml_string_pretty = prettyXml.parseString(xml_string).toprettyxml()

    save_xml(xml_string_pretty, file_path)


def save_xml(xml_string, filename):
    with open(filename, 'w') as f:
        f.write(xml_string)


def set_input_size(new_input_size):
    input_size = new_input_size


if __name__ == "__main__":
    build_xml()
