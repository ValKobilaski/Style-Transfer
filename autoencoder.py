import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from collections import namedtuple

LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])

class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes: list, channel_sizes: list, input_size = 256):
        """
        :param layer_sizes: The sizes of the layers in the autoencoder
        :param channel_sizes: The number of channels in each layer
        :param input_size: The size of the input image (assumed to be square)
        """
        super().__init__()
        moduleList = []

        cur_image_size = input_size
        cur_channel_size = 3

        # Encoder layers
        for idx, layer_size in enumerate(layer_sizes):
            # Check if the layer size is divisible by the current image size
            if cur_image_size % layer_size != 0:
                raise ValueError("Layer size must be achievable from the current image size at all intermediate steps:" + \
                                 " Issue at (layer size: {}, current image size: {})".format(layer_size, cur_image_size))
            # Find the stride required to reduce the image size to the layer size
            stride = cur_image_size // layer_size
            # Add the convolutional layer
            moduleList.append(nn.Conv2d(cur_channel_size, channel_sizes[idx], kernel_size=3, stride=1, padding="same"))
            # Add the ReLU activation
            moduleList.append(nn.ReLU())
            # Add the max pooling layer only if stride length is greater than 1
            if stride > 1:
                moduleList.append(nn.MaxPool2d(kernel_size=stride, stride=stride))
            # Update the current image size and channel size
            cur_image_size = layer_size
            cur_channel_size = channel_sizes[idx]

        # Final convolutional layer
        moduleList.append(nn.Conv2d(cur_channel_size, channel_sizes[-1], kernel_size=3, stride=1, padding="same"))
        # Flatten the output
        moduleList.append(nn.Flatten())

        # Construct the encoder
        self.encoder = nn.Sequential(*moduleList)

        # Empty the module list
        moduleList = []
        # Decoder layers
        # Reshape the output
        moduleList.append(Reshape((-1, channel_sizes[-1], cur_image_size, cur_image_size)))

        for idx, layer_size in enumerate(layer_sizes[::-1]):
            # Check if the layer size is a multiple of the current image size
            if layer_size % cur_image_size != 0:
                raise ValueError("Layer size must be achievable from the current image size at all intermediate steps:" + \
                                 " Issue at (layer size: {}, current image size: {})".format(layer_size, cur_image_size))
            # Find the stride required to increase the image size to the layer size
            stride = layer_size // cur_image_size
            # Upssample the image only if stride length is greater than 1
            if stride > 1:
                moduleList.append(nn.Upsample(scale_factor=stride))
            # Add the convolutional layer
            moduleList.append(nn.Conv2d(cur_channel_size, channel_sizes[-idx-1], kernel_size=3, stride=1, padding="same"))
            # Add the ReLU activation
            moduleList.append(nn.ReLU())
            # Update the current image size and channel size
            cur_image_size = layer_size
            cur_channel_size = channel_sizes[-idx-1]
        # Find the stride required to increase the image size to the input size
        stride = input_size // cur_image_size
        # Upsample the image only if stride length is greater than 1
        if stride > 1:
            moduleList.append(nn.Upsample(scale_factor=stride))
        # Add the final convolutional layer
        moduleList.append(nn.Conv2d(cur_channel_size, 3, kernel_size=3, stride=1, padding="same"))

        # Construct the decoder
        self.decoder = nn.Sequential(*moduleList)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoEncoderLossNetwork(nn.Module):
    def __init__(self, autoEncoder: AutoEncoder):
        """
        :param autoEncoder: The autoencoder to use for loss calculation
        """
        super().__init__()

        # Get the encoder from the autoencoder
        self.encoder = autoEncoder.encoder
        # Find the indices of the Conv2d layers in the encoder
        layer_indices = []
        for idx, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv2d):
                layer_indices.append(idx)
        self.layer_indices = set(layer_indices)
        self.LossOutput = namedtuple(
            "LossOutput", ["conv2d{}".format(idx + 1) for idx in range(len(layer_indices))])
        self.relu = nn.ReLU()

    def forward(self, x):
        output = {}
        for idx, module in enumerate(self.encoder):
            x = module(x)
            # If the current module is a Conv2d layer, save the output
            if idx in self.layer_indices:
                # Save the output after performing a ReLU activation
                output["conv2d{}".format(len(output) + 1)] = self.relu(x)
        # Return the output as a named tuple
        return self.LossOutput(**output)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(self.shape)