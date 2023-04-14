import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContentLoss(nn.Module):
    """
    Calculate the Content loss of images against a single fixed target image
    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # Target image is not a parameter of the model, so we don't need to backpropagate through it
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size() 
    # a = batch size
    # b = number of feature maps (channels of image)
    # (c,d) = dimensions of a f. map (N=c*d) (N = number of pixels in feature map)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """
    Computes the style loss of a given image against a target style image
    Style loss is calculated as the mean squared error between the Gram matrices of the target and the input image
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Store the gram matrix of the target image
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    """
    Module to normalize the input image so that the mean is 0 and the standard deviation is 1
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Mean and std are 1D tensors of size 3, calculated along the channels of the image
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

VGG_PARAMS = {
    "cnn": models.vgg19(weights=models.VGG19_Weights.DEFAULT).features,
    "normalization_mean": torch.tensor([0.485, 0.456, 0.406]),
    "normalization_std": torch.tensor([0.229, 0.224, 0.225]),
    "content_layers": ["conv_4"],
    "style_layers": ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
}

def get_loss_network_from_model(style_img, content_img, device, pretrained_model_params=VGG_PARAMS):
    """
    Function to extract the style and content losses from a sequential CNN model
    :param style_img: style image
    :param content_img: content image
    :param device: device to run the model on
    :param pretrained_model_params: dictionary of parameters for the pretrained model
    :return: model: Sequential CNN model with added style and content losses,
        style_losses: list of style losses detected from the model,
        content_losses: list of content losses detected from the model
    """
    # Extract the parameters from the dictionary
    cnn = pretrained_model_params["cnn"].to(device)
    normalization_mean = pretrained_model_params["normalization_mean"].to(device)
    normalization_std = pretrained_model_params["normalization_std"].to(device)
    content_layers = pretrained_model_params["content_layers"]
    style_layers = pretrained_model_params["style_layers"]

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses, since we don't need them
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    # i+1 is the index of the first layer after the last content/style loss
    model = model[:(i + 1)]

    return model, style_losses, content_losses