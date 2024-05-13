import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Load the pretrained model
model = models.resnet18(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

# Image transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()

get_vector()

# -*- coding: utf-8 -*-

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


class ResNetFeature(nn.Module):
    def __init__(self, feature='resnet101'):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetFeature, self).__init__()
        if feature == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return pool5


resnet_transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])