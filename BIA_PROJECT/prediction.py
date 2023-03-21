import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

def predict(input, model, device):
    '''
    :param input: input image
    :param model: CNN model
    :param device: device (default:cuda)
    :return: the prediction of category
    '''
    # change the model to cuda environment.
    model.to(device)
    with torch.no_grad():
        input=input.to(device)
        # the output is a softmax (dm = 1)
        out = model(input)
        # get the sequence of the bigger data in softmax as the prediction category
        _, pre = torch.max(out.data, 1)
        return pre.item()

def segmentation(input,model,device):
    '''
    :param input:  input image
    :param model: CNN model
    :param device: device (default:cuda)
    :return: mask image of the input image
    '''
    with torch.no_grad():
        input=input.to(device)
        # get masks by the model
        out = model(input).detach().cpu()
        # resize the model to (512,512)
        torch_resizee = Resize([512, 512])
        out = torch_resizee(out)
        out = torch.squeeze(out)
        out = torch.round(out)
        # generate the image from array
        out = Image.fromarray(np.uint8(out) * 255)
        return out

