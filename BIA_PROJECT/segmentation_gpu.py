
# `os` stands for operating system. This module provides functions to look for information
# on the file system (e.g. listing directories and appending paths).
import os

# `tqdm` is a library for convenient command-line progress bars
from tqdm import tqdm

# `numpy` provides arrays and numerical operations on them
import numpy as np

# `torch` is the root module for PyTorch, a deep learning framework
import torch
# `torch.nn` contains building blocks for neural networks
from torch import nn
# `torch.nn.functional` contains functions that operate on tensors.
# Tensors are multi-dimensional arrays, but some have autograd (auto-
# differentiation) functionalities that makes writing code easy
import torch.nn.functional as F
# `Dataset` and `DataLoader` provide interfaces from data to the model
from torch.utils.data import Dataset, DataLoader
# Tensorboard logs the performance (loss, accuracy etc.) of a training
# model in real time. But here we only use a conventional logger
from torch.utils.tensorboard import SummaryWriter
# `torchvision.transforms` provides transformations on images in tensor
# form
import torchvision.transforms

# PIL is a historic module for processing images. We import it only for
# reading and saving images. For this purpose, it is more convenient
# than skimage
from PIL import Image


class DoubleConv(nn.Module):
    """
    Repeat (Conv => ReLU).

    [Conv => ReLU] => [Conv         =>         ReLU]
    ^in_channels      ^mid_channels ^out_channels
    """
    def __init__(
        self, in_channels, out_channels,
        mid_channels=None,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        """
        For image with length L, L' = (L - kernel_size)/stride + 1
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        # `nn.Sequential` concatenates the blocks in the order they are supplied,
        # to form an integral block
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # This function overrides the inherited method, so that when
    # an instance of `DoubleConv` is called with an argument x,
    # it forwards x to the `double_conv` block
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    The downsampling block shrinks the size of layer and
    doubles the number of channels (not shown here because
    we allow setting number of in_channels and out_channels)
    """
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    The upsampling block upscales the layer but reduces the
    number of channels to synthesize feature information
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Taken from https://github.com/milesial/Pytorch-UNet

    We added control for the size of the network
    """
    def __init__(self, n_channels, n_classes=2, first_out_channels=64, bilinear=True):
        """
        Initialize an instance of U-Net segmentation network

        Parameters:
            n_channels: Number of channels in the image. For gray-scale images, 
                n_channels=1. For RGB images, n_channels=3
            n_classes: Number of pixel classes to output. For black-white contrast,
                n_classes=2. Default=2
            first_out_channels: Number of output channels of the first double
                convolution layer
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(n_channels, first_out_channels)

        self.down1 = Down(first_out_channels, first_out_channels * 2)
        self.down2 = Down(first_out_channels * 2, first_out_channels * 4)
        self.down3 = Down(first_out_channels * 4, first_out_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(first_out_channels * 8, first_out_channels * 16 // factor)

        self.up1 = Up(first_out_channels * 16, first_out_channels * 8 // factor, bilinear)
        self.up2 = Up(first_out_channels * 8, first_out_channels * 4 // factor, bilinear)
        self.up3 = Up(first_out_channels * 4, first_out_channels * 2 // factor, bilinear)
        self.up4 = Up(first_out_channels * 2, first_out_channels, bilinear)

        if n_classes == 2:
            self.out = nn.Sequential(
                nn.Conv2d(first_out_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Logistic probabilities indicating that the pixel is part of lung
        logits = self.out(x)
        return logits


class CXRDataset(Dataset):
    """
    Define how to read data from disk to tensors

    Structure is referenced from https://blog.csdn.net/kobayashi_/article/details/108951993
    """
    def __init__(self, image_path, mask_path, rescale_size) -> None:
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        # os.listdir() lists all file or folder names under the
        # given directory, in our case the file names of the masks
        self.mask_rel_fnames = os.listdir(mask_path)

        # Function that converts a PIL.Image to torch.Tensor
        self.to_tensor = torchvision.transforms.ToTensor()
        # Function that rescales the image
        self.resize = torchvision.transforms.Resize(rescale_size)
        # Function that converts RGB to single-channel gray-scale
        self.gray_scale = torchvision.transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        """
        Return length of this dataset
        """
        return len(self.mask_rel_fnames)

    def __getitem__(self, index: int):
        """
        Return the i-th data and mask in this dataset
        """
        mask_rel_fname = self.mask_rel_fnames[index]
        image_rel_fname = mask_rel_fname.replace('_mask.png', '.png')
        mask_fname = os.path.join(self.mask_path, mask_rel_fname)
        image_fname = os.path.join(self.image_path, image_rel_fname)

        image = Image.open(image_fname)
        mask = Image.open(mask_fname)
        image = self.resize(image)
        mask = self.resize(mask)
        image = self.gray_scale(image)
        mask = self.gray_scale(mask)

        # Normalize to range [0, 1] for easier training
        image = self.to_tensor(image) / 255
        mask = self.to_tensor(mask) / 255
        return image, mask


def dice_coef(label: torch.Tensor, predicted: torch.Tensor, smoother=1) -> torch.Tensor:
    """
    Computes the DICE coefficient, a superior similarity metric for
    image segmentation compared to pixel-wise similarity.

    Parameters:
        label: True values. Should have size [batch_size, n_channels, height, width]
        predicted: Model-predicted values. Should have same size as label
        smoother: Constant added to both numerator and denominator for numerical stability
    """
    intersection = torch.sum(label * predicted, dim=[1, 2, 3])
    union = torch.sum(label, dim=[1, 2, 3]) + torch.sum(predicted, dim=[1, 2, 3])
    return (2 * intersection + smoother)/(union + smoother)


class DiceLoss(nn.Module):
    """
    DICE loss that directly uses the formula of DICE metric. DICE loss
    is superior to cross entropy loss in medical image segmentation tasks
    which have small object sizes and large empty background, leading to
    a bias in training.

    Referenced from https://blog.csdn.net/m0_37477175/article/details/83004746
    """
    def __init__(self, smoother=1, reduction='mean') -> None:
        super().__init__()
        self.smoother = smoother
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return (torch.mean(1 - dice_coef(target, input, self.smoother), dim=0))
        else:
            raise NotImplementedError

class DiceWithBCELoss(nn.Module):
    """
    DICE loss and weighted binary cross entropy (BCE) loss. DICE loss suffers
    from numeric instability, and this is compensated by BCE loss.
    Referenced from https://blog.csdn.net/m0_37477175/article/details/83004746
    """
    def __init__(self, smoother=1, bce_weight=1e-3, reduction='mean'):
        super().__init__()
        self.smoother = smoother
        self.bce_weight = bce_weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return torch.mean(
                (self.bce_weight * F.binary_cross_entropy(input, target)
                - dice_coef(input, target, smoother=self.smoother)),
                dim=0
            )
        else:
            raise NotImplementedError


if __name__ == '__main__':  # Perform training
    IMAGE_PATH = None
    MASK_PATH = None
    SAVED_MODEL = None

    NUM_WORKERS = 0

    DEVICE = 'cuda'
    EPOCH = 20

    RESCALE_SIZE = (512, 512)
    FIRST_OUT_CHANNELS = 16
    BATCH_SIZE = 16
    LEARNING_RATE = 4e-5
    MOMENTUM = 0.5

    USE_MODEL = False

    net = UNet(n_channels=1, n_classes=2, first_out_channels=FIRST_OUT_CHANNELS, bilinear=False).to(DEVICE)

    # As before, choice of optimizer is referenced from https://github.com/milesial/Pytorch-UNet
    optimizer = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # When we tuned the model, we discovered that our model often generated deformed lung masks
    # after it seemed to generate normal ones. This led us to think of limiting learning rate after
    # the model had largely converged. We tested and found that the simplest learning rate
    # scheduler (StepLR) works best
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, verbose=True)

    # As before, referenced from https://blog.csdn.net/m0_37477175/article/details/83004746
    loss_func = DiceWithBCELoss()

    data = CXRDataset(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        rescale_size=RESCALE_SIZE
    )
    dataloader = DataLoader(data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    # We need a new DataLoader for validation because the high batch size
    # causes our CUDA to run out of memory
    val_loader = DataLoader(data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    writer = SummaryWriter()

    n_batches = len(dataloader)
    n_pixels = RESCALE_SIZE[0] * RESCALE_SIZE[1]
    for epoch in range(EPOCH):
        print("Begin training epoch", epoch)
        net.train()  # set to training mode

        running_loss = 0.0
        for i, (image, mask) in tqdm(enumerate(dataloader), total=n_batches):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            # Predict from input
            pred = net(image)

            # Compute loss between prediction and ground truth
            loss = loss_func(pred, mask)

            # Clear gradients of the optimizer for new round
            optimizer.zero_grad()

            # Compute the gradients of the loss function using
            # back-propagation (chain rule)
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Convert loss to numeric scalar and add to total loss
            running_loss += loss.item()
        running_loss /= n_batches  # average across batches
        scheduler.step()

        print("Begin validation")
        net.eval()  # set to evaluation mode
        # Unfortunately, computing DICE accuracy was not possible because
        # we ran out of CUDA memory while training.
        #
        # dice_acc = 0.0
        # for image, mask in tqdm(val_loader):
        #     image, mask = image.to(DEVICE), mask.to(DEVICE)
        #     pred = net(image)
        #     pred_class = torch.round(pred)
        #     dice_acc += dice_coef(mask, pred_class)
        # dice_acc /= n_batches

        pixel_acc = 0
        for image, mask in tqdm(val_loader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            pred = net(image)

            # Round predictions to the nearest integer (0 or 1)
            # to form a binary mask
            pred = torch.round(pred)
            pixel_acc += pred.eq(mask).sum().item()
        pixel_acc /= n_pixels
        pixel_acc /= len(data)

        writer.add_scalar("Pixel accuracy/Loss/epoch", pixel_acc, running_loss, epoch)

        # Serialize model parameters to disk every epoch as checkpoint
        torch.save(net.state_dict(), SAVED_MODEL)

        # Evaluate model's effect on one random picture
        image, mask = data[1]
        # Add a batch dimension (=1) at the front
        image = torch.unsqueeze(image, dim=0).to(DEVICE)
        net.eval()
        out = net(image)
        # Remove batch dimension and channel dimension
        out = torch.squeeze(out)
        # Move tensor values to CPU
        out = out.detach().numpy(force=True)
        # Convert to gray scale
        out2 = (out * 255).astype(np.uint8)
        # Convert to binary mask
        out = np.rint(out).astype(np.uint8) * 255
        out = Image.fromarray(out)
        out.save(R'ep{}_mask.png'.format(epoch))
        out2 = Image.fromarray(out2)
        out2.save(R'ep{}_predicted.png'.format(epoch))
        print("Completed training epoch {} with loss = {} and pixel accuracy = {}"
            .format(epoch, running_loss, pixel_acc))

    writer.close()
