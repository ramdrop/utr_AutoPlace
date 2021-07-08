import torch
import torch.nn.functional as F
from networks.layers import DoubleConv, OutConv, Down, Up

class UNet(torch.nn.Module):
    """ The UNet network, code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """
    def __init__(self, config):
        super().__init__()
        bilinear = config['networks']['unet']['bilinear']
        first_feature_dimension = config['networks']['unet']['first_feature_dimension']
        self.score_sigmoid = config['networks']['unet']['score_sigmoid']
        # check for steam 2x2 weight matrix setting
        outc_score_dim = 1
        if 'weight_matrix' in config['steam']:
            outc_score_dim = 3 if config['steam']['weight_matrix'] is True else 1
        # down
        input_channels = 1
        self.inc = DoubleConv(input_channels, first_feature_dimension)
        self.down1 = Down(first_feature_dimension, first_feature_dimension * 2)
        self.down2 = Down(first_feature_dimension * 2, first_feature_dimension * 4)
        self.down3 = Down(first_feature_dimension * 4, first_feature_dimension * 8)
        self.down4 = Down(first_feature_dimension * 8, first_feature_dimension * 16)

        self.up1_pts = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        self.up2_pts = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_pts = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_pts = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_pts = OutConv(first_feature_dimension, 1)

        self.up1_score = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        self.up2_score = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_score = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_score = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_score = OutConv(first_feature_dimension, outc_score_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """ A U-Net style network is used to output dense detector scores, weight scores, and
            a descriptor map with the same spatial dimensions as the input.
        Args:
            x (torch.tensor): (b*w,1,H,W) input 2D data
        Returns:
            detector_scores (torch.tensor): (b*w,1,H,W)
            weight_scores (torch.tensor): (b*w,S,H,W)
            descriptors (torch.tensor): (b*w,C,H,W)
        """
        _, _, height, width = x.size()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_up_pts = self.up1_pts(x5, x4)
        x3_up_pts = self.up2_pts(x4_up_pts, x3)
        x2_up_pts = self.up3_pts(x3_up_pts, x2)
        x1_up_pts = self.up4_pts(x2_up_pts, x1)
        detector_scores = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        weight_scores = self.outc_score(x1_up_score)
        if self.score_sigmoid:
            weight_scores = self.sigmoid(weight_scores)

        # Resize outputs of downsampling layers to the size of the original
        # image. Features are interpolated using bilinear interpolation to
        # get gradients for back-prop. Concatenate along the feature channel
        # to get pixel-wise descriptors of size Bx248xHxW
        f1 = F.interpolate(x1, size=(height, width), mode='bilinear')
        f2 = F.interpolate(x2, size=(height, width), mode='bilinear')
        f3 = F.interpolate(x3, size=(height, width), mode='bilinear')
        f4 = F.interpolate(x4, size=(height, width), mode='bilinear')
        f5 = F.interpolate(x5, size=(height, width), mode='bilinear')

        feature_list = [f1, f2, f3, f4, f5]
        descriptors = torch.cat(feature_list, dim=1)

        return detector_scores, weight_scores, descriptors
