import torch.nn as nn
import torch
import torch.nn.functional as F


class unet_Conv_Block(nn.Module):
    """
    The convolutional or downsampling block in U-Net
    """    
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(unet_Conv_Block, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1), 
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
                nn.ReLU()
            )

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout(outputs)
        return outputs


class unetUp(nn.Module):
    """
    the upsampling block in U-Net
    """
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unet_Conv_Block(in_size, out_size, False)

        if is_deconv:
            # the size of inputs will be amplified 2x
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            # ! ?
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # # '*' operation can extend the list
        # padding = 2 * [offset // 2, offset // 2]  # the list has 4 elements
        # outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([inputs1, outputs2], 1))


class unet(nn.Module):
    def __init__(self,
                 feature_scale=4,
                 n_classes=34,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True):
        """the U-Net model

        Args:
            feature_scale (int, optional): scale factor of the original filter numbers. Defaults to 4.
            n_classes (int, optional): the number of classes. Defaults to 34.
            is_deconv (bool, optional): if True, ConvTranspose2d() will be used for upsampling; Otherwise, UpsamplingBilinear2d() will be used. Defaults to True.
            in_channels (int, optional): Defaults to 3.
            is_batchnorm (bool, optional): if True, add a batchnorm layer after each conv layer. Defaults to True.
        """        
        super(unet, self).__init__()
        self.num_classes = n_classes
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # the original filter numbers in the paper
        filters = [64, 128, 256, 512, 1024]
        # scale the filter numbers to accelerate training
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unet_Conv_Block(self.in_channels, filters[0],
                                     self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = unet_Conv_Block(filters[0], filters[1],
                                              self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = unet_Conv_Block(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = unet_Conv_Block(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = unet_Conv_Block(filters[3], filters[4],
                                      self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        # 1/1
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # 1/2
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # 1/4
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # 1/8
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # 1/16
        center = self.center(maxpool4)
        # 1/16
        up4 = self.up_concat4(conv4, center)
        # 1/8
        up3 = self.up_concat3(conv3, up4)
        # 1/4
        up2 = self.up_concat2(conv2, up3)
        # 1/2
        up1 = self.up_concat1(conv1, up2)
        # 1/1
        final = self.final(up1)

        return final
