import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels//2
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down3D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#-------------------------------------------


class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Up2D,self).__init__()

        self.conv = conv_builder(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=False)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Up3D,self).__init__()
        
        self.conv = conv_builder(in_channels, out_channels, out_channels)
        self.upsample = nn.ConvTranspose3d(2*out_channels, 2*out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x1, x2):
        # x1 = F.interpolate(x1,scale_factor=2,mode='trilinear', align_corners=False)
        # # input is CDHW
        # diffD = x2.size()[2] - x1.size()[2]
        # diffH = x2.size()[3] - x1.size()[3]
        # diffW = x2.size()[4] - x1.size()[4]
        
        # ######## 需要是upconv
        # x1 = F.pad(x1, [diffD // 2, diffD - diffD // 2,
        #                 diffH // 2, diffH - diffH // 2,
        #                 diffW // 2, diffW - diffW // 2])
        x1 = self.upsample(x1)
        output_size = x1.size()[-3:]
        x2_size = x2.size()[-3:]
        if output_size != x2_size:
            x1 = F.interpolate(x1, size=x2_size)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Tail3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class UNet(nn.Module):
    def __init__(self, stem=DoubleConv3D, down=Down3D, up=Up3D, tail=Tail3D, width=[32,64,128,256,512], conv_builder=DoubleConv3D, n_channels=1, n_classes=2, dropout_flag=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = width
        self.dropout_flag = dropout_flag

        self.inc = stem(n_channels, width[0], width[0]//2)   # n_channels ->32 -> 64
        self.down1 = down(width[0], width[1], conv_builder)  # 64->64->128
        self.down2 = down(width[1], width[2], conv_builder)  # 128->128->256
        self.down3 = down(width[2], width[3], conv_builder)  # 256->256->512
        # self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[3]+width[2], width[2] , conv_builder) # 768->256->256
        self.up2 = up(width[2]+width[1], width[1] , conv_builder)
        self.up3 = up(width[1]+width[0], width[0] , conv_builder)
        #self.up4 = up(width[1], width[0], conv_builder)
        self.dropout = nn.Dropout(p=0.1)
        self.outc = tail(width[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print(x1.shape,"x1.shape")
        x2 = self.down1(x1)
        #print(x2.shape,"x2.shape")
        x3 = self.down2(x2)
        #print(x3.shape,"x3.shape")
        x4 = self.down3(x3)
        #print(x4.shape,"x4.shape")
        # x5 = self.down4(x4)

        x = self.up1(x4, x3)
        #print(x.shape,"upx1.shape")
        x = self.up2(x, x2)
        #print(x.shape,"upx2.shape")
        x = self.up3(x, x1)
        #print(x.shape,"upx3.shape")
        # x = self.up4(x, x1)
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)

        #print("1")
        return logits