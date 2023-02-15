import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv4x4(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class deconv4x4(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, feat):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, feat), dim=1)


class MLAttrEncoder(nn.Module):
    def __init__(self):
        super(MLAttrEncoder, self).__init__()
        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)

        self.deconv1 = deconv4x4(1024, 1024)
        self.deconv2 = deconv4x4(2048, 512)
        self.deconv3 = deconv4x4(1024, 256)
        self.deconv4 = deconv4x4(512, 128)
        self.deconv5 = deconv4x4(256, 64)
        self.deconv6 = deconv4x4(128, 32)

        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt) # Xt: 3x256x256, feat1: 32x128x128
        feat2 = self.conv2(feat1) # feat2: 64x64x64
        feat3 = self.conv3(feat2) # feat3: 128x32x32
        feat4 = self.conv4(feat3) # feat4: 256x16x16
        feat5 = self.conv5(feat4) # feat5: 512x8x8
        feat6 = self.conv6(feat5) # feat6: 1024x4x4

        z_att1 = self.conv7(feat6) # z_att1: 1024x2x2

        z_att2 = self.deconv1(z_att1, feat6) # z_att2: 2048 x 4 x 4
        z_att3 = self.deconv2(z_att2, feat5) # z_att3: 1024 x 8 x 8
        z_att4 = self.deconv3(z_att3, feat4) # z_att4: 512 x 16 x 16
        z_att5 = self.deconv4(z_att4, feat3) # z_att5: 256 x 32 x 32
        z_att6 = self.deconv5(z_att5, feat2) # z_att6: 128 x 64 x 64
        z_att7 = self.deconv6(z_att6, feat1) # z_att7: 64 x 128 x 128

        z_att8 = F.interpolate(z_att7, scale_factor=2, mode='bilinear', align_corners=True) # z_att7: 64 x 256 x 256
        return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8