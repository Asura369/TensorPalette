import torch
import torch.nn as nn

from styleforge.vgg import Vgg16


def adain(content_feat, style_feat):
    content_mean, content_std = content_feat.mean(dim=[2, 3], keepdim=True), content_feat.std(dim=[2, 3], keepdim=True)
    style_mean, style_std = style_feat.mean(dim=[2, 3], keepdim=True), style_feat.std(dim=[2, 3], keepdim=True)
    content_std = content_std.clamp(min=1e-6)
    style_std = style_std.clamp(min=1e-6)
    return style_std * (content_feat - content_mean) / content_std + style_mean


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        out = self.pad(x)
        out = self.conv(out)
        out = self.relu(out)
        return out


class AdaINDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = DecoderBlock(512, 256, upsample=True)
        self.dec2 = DecoderBlock(256, 256, upsample=False)
        self.dec3 = DecoderBlock(256, 256, upsample=False)
        self.dec4 = DecoderBlock(256, 128, upsample=True)
        self.dec5 = DecoderBlock(128, 128, upsample=False)
        self.dec6 = DecoderBlock(128, 64, upsample=True)
        self.dec7 = DecoderBlock(64, 64, upsample=False)
        self.pad_out = nn.ReflectionPad2d(1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.pad_out(x)
        x = self.conv_out(x)
        return x


class AdaINModel(nn.Module):
    def __init__(self, vgg_path="models/vgg16.pth"):
        super().__init__()
        self.encoder = Vgg16(requires_grad=True, vgg_path=vgg_path)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = AdaINDecoder()

    def forward(self, content, style, alpha=1.0):
        content_feats = self.encoder(content * 255.0)
        style_feats = self.encoder(style * 255.0)
        t = adain(content_feats.relu4_3, style_feats.relu4_3)
        if alpha < 1.0:
            t = alpha * t + (1 - alpha) * content_feats.relu4_3
        return self.decoder(t)
