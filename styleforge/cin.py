import torch
import torch.nn as nn


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_styles, channels):
        super().__init__()
        self.num_styles = num_styles
        self.channels = channels
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.gamma = nn.Parameter(torch.ones(num_styles, channels))
        self.beta = nn.Parameter(torch.zeros(num_styles, channels))

    def forward(self, x, style_id):
        out = self.instance_norm(x)
        g = self.gamma[style_id].unsqueeze(-1).unsqueeze(-1)
        b = self.beta[style_id].unsqueeze(-1).unsqueeze(-1)
        return out * (1 + g) + b

    def forward_interpolated(self, x, style_id_a, style_id_b, alpha):
        out = self.instance_norm(x)
        g = ((1 - alpha) * self.gamma[style_id_a] + alpha * self.gamma[style_id_b])
        b = ((1 - alpha) * self.beta[style_id_a] + alpha * self.beta[style_id_b])
        g = g.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)
        return out * (1 + g) + b


class CINResidualBlock(nn.Module):
    def __init__(self, num_styles, channels):
        super().__init__()
        from styleforge.transformer import ConvLayer
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.cin1 = ConditionalInstanceNorm2d(num_styles, channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.cin2 = ConditionalInstanceNorm2d(num_styles, channels)
        self.relu = nn.ReLU()

    def forward(self, x, style_id):
        residual = x
        out = self.relu(self.cin1(self.conv1(x), style_id))
        out = self.cin2(self.conv2(out), style_id)
        return out + residual

    def forward_interpolated(self, x, style_id_a, style_id_b, alpha):
        residual = x
        out = self.relu(self.cin1.forward_interpolated(self.conv1(x), style_id_a, style_id_b, alpha))
        out = self.cin2.forward_interpolated(self.conv2(out), style_id_a, style_id_b, alpha)
        return out + residual


class CINTransformer(nn.Module):
    def __init__(self, num_styles):
        super().__init__()
        self.num_styles = num_styles

        from styleforge.transformer import ConvLayer, UpsampleConvLayer

        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.cin1 = ConditionalInstanceNorm2d(num_styles, 32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.cin2 = ConditionalInstanceNorm2d(num_styles, 64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.cin3 = ConditionalInstanceNorm2d(num_styles, 128)

        self.res1 = CINResidualBlock(num_styles, 128)
        self.res2 = CINResidualBlock(num_styles, 128)
        self.res3 = CINResidualBlock(num_styles, 128)
        self.res4 = CINResidualBlock(num_styles, 128)
        self.res5 = CINResidualBlock(num_styles, 128)

        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.cin4 = ConditionalInstanceNorm2d(num_styles, 64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.cin5 = ConditionalInstanceNorm2d(num_styles, 32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x, style_id):
        y = self.relu(self.cin1(self.conv1(x), style_id))
        y = self.relu(self.cin2(self.conv2(y), style_id))
        y = self.relu(self.cin3(self.conv3(y), style_id))
        y = self.res1(y, style_id)
        y = self.res2(y, style_id)
        y = self.res3(y, style_id)
        y = self.res4(y, style_id)
        y = self.res5(y, style_id)
        y = self.relu(self.cin4(self.deconv1(y), style_id))
        y = self.relu(self.cin5(self.deconv2(y), style_id))
        y = self.deconv3(y)
        return y

    def forward_interpolated(self, x, style_id_a, style_id_b, alpha):
        y = self.relu(self.cin1.forward_interpolated(self.conv1(x), style_id_a, style_id_b, alpha))
        y = self.relu(self.cin2.forward_interpolated(self.conv2(y), style_id_a, style_id_b, alpha))
        y = self.relu(self.cin3.forward_interpolated(self.conv3(y), style_id_a, style_id_b, alpha))
        y = self.res1.forward_interpolated(y, style_id_a, style_id_b, alpha)
        y = self.res2.forward_interpolated(y, style_id_a, style_id_b, alpha)
        y = self.res3.forward_interpolated(y, style_id_a, style_id_b, alpha)
        y = self.res4.forward_interpolated(y, style_id_a, style_id_b, alpha)
        y = self.res5.forward_interpolated(y, style_id_a, style_id_b, alpha)
        y = self.relu(self.cin4.forward_interpolated(self.deconv1(y), style_id_a, style_id_b, alpha))
        y = self.relu(self.cin5.forward_interpolated(self.deconv2(y), style_id_a, style_id_b, alpha))
        y = self.deconv3(y)
        return y
