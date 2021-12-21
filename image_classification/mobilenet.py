import math
import torch.nn as nn


def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def conv_3x3_bn(input_channels, output_channels, stride):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        HSwish(),
    )


def conv_1x1_bn(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        HSwish(),
    )


class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, _make_divisible(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channels // reduction, 8), channels),
            HSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, kernel_size, stride, use_se, use_hs
    ):
        super().__init__()
        self.equal_dim = stride == 1 and input_dim == output_dim
        if input_dim == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(
                    hidden_dim,
                    output_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(output_dim),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    output_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(output_dim),
            )

    def forward(self, x):
        if self.equal_dim:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, config, mode, width_multiplier=1.0, num_classes=1000):
        super().__init__()
        self.config = config
        input_channels = _make_divisible(16 * width_multiplier, 8)
        layers = [conv_3x3_bn(3, input_channels, 2)]
        hidden_channels = 0
        for kernel, multiplier, output, use_se, use_hs, stride in self.config:
            output_channel = _make_divisible(output * width_multiplier, 8)
            hidden_channels = _make_divisible(input_channels * multiplier, 8)
            layers.append(
                InvertedResidual(
                    input_channels,
                    hidden_channels,
                    output_channel,
                    kernel,
                    stride,
                    use_se,
                    use_hs,
                )
            )
            input_channels = output_channel
        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channels, hidden_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {"large": 1280, "small": 1024}
        output_channel = (
            _make_divisible(output_channel[mode] * width_multiplier, 8)
            if width_multiplier > 1.0
            else output_channel[mode]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, output_channel),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenetv3_large(num_classes=1000):
    config = [
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1],
    ]
    return MobileNetV3(config, mode="large", num_classes=num_classes)


def mobilenetv3_small(num_classes=1000):
    config = [
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]
    return MobileNetV3(config, mode="small", num_classes=num_classes)

