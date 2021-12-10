import torch.nn as nn


architectures = {
    "resnet18": {"blocks": (2, 2, 2, 2)},
    "resnet34": {"blocks": (3, 4, 6, 3)},
}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        input = x
        if self.downsample:
            input = self.downsample(input)
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = output + input
        output = self.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, architecture, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.stem = self.add_stem()
        blocks = architectures[architecture]["blocks"]
        self.stage1 = self.add_stage(
            in_channels=64, out_channels=64, stride=1, blocks_count=blocks[0]
        )
        self.stage2 = self.add_stage(
            in_channels=64, out_channels=128, stride=2, blocks_count=blocks[1]
        )
        self.stage3 = self.add_stage(
            in_channels=128, out_channels=256, stride=2, blocks_count=blocks[2]
        )
        self.stage4 = self.add_stage(
            in_channels=256, out_channels=512, stride=2, blocks_count=blocks[3]
        )
        self.classifier = self.add_classifier()
        self.initialize()

    def add_stem(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def add_stage(self, in_channels, out_channels, stride, blocks_count):
        blocks = []
        blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for i in range(1, blocks_count):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def add_classifier(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, self.num_classes)
        )

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        result = self.stem(x)
        result = self.stage1(result)
        result = self.stage2(result)
        result = self.stage3(result)
        result = self.stage4(result)
        result = self.classifier(result)
        return result
