import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, in_channel=3, mid_channel=64):
        super(CNN, self).__init__()

        self.input_layer = nn.Conv2d(
            in_channels=in_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1)
        self.input_bn = nn.BatchNorm2d(mid_channel)

        self.layer1 = nn.Conv2d(
            in_channels=mid_channel, out_channels=mid_channel*2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channel*2)

        self.layer2 = nn.Conv2d(
            in_channels=mid_channel*2, out_channels=mid_channel*4, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel*4)

        self.layer3 = nn.Conv2d(
            in_channels=mid_channel*4, out_channels=mid_channel*8, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channel*8)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.output_layer = nn.Conv2d(
            in_channels=mid_channel*8, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(mid_channel*8)

        self.relu = nn.ReLU()

    def forward(self, x_data):
        h = self.input_layer(x_data)
        h = self.relu(self.input_bn(h))

        h = self.layer1(h)
        h = self.relu(self.bn1(h))

        h = self.layer2(h)
        h = self.relu(self.bn2(h))

        h = self.layer3(h)
        h = self.relu(self.bn3(h))

        h = self.avg_pool(h)

        h = self.output_layer(h)
        # h = self.relu(self.bn(h))

        return h.squeeze().unsqueeze(1)


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    net = CNN(in_channel=3, mid_channel=64)

    h = net(x)
    print(h.size())
