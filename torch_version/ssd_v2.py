import torch

from torch.nn import functional as F

from torch.nn import (
    Conv2d,
    Linear,
    Flatten,
    AdaptiveAvgPool2d,
    ZeroPad2d,
    MaxPool2d
)


class SSD300v2(torch.nn.Module):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image (3, 300, 300).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    def __init__(self, input_shape, num_classes=21):
        super(SSDv2, self).__init__()

        # Block 1
        self.conv1_1 = Conv2d(3, 64, (3, 3), padding=(1, 1))
        self.conv1_2 = Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2

        self.conv2_1 = Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.conv2_2 = Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 3
        self.conv3_1 = Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.conv3_2 = Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.conv3_3 = Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        # Block 4
        self.conv4_1 = Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.conv4_2 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv4_3 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 5
        self.conv5_1 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv5_2 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv5_3 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.pool5 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # FC6
        self.fc6 = Conv2d(512, 1024, (3, 3), dilation=(6, 6), padding=(6, 6))

        # FC7
        self.fc7 = Conv2d(1024, 1024, (1, 1), padding=(1, 1))

        # Block 6
        self.conv6_1 = Conv2d(1024, 256, (1, 1), padding=(0, 0))
        self.conv6_2 = Conv2d(256, 512, (3, 3), stride=(2, 2))

        # Block 7
        self.conv7_1 = Conv2d(512, 128, (1, 1), padding=(0, 0))
        self.conv7_1z = ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv7_2 = Conv2d(128, 256, (3, 3), padding='valid', stride=(2, 2))

        # Block 8
        self.conv8_1 = Conv2d(256, 128, (1, 1), padding=(0, 0))
        self.conv8_2 = Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(2, 2))

        self.pool6 = AdaptiveAvgPool2d((1, 1))

    def forward(self, input_object):

        def conv_activation(obj, conv, activ):
            return activ(conv(obj), inplace=True)

        conv1_1 = conv_activation(input_object, self.conv1_1, F.relu)
        conv1_2 = conv_activation(conv1_1, self.conv1_2, F.relu)
        pool1 = self.pool1(conv1_2)

        conv2_1 = conv_activation(pool1, self.conv2_1, F.relu)
        conv2_2 = conv_activation(conv2_1, self.conv2_2, F.relu)
        pool2 = self.pool2(conv2_2)

        conv3_1 = conv_activation(pool2, self.conv3_1, F.relu)
        conv3_2 = conv_activation(conv3_1, self.conv3_2, F.relu)
        conv3_3 = conv_activation(conv3_2, self.conv3_3, F.relu)
        pool3 = self.pool3(conv3_3)

        conv4_1 = conv_activation(pool3, self.conv4_1, F.relu)
        conv4_2 = conv_activation(conv4_1, self.conv4_2, F.relu)
        conv4_3 = conv_activation(conv4_2, self.conv4_3, F.relu)
        pool4 = self.pool4(conv4_3)

        conv5_1 = conv_activation(pool4, self.conv5_1, F.relu)
        conv5_2 = conv_activation(conv5_1, self.conv5_2, F.relu)
        conv5_3 = conv_activation(conv5_2, self.conv5_3, F.relu)
        pool5 = self.pool5(conv5_3)

        fc6 = conv_activation(pool5, self.fc6, F.relu)
        fc7 = conv_activation(fc6, self.fc7, F.relu)

        conv6_1 = conv_activation(fc7, self.conv6_1, F.relu)
        conv6_2 = conv_activation(conv6_1, self.conv6_2, F.relu)

        conv7_1 = conv_activation(conv6_2, self.conv7_1, F.relu)
        conv7_1z = self.conv7_1z(conv7_1) # zero padding
        conv7_2 = conv_activation(conv7_1z, self.conv7_2, F.relu)

        conv8_1 = conv_activation(conv7_2, self.conv8_1, F.relu)
        conv8_2 = conv_activation(conv8_1, self.conv8_2, F.relu)

        pool6 = self.pool6(conv8_2).view(input_object.size(0), -1)
        return pool6

if __name__ == '__main__':
    from torchsummary import summary
    model = SSDv2((300, 300, 3))
    summary(model, (3, 300, 300))