import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super(Block, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)  # padding = same
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.sbn = nn.BatchNorm2d(out_dim)
        self.sc = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        result = self.bn1(self.conv1(x))
        result = F.relu(result)
        result = F.relu(self.bn2(self.conv2(result)))

        if self.in_dim != self.out_dim or self.stride != 1:
            result = result + self.sbn(self.sc(x))
        else:
            result = result + x

        return result


class ResNet(nn.Module):
    # len(channel_num) must be block_num + 1 == len(strides) + 1
    def __init__(self, n, block_num=3, channel_num=[16, 16, 32, 64], block_strides=[1, 2, 2], category_num=10):
        super(ResNet, self).__init__()
        self.block_num = block_num
        self.channel_num = channel_num
        self.block_strides = block_strides
        self.blocks = []
        self.conv1 = nn.Conv2d(3, self.channel_num[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channel_num[0])
        self.fc = nn.Linear(self.channel_num[-1], category_num)
        for block in range(block_num):
            self.blocks.append(self.make_layers(channel_num[block], channel_num[block + 1], n, block_strides[block]))

        new_module = list()
        new_module.append(self.conv1)
        new_module.append(self.bn1)
        for block in range(block_num):
            new_module.append(self.blocks[block])
        new_module.append(self.fc)

        self.modules = nn.ModuleList(new_module)

    def make_layers(self, in_dim, out_dim, count, stride):
        layer_list = []
        for c in range(count):
            if c == 0:
                layer_list.append(Block(in_dim, out_dim, stride))
            else:
                layer_list.append(Block(out_dim, out_dim, 1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        for block in self.blocks:
            x = block(x)

        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = x.view(-1, self.channel_num[-1])
        x = self.fc(x)

        return x

    def to_device(self, device):
        self.to(device)
        for block in range(len(self.blocks)):
            self.blocks[block] = self.blocks[block].to(device)
