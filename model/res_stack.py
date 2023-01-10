import torch
import torch.nn as nn

from model import streaming as streaming
def casual_sum(x):
    x1, x2 = x
    return x1 + x2

class ResStack(nn.Module):
    def __init__(self, channel, is_streaming, dilation=1):
        super(ResStack, self).__init__()

        if is_streaming:
            conv1d = streaming.StreamingConv1d
            reflectionPad1d = streaming.StreamingReflectionPad1d
            self.sum = streaming.StreamingSum()
        else:
            conv1d = torch.nn.Conv1d
            reflectionPad1d = torch.nn.ReflectionPad1d
            self.sum = casual_sum
        
        self.block = nn.Sequential(
                nn.LeakyReLU(0.2),
                reflectionPad1d(padding=dilation),
                nn.utils.weight_norm(conv1d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=dilation)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(conv1d(in_channels=channel, out_channels=channel, kernel_size=1)),
            )
           

        self.shortcut = nn.utils.weight_norm(conv1d(in_channels=channel, out_channels=channel, kernel_size=1))
            

    def forward(self, x):
        return self.sum([self.shortcut(x), self.block(x)])

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.block[2])
        nn.utils.remove_weight_norm(self.block[4])
        nn.utils.remove_weight_norm(self.shortcut)
        # def _remove_weight_norm(m):
        #     try:
        #         torch.nn.utils.remove_weight_norm(m)
        #     except ValueError:  # this module didn't have weight norm
        #         return
        #
        # self.apply(_remove_weight_norm)
