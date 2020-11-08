import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
"""
class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(mask_type in ['A', 'B'])

        self.register_buffer('mask', torch.ones_like(self.weight.data))
        kh, kw = self.kernel_size

        self.mask[:, :, kh // 2, kw // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kh // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    cnn = MaskedCNN('A', 3, 16, 3, stride=1, padding=1)
    x = torch.randn(1, 3, 8, 8)
    print(cnn(x).shape)
