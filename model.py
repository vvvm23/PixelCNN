import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    https://github.com/tuelwer/conditional-pixelcnn-pytorch
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

class ConditionalCNNBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        out_channels = args[2]

        self.mconv1 = MaskedCNN(*args, **kwargs)
        self.mconv2 = MaskedCNN(*args, **kwargs)

        self.cconv1 = nn.Conv2d(1, out_channels, 1)
        self.cconv2 = nn.Conv2d(1, out_channels, 1)

    def forward(self, x, h):
        inp_gated = torch.tanh(self.mconv1(x)) * torch.sigmoid(self.mconv2(x))
        con_gated = torch.tanh(self.cconv1(h)) * torch.sigmoid(self.cconv2(h))
        return inp_gated + con_gated

class Embedding2d(nn.Module):
    def __init__(self, nb_classes, out_shape):
        super().__init__()
        self.nb_classes = nb_classes
        self.out_shape = out_shape

        self.emb = nn.Embedding(nb_classes, torch.prod(torch.tensor(out_shape)))

    def forward(self, x):
        return self.emb(x).view(-1, 1, *self.out_shape)

class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    embedding = Embedding2d(10, (28, 28))
    cconv = ConditionalCNNBlock('A', 1, 8, 3, padding=1)
    
    h = torch.tensor(5).view(1, -1)
    x = torch.randn(1, 1, 28, 28)

    print(cconv(x, embedding(h)).shape)
