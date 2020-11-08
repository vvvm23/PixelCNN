import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    https://github.com/tuelwer/conditional-pixelcnn-pytorch
    https://arxiv.org/abs/1606.05328
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

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, h):
        inp_gated = torch.tanh(self.mconv1(x)) * torch.sigmoid(self.mconv2(x))
        con_gated = torch.tanh(self.cconv1(h)) * torch.sigmoid(self.cconv2(h))
        return self.bn(inp_gated + con_gated)

class Embedding2d(nn.Module):
    def __init__(self, nb_classes, out_shape):
        super().__init__()
        self.nb_classes = nb_classes
        self.out_shape = out_shape

        self.emb = nn.Embedding(nb_classes, torch.prod(torch.tensor(out_shape)))

    def forward(self, x):
        return self.emb(x).view(-1, *self.out_shape)

class PixelCNN(nn.Module):
    def __init__(self, in_shape, nb_channels, nb_layers, nb_out, nb_classes):
        super().__init__()
        nb_in = in_shape[0]

        self.emb = Embedding2d(nb_classes, in_shape)
        self.layers = nn.ModuleList()

        self.layers.append(ConditionalCNNBlock('A', nb_in, nb_channels, 7, padding=3))
        for _ in range(0, nb_layers-1):
            self.layers.append(ConditionalCNNBlock('B', nb_channels, nb_channels, 7, 1, 3))
        self.layers.append(nn.Conv2d(nb_channels, nb_out, 1))

    def forward(self, x, h):
        h = self.emb(h)
        for l in self.layers:
            if isinstance(l, ConditionalCNNBlock):
                x = l(x, h)
            else:
                x = l(x)
        return x

if __name__ == "__main__":
    pixelcnn = PixelCNN((1, 28, 28), 16, 7, 1, 10)
    x, h = torch.randn(2, 1, 28, 28), torch.tensor([0,0])
    print(pixelcnn(x,h).shape)
