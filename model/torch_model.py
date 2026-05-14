# -*- coding: utf-8 -*-
"""
PyTorch version of the original Keras/TensorFlow DTCWT model.

Input format:
    PyTorch NCHW: [B, 3, H, W]

Dependencies:
    pip install torch pytorch_wavelets

Notes:
    - The original code uses dtcwt.tf.Transform2d with NHWC tensors.
    - This version uses pytorch_wavelets.DTCWTForward / DTCWTInverse.
    - High-pass coefficients are flattened to match the original 36-channel layout:
      real RGB x 6 orientations + imag RGB x 6 orientations = 36 channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
except ImportError as exc:
    raise ImportError(
        "Please install pytorch_wavelets first: pip install pytorch_wavelets"
    ) from exc


# -----------------------------
# Basic blocks
# -----------------------------

def kaiming_init(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SameConv2d(nn.Module):
    """Keras-like Conv2D(padding='same') for stride 1/2."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        out_h = (h + self.stride[0] - 1) // self.stride[0]
        out_w = (w + self.stride[1] - 1) // self.stride[1]

        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class SameConvTranspose2d(nn.Module):
    """Approximate Keras Conv2DTranspose(padding='same', strides=2)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        target_h = x.shape[-2] * self.stride
        target_w = x.shape[-1] * self.stride
        y = self.deconv(x)
        return center_crop_or_pad(y, target_h, target_w)


def center_crop_or_pad(x, target_h, target_w):
    h, w = x.shape[-2:]

    if h > target_h:
        top = (h - target_h) // 2
        x = x[..., top:top + target_h, :]
    elif h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        x = F.pad(x, (0, 0, pad_top, pad_bottom))

    h, w = x.shape[-2:]
    if w > target_w:
        left = (w - target_w) // 2
        x = x[..., :, left:left + target_w]
    elif w < target_w:
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        x = F.pad(x, (pad_left, pad_right, 0, 0))

    return x


class SliceLayer(nn.Module):
    def __init__(self, edge):
        super().__init__()
        self.edge = edge

    def forward(self, x):
        if self.edge == 0:
            return x
        return x[..., self.edge:-self.edge, self.edge:-self.edge]


class BoundReLU(nn.Module):
    def __init__(self, maxvalue, thres=0.0):
        super().__init__()
        self.maxvalue = float(maxvalue)
        self.thres = float(thres)

    def forward(self, x):
        return torch.clamp(F.relu(x - self.thres), max=self.maxvalue)


# -----------------------------
# DTCWT wrappers
# -----------------------------

class DTCWTLayer(nn.Module):
    """
    Returns:
        low: [B, C, H/2, W/2]
        high_0_flat: [B, C*12, H/2, W/2]
        high_1_flat: [B, C*12, H/4, W/4]
    """

    def __init__(self, nlevels=2):
        super().__init__()
        self.xfm = DTCWTForward(J=nlevels, biort="near_sym_b", qshift="qshift_b")

    @staticmethod
    def flatten_high(yh):
        # pytorch_wavelets DTCWT highpass commonly uses shape:
        # [B, C, 6, H, W, 2], where last dim is real/imag.
        if yh.dim() != 6:
            raise ValueError(f"Unexpected DTCWT highpass shape: {tuple(yh.shape)}")

        if yh.shape[-1] == 2:
            real = yh[..., 0]      # [B, C, 6, H, W]
            imag = yh[..., 1]
        elif yh.shape[2] == 2:
            real = yh[:, :, 0]     # fallback for [B, C, 2, 6, H, W]
            imag = yh[:, :, 1]
        else:
            raise ValueError(f"Cannot locate real/imag axis in shape: {tuple(yh.shape)}")

        # [B, C, 6, H, W] -> [B, 6, C, H, W] -> [B, 6*C, H, W]
        real = real.permute(0, 2, 1, 3, 4).contiguous().flatten(1, 2)
        imag = imag.permute(0, 2, 1, 3, 4).contiguous().flatten(1, 2)
        return torch.cat([real, imag], dim=1)

    def forward(self, x):
        low, highs = self.xfm(x)
        high_0 = self.flatten_high(highs[0])
        high_1 = self.flatten_high(highs[1])
        return low, high_0, high_1


class InverseDTCWTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifm = DTCWTInverse(biort="near_sym_b", qshift="qshift_b")

    @staticmethod
    def unflatten_high(high, channels=3):
        # high: [B, C*12, H, W]
        b, c12, h, w = high.shape
        assert c12 == channels * 12, f"Expected {channels * 12} channels, got {c12}"

        half = c12 // 2
        real = high[:, :half].reshape(b, 6, channels, h, w).permute(0, 2, 1, 3, 4)
        imag = high[:, half:].reshape(b, 6, channels, h, w).permute(0, 2, 1, 3, 4)
        return torch.stack([real, imag], dim=-1).contiguous()  # [B, C, 6, H, W, 2]

    def forward(self, low, high_0, high_1):
        channels = low.shape[1]
        yh0 = self.unflatten_high(high_0, channels=channels)
        yh1 = self.unflatten_high(high_1, channels=channels)
        return self.ifm((low, [yh0, yh1]))


# -----------------------------
# Network branches
# -----------------------------

class RefineH(nn.Module):
    def __init__(self, input_channels, inputD_channels, dropout=0.0):
        super().__init__()
        self.act05 = nn.LeakyReLU(0.5, inplace=True)
        self.drop = nn.Dropout2d(dropout)

        self.L1 = SameConv2d(inputD_channels, 64, 2, stride=2)
        self.L2 = SameConv2d(inputD_channels, 64, 3, stride=2)
        self.L3 = SameConv2d(inputD_channels, 64, 5, stride=2)
        self.L_bn = nn.BatchNorm2d(192)

        self.Lc1 = SameConv2d(192, 64, 2)
        self.Lc2 = SameConv2d(192, 64, 3)
        self.Lc3 = SameConv2d(192, 64, 5)
        self.Lc4 = SameConv2d(192, 64, 9)
        self.Lc_bn = nn.BatchNorm2d(256)

        self.x1 = SameConv2d(input_channels, 54, 2, stride=2)
        self.x2 = SameConv2d(input_channels, 54, 3, stride=2)
        self.x3 = SameConv2d(input_channels, 54, 5, stride=2)
        self.x_bn = nn.BatchNorm2d(162)

        self.xc1 = SameConv2d(162, 18, 2)
        self.xc2 = SameConv2d(162, 18, 3)
        self.xc3 = SameConv2d(162, 18, 5)
        self.xc_bn = nn.BatchNorm2d(54)

        self.m1 = SameConv2d(54 + 256, 48, 2)
        self.m2 = SameConv2d(54 + 256, 48, 3)
        self.m3 = SameConv2d(54 + 256, 48, 5)
        self.m_bn = nn.BatchNorm2d(144)

        self.de1 = SameConvTranspose2d(144, 32, 2, stride=2)
        self.de2 = SameConvTranspose2d(144, 32, 3, stride=2)
        self.de3 = SameConvTranspose2d(144, 32, 7, stride=2)

        self.gcn_11 = SameConv2d(96, 64, (15, 1))
        self.gcn_12 = SameConv2d(96, 64, (1, 15))
        self.gcn_21 = SameConv2d(64, 18, (1, 15))
        self.gcn_22 = SameConv2d(64, 18, (15, 1))

        self.conv1 = SameConv2d(36, 36, 3)
        self.conv2 = SameConv2d(36, 36, 3)
        self.c1 = SameConv2d(36, 36, 3)
        self.c2 = SameConv2d(36, 36, 3)
        self.out_conv = SameConv2d(36, input_channels, 3)
        self.prelu = nn.PReLU(num_parameters=input_channels)

    def forward(self, x, inputD):
        init = x

        L = torch.cat([self.L1(inputD), self.L2(inputD), self.L3(inputD)], dim=1)
        L = self.act05(self.drop(self.L_bn(L)))

        Lc = torch.cat([
            self.act05(self.Lc1(L)),
            self.act05(self.Lc2(L)),
            self.act05(self.Lc3(L)),
            self.act05(self.Lc4(L)),
        ], dim=1)
        Lc = self.act05(self.drop(self.Lc_bn(Lc)))

        xm = torch.cat([self.x1(x), self.x2(x), self.x3(x)], dim=1)
        xm = self.act05(self.drop(self.x_bn(xm)))

        xc = torch.cat([
            self.act05(self.xc1(xm)),
            self.act05(self.xc2(xm)),
            self.act05(self.xc3(xm)),
        ], dim=1)
        xc = self.act05(self.drop(self.xc_bn(xc)))

        xL = torch.cat([xc, Lc], dim=1)

        mc = torch.cat([
            self.act05(self.m1(xL)),
            self.act05(self.m2(xL)),
            self.act05(self.m3(xL)),
        ], dim=1)
        mc = self.drop(self.m_bn(mc))

        merge = torch.cat([
            self.act05(self.de1(mc)),
            self.act05(self.de2(mc)),
            self.act05(self.de3(mc)),
        ], dim=1)
        merge = center_crop_or_pad(merge, init.shape[-2], init.shape[-1])

        conv1_1 = self.gcn_11(merge)
        conv1_2 = self.gcn_12(merge)
        conv2_1 = self.gcn_21(conv1_1)
        conv2_2 = self.gcn_22(conv1_2)
        gcn = torch.cat([conv2_1, conv2_2], dim=1)

        y = self.act05(self.conv1(gcn))
        y = self.conv2(y)

        c = self.act05(self.c1(y))
        c = self.c2(c)
        br = self.out_conv(c + y)
        out = self.prelu(br)
        return init + out


class RefineL(nn.Module):
    def __init__(self, input_channels, inputD_channels, dropout=0.0):
        super().__init__()
        self.act03 = nn.LeakyReLU(0.3, inplace=True)
        self.l1 = SameConv2d(inputD_channels, 64, 7)
        self.l2 = SameConv2d(input_channels, 64, 7)

        self.x0 = SameConv2d(128, 64, 7, stride=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SameConv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.3, inplace=True),
                SameConv2d(128, 64, 3),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3, inplace=True),
            )
            for _ in range(4)
        ])

        self.conv = SameConv2d(64, 64, 3)
        self.deconv = SameConvTranspose2d(64, 64, 3, stride=2)
        self.output = SameConv2d(128, input_channels, 3)

    def forward(self, x, inputD):
        L = torch.cat([self.l1(inputD), self.l2(x)], dim=1)
        Lm = self.act03(L)

        y = self.x0(Lm)
        for block in self.blocks:
            y = y + block(y)

        y = self.act03(self.conv(y))
        deconv = self.deconv(y)
        up = F.interpolate(y, scale_factor=2, mode="nearest")
        final = torch.cat([deconv, up], dim=1)
        final = center_crop_or_pad(final, Lm.shape[-2], Lm.shape[-1])
        final = final + Lm
        return self.output(final)


# -----------------------------
# Full model
# -----------------------------

class DTCWTModelCore(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.dtcwt0 = DTCWTLayer(nlevels=2)
        self.dtcwt1 = DTCWTLayer(nlevels=2)
        self.idtcwt = InverseDTCWTLayer()

        self.down_inp_1 = nn.ModuleList([SameConv2d(in_channels, 12, k, stride=2) for k in (2, 3, 5)])
        self.down_x1l = nn.ModuleList([SameConv2d(in_channels, 12, k, stride=2) for k in (2, 3, 5)])
        self.down_inp_2 = nn.ModuleList([SameConv2d(36, 12, k, stride=2) for k in (2, 3, 5)])
        self.down_inp_3 = nn.ModuleList([SameConv2d(36, 12, k, stride=2) for k in (2, 3, 5)])

        self.refine_L2 = RefineL(input_channels=3, inputD_channels=72, dropout=0.3)
        self.refine_H0_2 = RefineH(input_channels=36, inputD_channels=75, dropout=0.2)
        self.refine_H1_2 = RefineH(input_channels=36, inputD_channels=36, dropout=0.2)

        self.refine_L1 = RefineL(input_channels=3, inputD_channels=39, dropout=0.3)
        self.refine_H0_1 = RefineH(input_channels=36, inputD_channels=39, dropout=0.2)
        self.refine_H1_1 = RefineH(input_channels=36, inputD_channels=36, dropout=0.2)

        self.apply(kaiming_init)

    @staticmethod
    def multi_down(x, layers):
        return torch.cat([layer(x) for layer in layers], dim=1)

    def forward(self, x):
        out0_low, out0_h0, out0_h1 = self.dtcwt0(x)
        out1_low, out1_h0, out1_h1 = self.dtcwt1(out0_low)

        downfromInp = self.multi_down(x, self.down_inp_1)
        downfromx1L = self.multi_down(out0_low, self.down_x1l)
        downfromInp2 = self.multi_down(downfromInp, self.down_inp_2)
        downfromInp3 = self.multi_down(downfromInp2, self.down_inp_3)

        downConcat2_L = torch.cat([downfromInp2, downfromx1L], dim=1)
        x2LR = self.refine_L2(out1_low, downConcat2_L)

        downConcat2 = torch.cat([x2LR, downfromInp2, downfromx1L], dim=1)
        x2H0R = self.refine_H0_2(out1_h0, downConcat2)
        x2H1R = self.refine_H1_2(out1_h1, downfromInp3)

        x_idwt2 = self.idtcwt(x2LR, x2H0R, x2H1R)
        x_idwt2 = center_crop_or_pad(x_idwt2, out0_low.shape[-2], out0_low.shape[-1])

        downConcat_L = torch.cat([x_idwt2, downfromInp], dim=1)
        x1LR = self.refine_L1(out0_low, downConcat_L)

        downConcat = torch.cat([x1LR, downfromInp], dim=1)
        x1H0R = self.refine_H0_1(out0_h0, downConcat)
        x1H1R = self.refine_H1_1(out0_h1, downfromInp2)

        x_idwt3 = self.idtcwt(x1LR, x1H0R, x1H1R)
        x_idwt3 = center_crop_or_pad(x_idwt3, x.shape[-2], x.shape[-1])

        refine_list = [x1H0R, x1H1R, x2H0R, x2H1R, x1LR, x2LR]
        return x_idwt3, refine_list


class DTCWTModel(nn.Module):
    def __init__(self, in_channels=3, edge=16):
        super().__init__()
        self.core = DTCWTModelCore(in_channels=in_channels)
        self.slice = SliceLayer(edge=edge)

    def forward(self, x, return_refine=False):
        y, refine_list = self.core(x)
        y = self.slice(y)
        if return_refine:
            return y, refine_list
        return y


def build_DTCWT_model(shape=(3, 256, 256)):
    """
    Compatibility constructor.
    Original Keras shape is usually HWC, e.g. (256, 256, 3).
    PyTorch shape should be CHW, e.g. (3, 256, 256).
    """
    if len(shape) == 3:
        in_channels = shape[0]
    else:
        in_channels = 3
    return DTCWTModel(in_channels=in_channels, edge=16)


if __name__ == "__main__":
    model = build_DTCWT_model((3, 256, 256))
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
