import math
import logging
from functools import partial
from collections import OrderedDict
from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


#


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class CMT(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=[46, 92, 184, 368],
                 stem_channel=16,
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[3.6, 3.6, 3.6, 3.6],
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 depths=[2, 2, 10, 2],
                 qk_ratio=1,
                 sr_ratios=[8, 4, 2, 1],
                 dp=0.1):
        super().__init__()
        self.num_features = self.embed_dim = embed_dims[-1]
        self.embed_dims=embed_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=7, stride=2, padding=3, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(img_size=img_size // 2, patch_size=2, in_chans=stem_channel,
                                        embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(num_heads[0], self.patch_embed_a.num_patches,
                                                       self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(num_heads[1], self.patch_embed_b.num_patches,
                                                       self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(num_heads[2], self.patch_embed_c.num_patches,
                                                       self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(num_heads[3], self.patch_embed_d.num_patches,
                                                       self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        feats = []
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feats.append(x)

        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feats.append(x)
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feats.append(x)

        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        feats.append(x)
        return feats

    def forward(self, x):
        x = self.forward_features(x)
        return x


@register_model
def cmt_tiny(img_size, in_channels, **kwargs):
    """
    CMT-Tiny
    flops:2.446G
    params:7.662M
    """
    model = CMT(img_size=img_size, in_chans=in_channels, **kwargs)
    return model


@register_model
def cmt_xs(img_size, in_channels, **kwargs):
    """
    CMT-XS: dim x 0.9, depth x 0.8, input 192
    flops:4.087G
    params:13.274M
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[52, 104, 208, 416], stem_channel=16, num_heads=[1, 2, 4, 8],
        depths=[3, 3, 12, 3], mlp_ratios=[3.77, 3.77, 3.77, 3.77], qk_ratio=1, sr_ratios=[8, 4, 2, 1], **kwargs)
    model = CMT(img_size=img_size, in_chans=in_channels, **model_kwargs)
    return model


@register_model
def cmt_small(img_size, in_channels, **kwargs):
    """
    CMT-Small
    flops:7.819G
    params:24.028M
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[1, 2, 4, 8],
        depths=[3, 3, 16, 3], mlp_ratios=[4, 4, 4, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1], **kwargs)
    model = CMT(img_size=img_size, in_chans=in_channels, **model_kwargs)
    return model


@register_model
def cmt_base(img_size, in_channels, **kwargs):
    """
    CMT-Base
    flops:13.746G
    params:43.160M
    """
    model_kwargs = dict(
        qkv_bias=True, embed_dims=[76, 152, 304, 608], stem_channel=38, num_heads=[1, 2, 4, 8],
        depths=[4, 4, 20, 4], mlp_ratios=[4, 4, 4, 4], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.3, **kwargs)
    model = CMT(img_size=img_size, in_chans=in_channels, **model_kwargs)
    return model


class FFN(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, image_size=[224, 224], embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.image_size = image_size
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i + 1}", FFN(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i + 2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        seg = F.interpolate(seg, size=self.image_size, mode='bilinear', align_corners=False)  # to original image shape
        return seg


class projection_conv(nn.Module):
    """
    A non-linear neck in DenseCL
    The non-linear neck, fc-relu-fc, conv-relu-conv
    """

    def __init__(self, in_dim, hid_dim=2048, out_dim=128, s=4):
        super(projection_conv, self).__init__()
        self.is_s = s
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hid_dim, out_dim))
        self.mlp_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dim, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hid_dim, out_dim, 1))
        if self.is_s:
            self.pool = nn.AdaptiveAvgPool2d((s, s))
        else:
            self.pool = None

    def forward(self, x):
        # Global feature vector
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.mlp(x1)

        # dense feature map
        if self.is_s:
            x = self.pool(x)                        # [N, C, S, S]
        x2 = self.mlp_conv(x)
        x2 = x2.view(x2.size(0), x2.size(1), -1)    # [N, C, SxS]

        return x1, x2


class CMT_S(nn.Module):
    def __init__(self, image_size=[224, 224], in_channels=3, num_classes=4):
        super().__init__()
        self.encoder = cmt_xs(img_size=image_size[0], in_channels=in_channels)
        # self.encoder = cmt_xs(img_size=image_size[0], in_channels=in_channels)
        self.decoder = SegFormerHead(self.encoder.embed_dims, image_size=image_size, embed_dim=256, num_classes=num_classes)

    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class CMT_Plus(nn.Module):
    def __init__(self, image_size=[224, 224], in_channels=3, num_classes=4):
        super().__init__()
        self.encoder = cmt_tiny(img_size=image_size[0], in_channels=in_channels)
        self.decoder = SegFormerHead(self.encoder.embed_dims, image_size=image_size, embed_dim=256, num_classes=num_classes)
        self.dense_projection_high = projection_conv(self.encoder.embed_dims[-1])
        self.dense_projection_head = projection_conv(num_classes, hid_dim=1024)

    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        high_feature = self.dense_projection_high(feature[-1])
        head_feature = self.dense_projection_head(output)
        return output, high_feature, head_feature


if __name__ == '__main__':
    # x = torch.randn(2, 3, 224, 224)
    # model = cmt_base(img_size=224, in_channels=3)
    # feats = model(x)
    # for i in feats:
    #     print(i.shape)

    # from thop import profile
    #
    # model=CMT_S(image_size=[224,224],num_classes=4,in_channels=3)
    # input = torch.randn(2, 3, 224, 224)
    # flops, params = profile(model, inputs=(input,))
    # print("flops:{:.3f}G".format(flops / 1e9))
    # print("params:{:.3f}M".format(params / 1e6))

    # x = torch.randn(2, 3, 224, 224)
    # model=CMT_S(image_size=[224,224],num_classes=4,in_channels=3)
    # y=model(x)
    # print(y.shape)
    x = torch.randn(2, 3, 224, 224)
    model=CMT_Plus(image_size=[224,224],num_classes=4,in_channels=3)
    output, high_feature, head_feature=model(x)
    print(output.shape)
    print(high_feature[0].shape,high_feature[1].shape)
    print(head_feature[0].shape,head_feature[0].shape)
