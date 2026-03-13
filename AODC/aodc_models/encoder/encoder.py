import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mmcv.ops import DeformConv2d as dfconv
from .conv4d import CenterPivotConv4d as Conv4d
from ..midlayer import ROIAlign
from .positional_encoding import PositionalEncodingsFixed
from .transformer import TransformerEncoder
from .cross_attn import CrossAttentionBlock
from clip import clip


class SizeBlock(nn.Module):
    def __init__(self, inc):
        super(SizeBlock, self).__init__()
        self.conv = dfconv(in_channels=inc, out_channels=inc, kernel_size=3, stride=1, padding=1)
        self.adapt = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        self.local = nn.Sequential(
            nn.Conv2d(inc, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 * 2, 3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x, scale):
        l_offset = self.local(x)
        g_offset = self.adapt(scale)
        offset = self.fuse(torch.cat((g_offset, l_offset), dim=1))
        fea = self.conv(x, offset)
        return self.relu(fea)


class encoder(nn.Module):
    def __init__(self, zero_shot, few_shot):
        super(encoder, self).__init__()
        self.zero_shot = zero_shot
        self.few_shot = few_shot
        self.reduction = 8
        self.backbone = models.resnet50(pretrained=True)

        self.in_proj = nn.Conv2d(1536, 512, 3, 1, padding=1)
        if few_shot:
            self.k = 3
            self.roi = 32
            self.roialign = ROIAlign(self.roi, 1. / 8)
        else:
            self.k = 1
        if zero_shot:
            self.clip, _ = clip.load("ViT-B/16")
            self.text_embed = nn.Linear(512, 512)
            self.text_cross = nn.ModuleList([CrossAttentionBlock(512, 8, 4.0, True, False) for _ in range(2)])
        self.mod = SizeBlock(512)
        self.conv = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512))

        self.pos_emb = PositionalEncodingsFixed(512)
        self.transformer = TransformerEncoder(3, 512, 8, 0.1, 1e-5, 8, False, nn.GELU, True)

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        self.scale_embed = make_building_block(self.k, [8, 16, 32, 64], [7, 5, 3, 3], [4, 2, 2, 1])

    def forward(self, x, boxes, categories):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        with torch.no_grad():
            if self.zero_shot:
                texts = clip.tokenize(categories).cuda()
                texts = self.clip.encode_text(texts).to(torch.float)

            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = layer2 = self.backbone.layer2(x)
            layer3 = self.backbone.layer3(x)

            x = torch.cat([
                F.interpolate(f, size=size, mode='bilinear', align_corners=True)
                for f in [layer2, layer3]
            ], dim=1)
        x = self.in_proj(x)
        bsz, ch, h, w = x.size()
        eps = 1e-5
        if self.zero_shot:
            x = x.flatten(2).transpose(1, 2)
            texts = self.text_embed(texts).unsqueeze(-2)
            cross_pos = self.pos_emb(bsz, h, w, x.device).flatten(2).permute(0, 2, 1)
            x = x + cross_pos
            for blk in self.text_cross:
                x = blk(x, texts)
            x1 = x = x.transpose(1, 2)
            x = x.view(bsz, ch, h, w)
        else:
            x1 = x.flatten(2)

        x1 = x1 / (x1.norm(dim=1, p=2, keepdim=True) + eps)
        if self.few_shot:
            anchors_patches = self.roialign(x, boxes)
            anchors_patches = anchors_patches.view(bsz, 3, -1, self.roi, self.roi).transpose(1, 2).flatten(2)
            anchors_patches = anchors_patches / (anchors_patches.norm(dim=1, p=2, keepdim=True) + eps)
            sim = torch.bmm(x1.transpose(1, 2), anchors_patches).view(bsz, h, w, 3, self.roi, self.roi).permute(0, 3, 1, 2, 4, 5)
        else:
            sim = torch.bmm(x1.transpose(1, 2), x1).view(bsz, h, w, h, w).unsqueeze(1)
        sim = sim.clamp(min=0)

        scale_fea = self.scale_embed(sim).view(bsz, 64, h, w, -1).mean(dim=-1)
        x = self.mod(x, scale_fea)
        x = self.conv(x)

        pos_emb = self.pos_emb(bsz, h, w, x.device).flatten(2).permute(2, 0, 1)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x, pos_emb, src_key_padding_mask=None, src_mask=None)
        x = x.permute(1, 2, 0).reshape(bsz, 512, h, w)

        return x
