# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
import cv2
# from layers import GCN
# from layers import KnnGraph
# from RoIlayer import RROIAlign
# from layers import Graph_RPN

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        # self.Attention=ChannelAttention()#消融实验需注释掉注意力模块
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        # self.gcn_model = GCN(600, 32)  # 600 = 480 + 120
        # self.pooling = RROIAlign((3, 4), 1.0 / 1)  # (32+8)*3*4 =480对应drrg3.4节（1 × 3 × 4 × Cr feature block is obtained）


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")


        images = to_image_list(images)

        features = self.backbone(images.tensors)#通过backbone输出的特征图
        # print(features.shape)#根据打印出来的特征图的通道数定义in_planes=？


        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            #self.warm_start -= 1
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)\

            return losses
        else:
            return result

# class ChannelAttention(nn.Module):#改
#     def __init__(self, in_planes=, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class UseAttentionModel(nn.Module):
#     def __init__(self, H):
#             super(UseAttentionModel, self).__init__()
#             self.channel_attention = ChannelAttention(H)
#     def forward(self, x):  # 反向传播
#         attention_value = self.channel_attention(x)
#         out = x.mul(attention_value)  # 得到借助注意力机制后的输出
#         return out

# def get_total_train_data(H, W, C, class_count):
#     """得到全部的训练数据，这里需要替换成自己的数据"""
#     import numpy as np
#     x_train = torch.Tensor(
#         np.random.random((1000, H, W, C)))  # 维度是 [ 数据量, 高H, 宽W, 长C]
#     y_train = torch.Tensor(
#         np.random.randint(0, class_count, size=(1000, 1))).long()  # [ 数据量, 句子的分类], 这里的class_count=2，就是二分类任务
#     return x_train, y_train


# if __name__ == '__main__':
#     # ================训练参数=================
#     epochs = 100
#     batch_size = 30
#     output_class = 14
#     H = 40
#     W = 50
#     C = 30
#     # ================准备数据=================
#     x_train, y_train = get_total_train_data(H, W, C, class_count=output_class)
#     train_loader = Data.DataLoader(
#         dataset=Data.TensorDataset(x_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
#         batch_size=batch_size,  # 每块的大小
#         shuffle=True,  # 要不要打乱数据 (打乱比较好)
#         num_workers=6,  # 多进程（multiprocess）来读数据
#         drop_last=True,
#     )
#     # 初始化模型
#     model = ChannelAttention(in_planes=H)
#     # 开始训练
#     for i in range(epochs):
#         for seq, labels in train_loader:
#             attention_out = model(seq)
#             seq_attention_out = attention_out.squeeze()
#             for i in range(seq_attention_out.size()[0]):
#                 print(seq_attention_out[i])