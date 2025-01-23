# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        depth_module: dict = None,
        dense_depth_module: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "category_ids",
        gt_reg_key: str = "boxes",
        reg_weights=None,
        operation_order: Optional[List[str]] = None,
        kps_generator=None,
        max_queue_length=0,
        cls_threshold_to_reg=-1,
        init_cfg=None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.max_queue_length = max_queue_length
        self.cls_threshold_to_reg = cls_threshold_to_reg
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None: # 默认v1
            operation_order = [
                "gnn", # self attention
                "norm",
                "deformable", # 特征采样、聚合
                "norm",
                "ffn", # 全连接层
                "norm", # norm在两个功能层中间
                "refine", # 回归层
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.depth_module = build(depth_module, PLUGIN_LAYERS)
        self.dense_depth_module = build(dense_depth_module, PLUGIN_LAYERS)
        self.kps_generator = build(kps_generator, PLUGIN_LAYERS)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        # 得到anchor和instance_feature（当前帧+历史帧）
        (
            instance_feature, # bs * n_anchors * embed_dims。当前帧
            anchor, # bs * n_anchors * anchor_dims
            temp_instance_feature, # bs * n_temp_instances * embed_dims。上一帧
            temp_anchor, # bs * n_temp_instances * anchor_dims。上一帧的anchor转换到当前帧
            time_interval, # bs * 1
        ) = self.instance_bank.get(batch_size, metas)
        anchor_embed = self.anchor_encoder(anchor) # anchor 生成 anchor embedding。 bs * n_anchors * embed_dims
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # 从insatnce_bank提取历史帧的特征图和元信息
        _feature_queue = self.instance_bank.feature_queue
        _meta_queue = self.instance_bank.meta_queue
        if feature_queue is not None and _feature_queue is not None:
            feature_queue = feature_queue + _feature_queue # list 的拼接
            meta_queue = meta_queue + _meta_queue
        elif feature_queue is None:
            feature_queue = _feature_queue
            meta_queue = _meta_queue

        prediction = []
        classification = []
        # 特征聚合、检测头（v1 v2 有区别）
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn": # CA
                instance_feature = self.layers[i](
                    instance_feature, # 当前帧的实例特征，堆叠的模块中不断更新。Query. bs * n_anchors * embed_dims. (batch_first=True)
                    temp_instance_feature, # 上一帧的实例特征，不会更新。Key. bs * n_temp_instances * embed_dims
                    temp_instance_feature, # Value
                    query_pos=anchor_embed, # 当前帧的anchor embedding，不断更新
                    key_pos=temp_anchor_embed, # 上一帧的anchor embedding，不断更新
                ) # bs * n_anchors * embed_dims
            elif op == "gnn": # SA
                instance_feature = self.layers[i](
                    instance_feature, # Q K V. bs * n_anchors * embed_dims
                    query_pos=anchor_embed,
                ) # bs * n_anchors * embed_dims
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable": # 每一次迭代，都会基于refine后的anchor-keypoints进行特征采样、融合
                # multi-view multi-level multi-timestamp multi-keypoints 特征采样与融合
                instance_feature = self.layers[i](
                    # 当前帧信息
                    instance_feature, # 实例特征。bs * n_anchors * embed_dims
                    anchor, # 锚点。bs * n_anchors * anchor_dims
                    anchor_embed, # 锚点embedding。bs * n_anchors * embed_dims
                    feature_maps, # 特征图。n_levels * [bs * n_cams * n_channels * W * H]
                    metas,
                    # 历史帧信息
                    feature_queue=feature_queue, # 历史帧的特征图
                    meta_queue=meta_queue, # 历史帧的元信息
                    # 模块
                    depth_module=self.depth_module,
                    anchor_encoder=self.anchor_encoder,
                )
            elif op == "refine": # 一次迭代的最后模块，对anchor/anchor embedding进行微调，并作为下一次迭代的输入
                # box 属性回归，分类
                anchor, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                # 组合当前帧与历史帧的实例
                if len(prediction) == self.num_single_frame_decoder: # v2，新的实例，上一帧传递的实例，需要topk筛选
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                # 更新 anchor embedding
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                # 更新 上一帧的 anchor embedding
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        # 更新instance_bank，保存当前帧(topk)
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        return classification, prediction # 多次迭代的分类、回归结果均返回

    @force_fp32(apply_to=("cls_scores", "reg_preds")) # 混合精度训练
    def loss(self, 
             cls_scores, # bs * n_anchors * n_classes
             reg_preds,  # bs * n_anchors * n_state
             data, # dict, 包含gt_cls_key, gt_reg_key
             feature_maps=None
             ): # 损失函数，分类focal loss， 回归smooth l1 loss
        output = {}
        # 对每一个迭代块计算分类、回归损失
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            reg = reg[..., : len(self.reg_weights)] # 只监督部分属性（VZ不监督）
            # 对每一个pred生成监督信号（匹配，set-loss）
            # bs * n_anchors, bs * n_anchors * n_states
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls, # bs * n_anchors * n_classes
                reg, # bs * n_anchors * n_state_
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)] # 截取回归监督信号部分属性（VZ不监督）
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1)) # 找到match非空的anchor。bs * n_anchors * 1
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            ) # 计算match上的anchor数。涉及到分布式计算
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )
            # 分类损失 - focal loss
            cls = cls.flatten(end_dim=1) # (bs*n_anchors) * n_classes
            cls_target = cls_target.flatten(end_dim=1) # (bs*n_anchors)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos) # focal loss 支持 非 one-hot 的target

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights) # 回归权重
            reg_target = reg_target.flatten(end_dim=1)[mask] # (bs*n_anchors) * n_states 的部分。真值
            reg = reg.flatten(end_dim=1)[mask] # (bs*n_anchors) * n_states 的部分。预测值
            reg_weights = reg_weights.flatten(end_dim=1)[mask] # (bs*n_anchors) * n_states 的部分
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            ) # 将nan替换为0
            # 回归损失 - l1 loss
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos # 以anchor数为归一化因子
            ) #

            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )

        # 深度损失
        if (
            self.depth_module is not None # v1，对深度进行监督
            and self.kps_generator is not None
            and feature_maps is not None
        ):
            reg_target = self.sampler.encode_reg_target(
                data[self.gt_reg_key], reg_preds[0].device
            ) # bs * [n_gt * n_states]
            loss_depth = []
            for i in range(len(reg_target)):
                if len(reg_target[i]) == 0:
                    continue
                # 特征点采样
                key_points = self.kps_generator(reg_target[i][None]) # bs * n_gt * n_keypoints * 3
                features = (
                    DFG.feature_sampling(
                        [f[i : i + 1] for f in feature_maps],
                        key_points,
                        data["projection_mat"][i : i + 1],
                        data["image_wh"][i : i + 1],
                    ) # bs * n_gt * n_cam * n_level * n_keypoints * n_channels
                    .mean(2) # bs * n_gt * n_level * n_keypoints * n_channels
                    .mean(2) # bs * n_gt * n_keypoints * n_channels
                )
                # 基于真值anchor的采样特征进行深度加权。其实是一条单独的监督链路了
                depth_confidence = self.depth_module(
                    features, reg_target[i][None, :, None], output_conf=True
                ) # 返回的是真值的深度在估计的深度分布中的置信度，本身就是损失
                loss_depth.append(-torch.log(depth_confidence).sum())
            output["loss_depth"] = (
                sum(loss_depth) / num_pos / self.kps_generator.num_pts
            )

        if self.dense_depth_module is not None:
            output["loss_dense_depth"] = self.dense_depth_module(
                feature_maps,
                focal=data.get("focal"),
                gt_depths=data["gt_depth"],
            )
        return output

    @force_fp32(apply_to=("cls_scores", "reg_preds"))
    def post_process(self, cls_scores, reg_preds):
        return self.decoder.decode(cls_scores, reg_preds)
