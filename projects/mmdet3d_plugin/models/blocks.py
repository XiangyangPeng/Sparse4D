# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import DeformableAggregationFunction as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "LinearFusionModule",
    "DepthReweightModule",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        # 当前帧信息
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        # 历史帧信息
        feature_queue=None, # 历史帧的特征图
        meta_queue=None, # 历史帧的元信息
        # 模块
        depth_module=None, # 深度 reweight 模块
        anchor_encoder=None, # anchor 编码器
        **kwargs: dict,
    ): # multi-view multi-scale multi-timestamp multi-keypoint 融合
        bs, num_anchor = instance_feature.shape[:2]
        if feature_queue is not None and len(feature_queue) > 0:
            T_cur2temp_list = [] # n_time * [bs * 4 * 4]
            for meta in meta_queue:
                T_cur2temp_list.append(
                    instance_feature.new_tensor(
                        [
                            x["T_global_inv"]
                            @ metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(meta["img_metas"])
                        ]
                    )
                )
            # 生成采样点（当前帧、历史帧）
            # bs * n_anchor * (n_learnable_pts + 1) * 3
            # n_time * [bs * n_anchor * (n_learnable_pts + 1) * 3]
            key_points, temp_key_points_list = self.kps_generator(
                anchor,
                instance_feature,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue], # n_time * [bs * 1]
            )
            # 生成历史帧的anchor
            # n_time * [bs * n_anchor * anchor_dims]
            temp_anchors = self.kps_generator.anchor_projection(
                anchor,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )
            # n_time * [bs * n_anchor * anchor_embed_dims]
            temp_anchor_embeds = [
                anchor_encoder(x)
                if self.use_temporal_anchor_embed
                and anchor_encoder is not None
                else None
                for x in temp_anchors
            ]
            # (n_time + 1) * 1
            time_intervals = [
                (metas["timestamp"] - x["timestamp"]).to(
                    dtype=instance_feature.dtype
                )
                for x in [metas] + meta_queue
            ]
        else:
            key_points = self.kps_generator(anchor, instance_feature)
            temp_key_points_list = (
                feature_queue
            ) = meta_queue = temp_anchor_embeds = temp_anchors = []
            time_intervals = [instance_feature.new_tensor([0])]

        if self.temp_module is not None and len(feature_queue) == 0:
            # 
            features = instance_feature.new_zeros(
                [bs, num_anchor, self.num_pts, self.embed_dims]
            ) # bs * num_anchor * num_pts * embed_dims
        else:
            features = None

        # 权重
        if not self.use_temporal_anchor_embed or anchor_encoder is None:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        # 遍历每个历史帧+当前帧
        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
            temp_anchor,
            time_interval,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
            temp_anchors[::-1] + [anchor],
            time_intervals[::-1],
        ):
            if self.use_temporal_anchor_embed and anchor_encoder is not None:
                weights = self._get_weights(
                    instance_feature, temp_anchor_embed, metas
                )
            if self.use_deformable_func: # v2
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                ) 
                points_2d = (
                    self.project_points(
                        temp_key_points,
                        temp_metas["projection_mat"],
                        temp_metas.get("image_wh"),
                    )
                    .permute(0, 2, 3, 1, 4)
                    .reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                )
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            else: # v1
                # 特征采样
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                ) # bs * n_anchor * n_cam * n_level * n_keypoints * embed_dims
                # multi-view multi-level 加权融合
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                ) # bs * n_anchor * n_keypoints * embed_dims
            # 深度 reweight
            # anchor 的深度 d，特征的深度分布预测 f，权重 w = f(x=d)
            if depth_module is not None:
                temp_features_next = depth_module(
                    temp_features_next, # bs * n_anchor * n_keypoints * embed_dims
                    temp_anchor[:, :, None] # bs * n_anchor * 1 * anchor_dims
                ) # bs * n_anchors * n_keypoints * embed_dims

            # 时序融合
            if features is None: # 初始化
                features = temp_features_next
            elif self.temp_module is not None:
                # multi-timestamp 融合
                features = self.temp_module(
                    features, temp_features_next, time_interval
                )
            else:
                # 直接相加以融合
                features = features + temp_features_next

        # multi-keypoint 融合。已经加权过了, n_keypoints n_groups 的权重收回，保证权重和为1
        features = features.sum(dim=2)  # fuse multi-point features。bs * n_anchors * embed_dims
        # 输出层，输出最终采样的特征
        output = self.proj_drop(self.output_proj(features))
        # 实例特征融合
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, 
                     instance_feature, # bs * n_anchor * embed_dims
                     anchor_embed, # bs * n_anchor * embed_dims
                     metas=None): # 加权是对 multi-level multi-view multi-keypoints 以及 embedding 多头(group) 的加权
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed # bs * n_anchor * embed_dims
        if self.camera_encoder is not None:
            # 将相机参数编码
            camera_embed = self.camera_encoder(
                # bs * n_cam * 4 * 4
                metas["projection_mat"][:, :, :3].reshape( # bs * n_cam * 3 * 4，去除齐次坐标转换矩阵的最后一行
                    bs, self.num_cams, -1
                ) # bs * n_cam * 12
            ) # bs * n_cam * embed_dims
            # bs * n_anchor * 1 * embed_dims + bs * 1 * n_cam * embed_dims
            feature = feature[:, :, None] + camera_embed[:, None] # bs * n_anchor * n_cam * embed_dims
            # 不同的cam的anchor embedding 不同了
        weights = (
            # bs * n_anchor * (n_groups * n_cams * n_levels * n_keypoints) 或者 
            # bs * n_anchor * n_cams * (n_groups * n_levels * n_keypoints)
            self.weights_fc(feature) 
            .reshape(bs, num_anchor, -1, self.num_groups) # bs * n_anchor * (n_cams * n_levels * n_keypoints) * n_groups
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            ) # bs * n_anchor * n_cams * n_level * n_keypoints * n_groups
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, 
                       projection_mat, 
                       image_wh=None
                       ):
        bs, num_anchor, num_pts = key_points.shape[:3]

        # 齐次坐标
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        ) # bs * n_anchor * n_keypoints * 4
        # 投影
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], # bs * n_cam * 1 * 1 * 4 * 4
            pts_extend[:, None, ..., None] # bs * 1 * n_anchor * n_keypoints * 4 * 1
        ).squeeze(-1) # bs * n_cam * n_anchor * n_keypoints * 4
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        ) # 视觉投影，以z轴归一化。bs * n_cam * n_anchor * n_keypoints * 2
        if image_wh is not None:  # bs * n_cam * 2
            # bs * n_cam * n_anchor * n_keypoints * 2 / bs * n_cam * 1 * 1 * 2
            points_2d = points_2d / image_wh[:, :, None, None] # 归一化到[0, 1]
            # bs * n_cam * n_anchor * n_keypoints * 2
            
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor], # n_levels * [bs * n_cams * n_channels * W * H]
        key_points: torch.Tensor, # bs * n_anchor * n_pkeypoints * 3
        projection_mat: torch.Tensor, # 投影矩阵，
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: # 在多视角多level的特征图上基于anchor的keypoints进行特征采样
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # 关键点投影
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        ) # bs * n_cam * n_anchor * n_keypoints * 2
        points_2d = points_2d * 2 - 1 # 映射到[-1, 1]
        points_2d = points_2d.flatten(end_dim=1) # (bs * n_cam) * n_anchor * n_keypoints * 2

        # 特征图采样
        features = []  
        for fm in feature_maps: # n_levels
            features.append(
                # 特征图采样，（差值）
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), # (bs*n_cam) * n_channels * w * h
                    points_2d # (bs * n_cam) * n_anchor * n_keypoints * 2
                ) # (bs*n_cam) * n_channels * n_anchor * n_keypoints
            ) # n_levels * [(bs*n_cam) * n_channels * n_anchor * n_keypoints]

        # 输出格式整理
        features = torch.stack(features, dim=1) # (bs*n_cam) * n_levels * n_channels * n_anchor * n_keypoints
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute( # bs * n_cam * n_levels * n_channels * n_anchor * n_keypoints
            0, 4, 1, 2, 5, 3
        )  # bs * n_anchor * n_cam * n_level * n_keypoints * n_channels
        # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor, # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims
        weights: torch.Tensor, # bs * n_anchor * n_cam * n_level * n_keypoints * n_groups
    ):
        bs, num_anchor = weights.shape[:2]
        features = (weights[..., None] * # bs * n_anchor * n_cam * n_level * n_keypoints * n_groups * 1
                    features.reshape(
                        features.shape[:-1] + (self.num_groups, self.group_dims)) 
                        # bs * n_anchor * n_cam * n_level * n_keypoints * n_groups * group_dims
        ) # bs * n_anchor * n_cam * n_level * n_keypoints * n_groups * group_dims
        # multi-view multi-level 相加以融合
        features = features.sum(dim=2).sum(dim=2) # bs * n_anchor * n_keypoints * n_groups * group_dims
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        ) # bs * n_anchor * n_keypoints * embed_dimss
        return features


@PLUGIN_LAYERS.register_module()
class LinearFusionModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        alpha=0.9,
        beta=10,
        enable_temporal_weight=True,
    ):
        super(LinearFusionModule, self).__init__()
        self.embed_dims = embed_dims
        self.alpha = alpha
        self.beta = beta
        self.enable_temporal_weight = enable_temporal_weight
        self.fusion_layer = nn.Linear(self.embed_dims * 2, self.embed_dims)

    def init_weight(self):
        xavier_init(self.fusion_layer, distribution="uniform", bias=0.0)

    def forward(self, feature_1, feature_2, time_interval=None):
        if self.enable_temporal_weight:
            temp_weight = self.alpha ** torch.abs(time_interval * self.beta)
            feature_2 = torch.transpose(
                feature_2.transpose(0, -1) * temp_weight, 0, -1
            )
        return self.fusion_layer(torch.cat([feature_1, feature_2], dim=-1)) # bs * n_anchor * n_keypoints * embed_dims


@PLUGIN_LAYERS.register_module()
class DepthReweightModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        min_depth=1,
        max_depth=56,
        depth_interval=5,
        ffn_layers=2,
    ):
        super(DepthReweightModule, self).__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        self.depths = np.arange(min_depth, max_depth + 1e-5, depth_interval) # 预定义的深度网格
        self.max_depth = max(self.depths)

        layers = []
        for i in range(ffn_layers):
            layers.append(
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    dropout=0.0,
                    add_residual=True,
                )
            )
        layers.append(nn.Linear(embed_dims, len(self.depths))) # 分类网络
        self.depth_fc = Sequential(*layers)

    def forward(self, 
                features, # bs * n_anchor * n_keypoints * embed_dims
                points_3d, # bs * n_anchor * 1 * anchor_dims
                output_conf=False): # 基于深度计算权重，对特征加权
        # 计算anchor的深度（距离）
        reference_depths = torch.norm(
            points_3d[..., :2], dim=-1, p=2, keepdim=True
        ) # bs * n_anchor * 1 * 1
        reference_depths = torch.clip(
            reference_depths,
            max=self.max_depth - 1e-5,
            min=self.min_depth + 1e-5,
        ) # bs * n_anchor * 1
        weights = (
            1
            # bs * n_anchor * 1 * 1, n_depths
            - torch.abs(reference_depths - points_3d.new_tensor(self.depths))
            / self.depth_interval
        ) # bs * n_anchors * 1 * n_depths

        top2 = weights.topk(2, dim=-1)[0]
        weights = torch.where(
            weights >= top2[..., 1:2], weights, weights.new_tensor(0.0)
        )
        scale = torch.pow(top2[..., 0:1], 2) + torch.pow(top2[..., 1:2], 2) # 归一化系数
        confidence = self.depth_fc(features).softmax(dim=-1) # bs * n_anchor * n_keypoints * n_depths
        # bs * n_anchors * n_keypoints * n_depths
        # bs * n_anchors * n_keypoints * 1
        confidence = torch.sum(weights * confidence, dim=-1, keepdim=True)
        confidence = confidence / scale # bs * n_anchors * n_keypoints * 1

        if output_conf:
            return confidence
        return features * confidence # bs * n_anchors * n_keypoints * embed_dims


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = (depth.T * focal / self.equal_focal).T
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
