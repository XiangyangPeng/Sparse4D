import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from ..blocks import linear_relu_ln
from .decoder import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ

__all__ = [
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(self, embed_dims: int = 256, vel_dims: int = 3):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, 
                box_3d: torch.Tensor # bs * n_anchors * anchor_dims
                ):
        # anchors 生成 anchor_embedding
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]]) # 中心点位置的embedding
        size_feat = self.size_fc(box_3d[..., [W, L, H]]) # 尺寸的embedding
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]]) # 角度embedding
        output = pos_feat + size_feat + yaw_feat # 拼接
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims]) # 速度embedding
            output = output + vel_feat # 拼接
        output = self.output_fc(output) # 输出
        return output

@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11, # X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        ) # box 回归层
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            ) # 分类层

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor, # bs * n_anchor * embed_dims
        anchor: torch.Tensor, # bs * n_anchor * anchor_dims
        anchor_embed: torch.Tensor, # bs * n_anchor * embed_dims
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        # 检测头
        output = self.layers(instance_feature + anchor_embed) # 回归。bs * n_anchor * output_dim
        # 回归的是offset，所以需要加上anchor
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        # 速度回归的是相对位移的offset，需要转换为绝对速度
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        # 分类
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature) # bs * n_anchor * num_cls
        else:
            cls = None
        return output, cls


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale) # 固定的采样点
        self.num_pts = len(self.fix_scale) + num_learnable_pts # 总的采样点数
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3) # 可学习的采样点

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor, # bs * n_anchor * 11, 锚框
        instance_feature=None, # 实例特征——>可学习的采样点, bs * n_anchor * embed_dims
        T_cur2temp_list=None, # 当前帧到历史帧的变换矩阵, list
        cur_timestamp=None, # 当前帧的时间戳, float
        temp_timestamps=None, # 历史帧的时间戳，list
    ): # 基于锚框（实例特征）生成采样点（当前帧、历史帧）
        # 当前帧的3D采样点
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1]) # bs * n_anchor * 1 * 3
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            ) # bs * n_anchor * n_learnable_pts * 3, (-0.5, 0.5)
            scale = torch.cat([scale, learnable_scale], dim=-2) # bs * n_anchor * (n_learnable_pts + 1) * 3
        key_points = scale * anchor[..., None, [W, L, H]].exp() # bs * n_anchor * (n_learnable_pts + 1) * 3
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3]) # bs * n_anchor * 3 * 3
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(
            # bs * n_anchor * 1 * 3 * 3, bs * n_anchor * (n_learnable_pts + 1) * 3 * 1
            rotation_mat[:, :, None], key_points[..., None] # bs * n_anchor * (n_learnable_pts + 1) * 3 * 1
        ).squeeze(-1) # bs * n_anchor * (n_learnable_pts + 1) * 3
        # 目标内的3D采样点
        key_points = key_points + anchor[..., None, [X, Y, Z]] # bs * n_anchor * (n_learnable_pts + 1) * 3

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ): 
            return key_points

        # 历史帧+运动补偿
        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            # 计算位移
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            # 运动补偿
            temp_key_points = key_points - translation[:, :, None]
            # 坐标系转换
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points) # n_time * [bs * n_anchor * (n_learnable_pts + 1) * 3]
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor, # 锚框
        T_src2dst_list, # 历史帧到目标帧的变换矩阵 list
        src_timestamp=None, # 历史帧的时间戳
        dst_timestamps=None, # 目标帧的时间戳 list
    ): # 将锚框从当前帧转换到目标帧
        dst_anchors = []
        for i in range(len(T_src2dst_list)): # 遍历所有目标帧
            dst_anchor = anchor.clone()
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = dst_anchor[..., [X, Y, Z]]
            if src_timestamp is not None and dst_timestamps is not None:
                translation = vel.transpose(0, -1) * (
                    src_timestamp - dst_timestamps[i]
                ).to(dtype=vel.dtype)
                translation = translation.transpose(0, -1)
                center = center - translation # CV 模型，中心点位移
            dst_anchor[..., [X, Y, Z]] = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3] # 帧间的坐标系转换
            )

            dst_anchor[..., [COS_YAW, SIN_YAW]] = torch.matmul(
                T_src2dst[..., :2, :2], dst_anchor[..., [COS_YAW, SIN_YAW], None]
            ).squeeze(-1) # 方向的坐标系转换，只考虑yaw

            dst_anchor[..., VX:] = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1) # 速度的坐标系转换，2维或者3维

            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
