import numpy as np
import torch
from torch import nn
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        max_queue_length=-1,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.max_queue_length = max_queue_length
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler # 
        if isinstance(anchor, str):
            anchor = np.load(anchor) # 从文件加载numpy数组
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor) # 从列表或元组转换为numpy数组
        self.num_anchor = min(len(anchor), num_anchor) # 限定anchor数量
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32), # 转换为tensor
            requires_grad=anchor_grad, # anchor是否参与训练，默认为True
        ) # n_anchors * anchor_dims
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        ) # n_anchors * embed_dims
        self.cached_feature = None # 历史帧的实例特征
        self.cached_anchor = None # 历史帧的anchor
        self.metas = None # 历史帧的参数，包括相机参数、时间戳等
        self.mask = None
        self.confidence = None
        self.feature_queue = [] if max_queue_length > 0 else None
        self.meta_queue = [] if max_queue_length > 0 else None

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def get(self, 
            batch_size, 
            metas=None # 相机参数, bs * 1
            ): # 获取历史帧的anchor（投影到当前帧）和instance_feature
        # batch 中的每个 sample 对应一个 anchor + instance_feature
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        ) # batch_size * n_anchors * embed_dims
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
          # batch_size * n_anchors * anchor_dims

        # 将历史帧的anchor投影到当前帧
        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ): # 若有缓存（历史帧）
            if self.anchor_handler is not None:
                # 坐标系转换
                T_temp2cur = self.cached_anchor.new_tensor(
                    [
                        x["T_global_inv"] # 当前帧的一个 sample 的坐标系转换矩阵
                        @ self.metas["img_metas"][i]["T_global"] # 历史帧的一个 sample 的坐标系转换矩阵
                        for i, x in enumerate(metas["img_metas"])
                    ]
                ) # batch_size * 4 * 4
                # anchor 投影到目标帧
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor, # 历史帧的anchor
                    [T_temp2cur], # [batch_size * 4 * 4]，只有一帧历史帧(src2dst)
                    self.metas["timestamp"], # 历史帧的时间戳(src)
                    [metas["timestamp"]], # [bs * 1]，当前帧(dst)
                )[0]
            self.mask = (
                torch.abs(metas["timestamp"] - self.metas["timestamp"])
                <= self.max_time_interval
            ) # mask，当前帧与历史帧的时间间隔小于阈值
        else: # 若没有缓存（历史帧）
            self.cached_feature = None
            self.cached_anchor = None
            self.confidence = None

        if (
            self.metas is None
            or batch_size != self.metas["timestamp"].shape[0]
        ) and (
            self.meta_queue is None
            or len(self.meta_queue) == 0
            or batch_size != self.meta_queue[0]["timestamp"].shape
        ):
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            ) # batch_size * 1
        else:
            if self.metas is not None:
                history_time = self.metas["timestamp"]
            else:
                history_time = self.meta_queue[0]["timestamp"]
            time_interval = metas["timestamp"] - history_time # bs * 1
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            time_interval = torch.where(
                torch.logical_or(
                    time_interval == 0, torch.abs(time_interval) > self.max_time_interval
                ),
                time_interval.new_tensor(self.default_time_interval),
                time_interval,
            ) # 异常时间间隔的处理
        return (
            instance_feature, # 初始化的instance_feature，随机初始化。bs * n_anchors * embed_dims
            anchor, # 初始化的anchor，kmeans初始化。bs * n_anchors * anchor_dims
            self.cached_feature, # 历史帧的instance_feature。bs * n_anchors * embed_dims
            self.cached_anchor,  # 历史帧的anchor。bs * n_anchors * anchor_dims
            time_interval, # 时间间隔。bs * 1
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        # 新生的实例中挑选topk部分
        N = self.num_anchor - self.num_temp_instances # 保留的anchor数量
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        # 与上一帧传递来的实例组合
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1 # 上一帧传递来的实例特征 + 当前帧新的实例特征
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1 # 上一帧传递来的实例anchor + 当前帧新的实例anchor
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor) # 兜底
        return instance_feature, anchor

    def cache(
        self,
        instance_feature, # bs * n_anchors * embed_dims
        anchor, # bs * n_anchors * anchor_dims
        confidence, # bs * n_anchors * 1
        metas=None,
        feature_maps=None,
    ): # 缓存anchor和instance_feature（topk），传递到下一帧的
        if self.feature_queue is not None and not self.training:
            # 如果队列已满，则删除最后一个元素
            while len(self.feature_queue) > self.max_queue_length - 1:
                self.feature_queue.pop()
                self.meta_queue.pop()
            # 将新元素（特征图、相机参数/时间戳）插入队列的开头
            self.feature_queue.insert(0, feature_maps)
            self.meta_queue.insert(0, metas)

        # v2 引入的新特征，temporal instances 时序融合
        if self.num_temp_instances > 0:
            instance_feature = instance_feature.detach()
            anchor = anchor.detach()
            confidence = confidence.detach()

            self.metas = metas # 缓存相机参数/时间戳信息
            confidence = confidence.max(dim=-1).values.sigmoid() 
            # 置信度更新
            if self.confidence is not None:
                # 默认temporal instances 保存在前半部分
                confidence[:, : self.num_temp_instances] = torch.maximum(
                    self.confidence * self.confidence_decay,
                    confidence[:, : self.num_temp_instances],
                ) # 衰减置信度（历史置信度*衰减系数，当前置信度）

            # topk 选择temporal instances
            (
                self.confidence, # bs * num_temp_instances * 1
                (self.cached_feature, # bs * num_temp_instances * embed_dims
                 self.cached_anchor), # bs * num_temp_instances * anchor_dims
            ) = topk(
                confidence, self.num_temp_instances, instance_feature, anchor
            )
