import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet.core.bbox.builder import BBOX_SAMPLERS

from .decoder import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ, YAW

__all__ = ["SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(object):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
    ):
        super(SparseBox3DTarget, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights

    def encode_reg_target(self, 
                          box_target, # bs * n_gt * n_state
                          device=None):
        outputs = []
        for box in box_target:
            if isinstance(box, BaseInstance3DBoxes):
                center = box.gravity_center
                box = box.tensor.clone()
                box[..., [X, Y, Z]] = center

            # box 属性对齐预测格式
            output = torch.cat(
                [
                    box[..., [X, Y, Z]], # n_gt * 3
                    box[..., [W, L, H]].log(), # n_gt * 3
                    torch.sin(box[..., YAW]).unsqueeze(-1), # n_gt * 1
                    torch.cos(box[..., YAW]).unsqueeze(-1), # n_gt * 1
                    box[..., YAW + 1 :], # n_gt * 3
                ],
                dim=-1,
            ) # n_gt * 11
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs # bs * [n_gt * 11]

    def sample(
        self,
        cls_pred, # bs * n_anchors * n_classes
        box_pred, # bs * n_anchors * n_state
        cls_target, # bs * n_gt
        box_target, # bs * n_gt * n_state
    ): # 基于 pred target 生成监督信号，即每一个pred对应的真值
        bs, num_pred, num_cls = cls_pred.shape

        # 类别距离
        cls_cost = self._cls_cost(cls_pred, cls_target) # bs * [n_anchors, n_gt]
        # box距离
        box_target = self.encode_reg_target(box_target, box_pred.device) # bs * [n_gt * 11]。

        instance_reg_weights = []
        for i in range(len(box_target)): # 遍历每一个sample
            # 处理异常数据。非Nan的保留
            weights  = torch.logical_not(
                box_target[i].isnan()
            ).to(dtype=box_target[i].dtype)
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights
                    )
            instance_reg_weights.append(weights) # bs * [n_gt * 11]

        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights) # bs * [n_anchors * n_gt]

        # 匹配
        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost) # n_anchors * n_gt
                indices.append(
                    [
                        cls_pred.new_tensor(x, dtype=torch.int64)
                        for x in linear_sum_assignment(cost) # 返回 pred_id, target_id，代表匹配的结果
                    ]
                ) # bs * [pred_idx, target_idx]
            else:
                indices.append([None, None])

        # 基于匹配距离+匹配结果，生成监督信号
        output_cls_target = cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls # bs * n_anchors
        output_box_target = box_pred.new_zeros(box_pred.shape) # bs * n_anchors * n_state
        output_reg_weights = box_pred.new_zeros(box_pred.shape) # bs * n_anchors * n_state
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            # 匹配上的pred有监督信号，未匹配上的pred的监督信号为0 或者 num_cls
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx]

        # 每个sample中的每个预测insatnce对应的真实class，box，reg_weights（非nan的mask）
        # bs * n_anchors, bs * n_anchors * n_state, bs * n_anchors * n_state
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, 
                  cls_pred, # bs * n_anchors * n_classes
                  cls_target # bs * n_gt
                  ):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid() # 损失计算的时候才计算sigmoid，所以这里直接计算sigmoid
        cost = []
        for i in range(bs): # 遍历每一个sample
            # focal loss
            if len(cls_target[i]) > 0: # 如果有真实的目标
                # 每一个类别单独计算一次 focal loss
                # 负例的损失 -(1-alpha)*(p)^gamma*log(1-p). p越大，权重越大，loss越大
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                ) # n_anchors * n_classes
                # 正例的损失 -alpha*(1-p)^gamma*log(p). p越小，权重越大，loss越大
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                ) # n_anchors * n_classes

                cost.append(
                    # gt 的类别 对应一个 dt 的损失。越接近，正例损失越小，负例损失越大，cost越小
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]]) # n_anchors * n_gt
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost # bs * [n_anchors * n_gt]

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        # n_anchors * 1 * n_state, 1 * n_gt * n_state
                        torch.abs(box_pred[i, :, None] - box_target[i][None]) # n_anchors * n_gt * n_state
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    ) # L1 loss. n_anchors * n_gt
                    * self.box_weight
                ) # bs * [n_anchors * n_gt]
            else:
                cost.append(None)
        return cost
