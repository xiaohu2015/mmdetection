import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from torch.nn import BatchNorm2d, ReLU
import math
from typing import Tuple

from ..builder import HEADS
from .anchor_head import AnchorHead

from mmdet.core import (anchor_inside_flags, images_to_levels, multi_apply,
                        reduce_mean, unmap)

try:
    from rraitools import VisualHelper
except ImportError:
    VisualHelper = None


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def cat(tensors, dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def show_pos_anchor(img_meta, gt_anchors, gt_bboxes, is_show=True):
    # 显示正样本
    assert 'img' in img_meta, print('Collect类中的meta_keys需要新增‘img’，用于可视化调试')
    img = img_meta['img'].data.numpy()
    mean = img_meta['img_norm_cfg']['mean']
    std = img_meta['img_norm_cfg']['std']
    # 默认输入是rgb数据，需要切换为bgr显示
    img = np.transpose(img.copy(), (1, 2, 0))
    img = img * std.reshape([1, 1, 3]) + mean.reshape([1, 1, 3])
    img = img.astype(np.uint8)
    img = VisualHelper.show_bbox(img, gt_anchors.cpu().numpy(), is_show=False)
    return VisualHelper.show_bbox(
        img, gt_bboxes.cpu().numpy(), color=(255, 255, 255), is_show=is_show)


from torch.nn import functional as F


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def giou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "sum",
        eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class UniformMatcher(nn.Module):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.
    """

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad()
    def forward(self, pred_boxes, anchors, targets):
        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        # Also concat the target boxes
        tgt_bbox = torch.cat([v.gt_boxes.tensor for v in targets])

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(tgt_bbox), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(tgt_bbox), p=1)

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v.gt_boxes.tensor) for v in targets]
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        indices = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # indices11 = torch.topk(C.split(sizes, -1)[0][0],
        #                        k=self.match_times,
        #                        dim=0,
        #                        largest=False)[1]
        # print(indices11)
        # indices11 = torch.topk(C.split(sizes, -1)[1][1],
        #                        k=self.match_times,
        #                        dim=0,
        #                        largest=False)[1]
        # print(indices11)
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist())
            for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in all_indices
        ]


_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Instances:
    def __init__(self, gt_bbox, gt_label):
        self.gt_boxes = Boxes(gt_bbox)
        self.gt_classes = gt_label


@torch.jit.script
class YOLOFBox2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is
    parameterized by 4 deltas: (dx, dy, dw, dh). The transformation scales
    the box's width and height by exp(dw), exp(dh) and shifts a box's center
    by the offset (dx * width, dy * height).

    We add center clamp for the predict boxes.
    """

    def __init__(
            self,
            weights: Tuple[float, float, float, float],
            scale_clamp: float = _DEFAULT_SCALE_CLAMP,
            add_ctr_clamp: bool = True,
            ctr_clamp: int = 32
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally
                set such that the deltas have unit variance; now they are
                treated as hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box
                scaling factors (dw and dh) are clamped such that they are
                <= scale_clamp.
            add_ctr_clamp (bool): Whether to add center clamp, when added, the
                predicted box is clamped is its center is too far away from
                the original anchor's center.
            ctr_clamp (int): the maximum pixel shift to clamp.

        """
        self.weights = weights
        self.scale_clamp = scale_clamp
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be
        used to transform the `src_boxes` into the `target_boxes`. That is,
        the relation ``target_boxes == self.apply_deltas(deltas,
        src_boxes)`` is true (unless any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g.,
                ground-truth boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[..., 2] - src_boxes[..., 0]
        src_heights = src_boxes[..., 3] - src_boxes[..., 1]
        src_ctr_x = src_boxes[..., 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[..., 1] + 0.5 * src_heights

        target_widths = target_boxes[..., 2] - target_boxes[..., 0]
        target_heights = target_boxes[..., 3] - target_boxes[..., 1]
        target_ctr_x = target_boxes[..., 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[..., 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=-1)
        assert (src_widths > 0).all().item(), \
            "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially different
                class-specific box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = boxes.to(deltas.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dx_width = dx * widths[..., None]
        dy_height = dy * heights[..., None]
        if self.add_ctr_clamp:
            dx_width = torch.clamp(dx_width,
                                   max=self.ctr_clamp,
                                   min=-self.ctr_clamp)
            dy_height = torch.clamp(dy_height,
                                    max=self.ctr_clamp,
                                    min=-self.ctr_clamp)
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx_width + ctr_x[..., None]
        pred_ctr_y = dy_height + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return pred_boxes.reshape(deltas.shape)


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
                (self.tensor[..., 0] >= -boundary_threshold)
                & (self.tensor[..., 1] >= -boundary_threshold)
                & (self.tensor[..., 2] < width + boundary_threshold)
                & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list):
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self):
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class YOLOFHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_num_convs=2,
                 reg_num_convs=4,
                 **kwargs):
        self.cls_num_convs = cls_num_convs
        self.reg_num_convs = reg_num_convs
        self.INF = 1e8
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)
        self.box2box_transform = YOLOFBox2BoxTransform(weights=(1., 1., 1., 1.))
        self.anchor_matcher = UniformMatcher()
        self.i = 0

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(BatchNorm2d(self.in_channels))
            cls_subnet.append(ReLU())
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(BatchNorm2d(self.in_channels))
            bbox_subnet.append(ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_anchors,
            kernel_size=3,
            stride=1,
            padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) +
            torch.clamp(objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    @torch.no_grad()
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss1(self,
              cls_scores,
              bbox_preds,
              gt_bboxes,
              gt_labels,
              img_metas,
              gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_idx_list, pos_predicted_boxes_list, target_boxes_list) = cls_reg_targets

        flatten_labels = torch.cat(labels_list).reshape(-1)
        pos_inds = ((flatten_labels >= 0)
                    &
                    (flatten_labels < self.num_classes)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_idx_list,
            pos_predicted_boxes_list,
            target_boxes_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox), num_total_samples

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, pos_idxs, pos_predicted_boxes, target_boxes, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        # ll = labels[label_weights > 0]
        # ll = ll[ll != 80]
        # print('---l', ll)

        # cls loss
        gt_classes_target = torch.zeros_like(cls_score)
        valid_idxs = (label_weights > 0) & (labels != 80)
        gt_classes_target[valid_idxs, labels[valid_idxs]] = 1
        loss_cls = sigmoid_focal_loss(
            cls_score[label_weights > 0],
            gt_classes_target[label_weights > 0],
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        )
        loss_cls = loss_cls / num_total_samples
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        loss_bbox = self.loss_bbox(
            pos_predicted_boxes,
            target_boxes,
            pos_idxs.float(),
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)

        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = [torch.cat(r, 0)]

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        # inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
        #                                    img_meta['img_shape'][:2],
        #                                    self.train_cfg.allowed_border)
        # if not inside_flags.any():
        #     return (None,) * 7
        # # assign gt and sample anchors
        # anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        # bbox_preds = bbox_preds[inside_flags, :]
        anchors = flat_anchors

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)

        assign_result = self.assigner.assign(
            decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        # TODO
        if False and VisualHelper is not None:
            # 统计下正样本个数
            print('----anchor分配正负样本后，正样本anchor可视化，白色bbox是gt----')
            gt_inds = assign_result.gt_inds  # 0 1 -1 正负忽略样本标志
            index = gt_inds > 0
            gt_inds = gt_inds[index]
            gt_anchors = anchors[index]
            print('单张图片中正样本anchor个数', len(gt_inds))
            show_pos_anchor(img_meta, gt_anchors, gt_bboxes)

        pos_idx = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property('pos_predicted_boxes')
        target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        # print('---', pos_inds)
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, pos_idx, pos_predicted_boxes, target_boxes)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.i += 1
        loss2, num_foreground_mm = self.loss1(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

        num_images = len(cls_scores[0])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, _ = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        anchors = [[Boxes(anchor_list[i][0])] for i in range(num_images)]
        pred_logits, pred_anchor_deltas = cls_scores, bbox_preds

        pred_logits = [permute_to_N_HWA_K(pred_logits[0], self.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(pred_anchor_deltas[0], 4)]

        gt_instances = [Instances(gt_bboxes[i], gt_labels[i]) for i in range(num_images)]

        indices = self.get_ground_truth(
            anchors, pred_anchor_deltas, gt_instances)
        losses, num_foreground = self.losses(
            indices, gt_instances, anchors,
            pred_logits, pred_anchor_deltas)
        if self.i % 10 == 0:
            print('==============================')
            d2_cls = round(losses['loss_cls'].item(), 4)
            d2_bbox = round(losses['loss_box_reg'].item(), 4)
            mm_cls = round(loss2['loss_cls'][0].item(), 4)
            mm_bbox = round(loss2['loss_bbox'][0].item(), 4)
            print(
                f'{self.i} d2_num_samper={num_foreground},mm_num_samper={num_foreground_mm},d2_loss_cls={d2_cls},mm_loss_cls={mm_cls},'
                f'd2_loss_bbox={d2_bbox},mm_loss_bbox={mm_bbox}', flush=True)
            assert num_foreground == num_foreground_mm
            assert d2_cls == mm_cls
            assert d2_bbox == mm_bbox
        return losses

    @torch.no_grad()
    def get_ground_truth(self, anchors, bbox_preds, targets):
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor.reshape(N, -1, 4)
        # Boxes(Tensor(N*R, 4))
        box_delta = cat(bbox_preds, dim=1)
        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.anchor_matcher(box_pred, all_anchors, targets)
        return indices

    def losses(self,
               indices,
               gt_instances,
               anchors,
               pred_class_logits,
               pred_anchor_deltas):
        pred_class_logits = cat(
            pred_class_logits, dim=1).view(-1, self.num_classes)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1).view(-1, 4)

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor
        # Boxes(Tensor(N*R, 4))
        predicted_boxes = self.box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors)
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)

        ious = []
        pos_ious = []
        for i in range(N):
            src_idx, tgt_idx = indices[i]
            # print(src_idx)
            iou = box_iou(predicted_boxes[i, ...],
                          gt_instances[i].gt_boxes.tensor)
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            a_iou = box_iou(anchors[i].tensor,
                            gt_instances[i].gt_boxes.tensor)
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        ious = torch.cat(ious)
        ignore_idx = ious > 0.7
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < 0.15

        src_idx = torch.cat(
            [src + idx * anchors[0].tensor.shape[0] for idx, (src, _) in
             enumerate(indices)])
        gt_classes = torch.full(pred_class_logits.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=pred_class_logits.device)
        gt_classes[ignore_idx] = -1
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(gt_instances, indices)])

        # target_classes_o[pos_ignore_idx] = -1
        pos_ignore_idx = pos_ignore_idx.cpu()
        target_classes_o = target_classes_o.cpu()
        target_classes_o[pos_ignore_idx] = -1
        pos_ignore_idx = pos_ignore_idx.to(pred_class_logits.device)

        # print(target_classes_o)
        # GPU
        # gt_classes[src_idx] = target_classes_o

        # cpu
        gt_classes = gt_classes.cpu()
        src_idx = src_idx.cpu()
        target_classes_o = target_classes_o.cpu()
        gt_classes[src_idx] = target_classes_o
        gt_classes = gt_classes.to(pred_class_logits.device)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        # print(gt_classes[foreground_idxs])
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        if get_world_size() > 1:
            dist.all_reduce(num_foreground)
        num_foreground = num_foreground * 1.0 / get_world_size()

        # cls loss
        loss_cls = sigmoid_focal_loss(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        )
        # reg loss
        target_boxes = torch.cat(
            [t.gt_boxes.tensor[i] for t, (_, i) in zip(gt_instances, indices)],
            dim=0)
        target_boxes = target_boxes[~pos_ignore_idx]
        matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
            src_idx[~pos_ignore_idx]]
        loss_box_reg = giou_loss(
            matched_predicted_boxes, target_boxes, reduction="sum")

        return {
                   "loss_cls": loss_cls / max(1, num_foreground),
                   "loss_box_reg": loss_box_reg / max(1, num_foreground),
               }, num_foreground
