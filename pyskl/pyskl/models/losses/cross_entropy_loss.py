# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):
    """Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard, misclassified
    examples. Originally proposed in "Focal Loss for Dense Object Detection"
    (Lin et al., ICCV 2017).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Higher values focus more on hard
            examples. Default: 2.0.
        alpha (float | list[float] | None): Class balance weight. Can be:
            - float: Applied to positive class in binary classification
            - list: Per-class weights for multi-class classification
            - None: No class weighting (alpha=1 for all classes)
            Default: None.
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Alternative to alpha for per-class
            weights. If both alpha and class_weight are set, class_weight is
            used. Default: None.
    """

    def __init__(self, gamma=2.0, alpha=None, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.gamma = gamma
        # class_weight takes precedence over alpha for compatibility
        if class_weight is not None:
            self.alpha = torch.Tensor(class_weight)
        elif alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score (logits), shape (N, C).
            label (torch.Tensor): The ground truth label, shape (N,) for hard
                labels or (N, C) for soft labels.
            kwargs: Any keyword argument (unused).

        Returns:
            torch.Tensor: The computed focal loss.
        """
        num_classes = cls_score.size(1)
        
        # Compute softmax probabilities
        p = F.softmax(cls_score, dim=1)
        
        if cls_score.size() == label.size():
            # Soft label case
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            
            # p_t for soft labels: weighted sum
            p_t = (p * label).sum(dim=1)
            
            # Focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # Log softmax for numerical stability
            log_p = F.log_softmax(cls_score, dim=1)
            
            # Compute loss
            loss = -focal_weight.unsqueeze(1) * label * log_p
            
            # Apply alpha if provided
            if self.alpha is not None:
                if isinstance(self.alpha, torch.Tensor):
                    alpha = self.alpha.to(cls_score.device)
                    loss = loss * alpha.unsqueeze(0)
            
            loss = loss.sum(dim=1).mean()
        else:
            # Hard label case
            # Get probability of true class
            ce_loss = F.cross_entropy(cls_score, label, reduction='none')
            
            # p_t: probability of correct class
            p_t = p.gather(1, label.unsqueeze(1)).squeeze(1)
            
            # Focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # Apply alpha (class weights)
            if self.alpha is not None:
                if isinstance(self.alpha, torch.Tensor):
                    alpha = self.alpha.to(cls_score.device)
                    # Get alpha for each sample's true class
                    alpha_t = alpha.gather(0, label)
                    focal_weight = alpha_t * focal_weight
                else:
                    # Scalar alpha (for binary-like weighting)
                    focal_weight = self.alpha * focal_weight
            
            loss = (focal_weight * ce_loss).mean()
        
        return loss
