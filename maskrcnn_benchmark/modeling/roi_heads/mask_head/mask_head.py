# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def keep_only_positive_boxes(boxes, batch_size_per_im):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        if len(inds) > batch_size_per_im:
            new_inds = inds[:batch_size_per_im]
            inds_mask[inds[batch_size_per_im:]] = 0
        else:
            new_inds = inds
        positive_boxes.append(boxes_per_image[new_inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

# TODO
def project_char_masks_on_boxes(segmentation_masks, segmentation_char_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    char_masks = []
    char_mask_weights = []
    decoder_targets = []
    word_targets = []
    M_H, M_W = discretization_size[0], discretization_size[1]
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    assert segmentation_char_masks.size == proposals.size, "{}, {}".format(
        segmentation_char_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, segmentation_char_mask, proposal in zip(segmentation_masks, segmentation_char_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M_W, M_H))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
        cropped_char_mask = segmentation_char_mask.crop(proposal)
        scaled_char_mask = cropped_char_mask.resize((M_W, M_H))
        char_mask, char_mask_weight, decoder_target, word_target = scaled_char_mask.convert(mode="seq_char_mask")
        char_masks.append(char_mask)
        char_mask_weights.append(char_mask_weight)
        decoder_targets.append(decoder_target)
        word_targets.append(word_target)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.long, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32), torch.stack(char_masks, dim=0).to(device, dtype=torch.long), torch.stack(char_mask_weights, dim=0).to(device, dtype=torch.float32), torch.stack(decoder_targets, dim=0).to(device, dtype=torch.long), torch.stack(word_targets, dim=0).to(device, dtype=torch.long)


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, proposal_matcher, discretization_size):
        super(ROIMaskHead, self).__init__()
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "char_masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        masks = []
        char_masks = []
        char_mask_weights = []
        decoder_targets = []
        word_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            char_segmentation_masks = matched_targets.get_field("char_masks")
            char_segmentation_masks = char_segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image, char_masks_per_image, char_masks_weight_per_image, decoder_targets_per_image, word_targets_per_image = project_char_masks_on_boxes(
                segmentation_masks, char_segmentation_masks, positive_proposals, self.discretization_size
            )

            masks.append(masks_per_image)
            char_masks.append(char_masks_per_image)
            char_mask_weights.append(char_masks_weight_per_image)
            decoder_targets.append(decoder_targets_per_image)
            word_targets.append(word_targets_per_image)

        return masks, char_masks, char_mask_weights, decoder_targets, word_targets

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals, self.cfg.MODEL.ROI_MASK_HEAD.MASK_BATCH_SIZE_PER_IM)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        if self.training and self.cfg.MODEL.CHAR_MASK_ON:
            mask_targets, char_mask_targets, char_mask_weights, decoder_targets, word_targets = self.prepare_targets(proposals, targets)
            decoder_targets = cat(decoder_targets, dim=0)
            word_targets = cat(word_targets, dim=0)
        if self.cfg.MODEL.CHAR_MASK_ON:
            if self.cfg.SEQUENCE.SEQ_ON:
                if not self.training:
                    if x.numel()>0:
                        mask_logits, char_mask_logits, seq_outputs, seq_scores, detailed_seq_scores = self.predictor(x)
                        result = self.post_processor(mask_logits, char_mask_logits, proposals, seq_outputs=seq_outputs, seq_scores=seq_scores, detailed_seq_scores=detailed_seq_scores)
                        return x, result, {}
                    else:
                        return None, None, {}
                mask_logits, char_mask_logits, seq_outputs = self.predictor(x, decoder_targets=decoder_targets, word_targets=word_targets)
                loss_mask, loss_char_mask = self.loss_evaluator(proposals, mask_logits, char_mask_logits, mask_targets, char_mask_targets, char_mask_weights)
                return x, all_proposals, dict(loss_mask=loss_mask, loss_char_mask=loss_char_mask, loss_seq=seq_outputs)
            else:
                mask_logits, char_mask_logits = self.predictor(x)
                if not self.training:
                    result = self.post_processor(mask_logits, char_mask_logits, proposals)
                    return x, result, {}
                loss_mask, loss_char_mask = self.loss_evaluator(proposals, mask_logits, char_mask_logits, mask_targets, char_mask_targets, char_mask_weights)
                return x, all_proposals, dict(loss_mask=loss_mask, loss_char_mask=loss_char_mask)
        else:
            mask_logits = self.predictor(x)
            if not self.training:
                result = self.post_processor(mask_logits, proposals)
                return x, result, {}
            loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
            return x, all_proposals, dict(loss_mask=loss_mask)

def build_roi_mask_head(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    return ROIMaskHead(cfg, matcher, (cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_H, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_W))
