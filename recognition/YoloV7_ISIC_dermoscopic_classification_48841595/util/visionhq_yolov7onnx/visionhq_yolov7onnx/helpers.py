from typing import Dict, List, Optional

import numpy as np

from .utils.box import numpy_nms, xywh2xyxy


def filter_detections(
    predictions: np.ndarray,
    obj_thres: float,
    conf_thres: float,
    iou_thres: float,
    classes: Optional[Dict[int, dict]] = None,
    agnostic: bool = False,
) -> List[np.ndarray]:
    """Perform Non-Maximum Suppression (NMS) and filter by classes and confidence threshold

    Args:
        predictions:
            A 3-dimensional tensor of shape (batch_size, num_anchors, num_features).
            - batch_size: The number of images in the batch.
            - num_anchors: The number of anchor boxes used in the YOLOv7 model.
            - num_features: The total number of features for each anchor.
                * Features [0:4] represent bbox coordinates in [x_center, y_center, w, h] format.
                * Feature 4 represents the objectness score, indicating the confidence of an
                    object's presence in the bounding box.
                * Features from index 5 onwards represent the confidence scores for each class.
        obj_thres: The minimum objectness score threshold for a detection to be considered.
        conf_thres: The default minimum confidence score threshold for a detection to be considered.
        iou_thres: The Intersection Over Union (IOU) threshold for NMS. Detections with IOU greater
            than this threshold are considered overlapping and suppressed.
        classes: An optional dictionary mapping class indices (int) to class info.
            If provided, only detections of specified classes will be retained.
            Each class may have an optional 'conf_thres' key for class-specific threshold.
        agnostic: Whether to perform NMS agnostic of class or inclusive.

    Returns:
        A list of numpy arrays, one for each image in the batch.
        Each array contains detections for that image in the format [x1, y1, x2, y2, conf, cls].
    """
    max_wh = 7680  # maximum box width and height
    max_det = 300  # maximum number of detections per image

    output = np.zeros((0, 6))
    for idx, image_preds in enumerate(predictions):
        # Initial confidence-based filtering
        candidates = image_preds[image_preds[:, 4] > obj_thres]
        if not len(candidates):
            continue

        # Convert bounding box and compute class-wise scores
        boxes = xywh2xyxy(candidates[:, :4])
        class_scores = candidates[:, 5:] * candidates[:, 4:5]
        max_scores = np.max(class_scores, axis=1, keepdims=True)
        max_classes = np.argmax(class_scores, axis=1).astype(np.float32).reshape((-1, 1))

        # Assemble the boxes, scores, and class indices
        detections = np.hstack((boxes, max_scores, max_classes))

        # Class-based filtering and apply class-specific thresholds
        if classes:
            class_mask = np.isin(detections[:, 5], list(classes.keys()))
            detections = detections[class_mask]

            # Apply class-specific thresholds
            class_specific_mask = np.zeros(len(detections), dtype=bool)
            for class_idx, class_info in classes.items():
                class_mask = detections[:, 5] == class_idx
                class_conf_thres = class_info.get("conf_thres", conf_thres)
                class_specific_mask |= class_mask & (detections[:, 4] > class_conf_thres)

            detections = detections[class_specific_mask]
        else:
            # If no classes specified, use the default conf_thres
            detections = detections[detections[:, 4] > conf_thres]

        # If no boxes remain after filtering, continue
        if not len(detections):
            continue

        # Sort by confidence
        sorted_detections = detections[detections[:, 4].argsort()[::-1]]

        # Offset boxes by classes for NMS
        class_offsets = sorted_detections[:, 5:6] * (0 if agnostic else max_wh)
        offset_boxes = sorted_detections[:, :4] + class_offsets
        nms_indices = numpy_nms(offset_boxes, sorted_detections[:, 4], iou_thres)

        output = sorted_detections[nms_indices[:max_det]]

    return output
