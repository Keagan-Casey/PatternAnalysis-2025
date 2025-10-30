from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel


class Box(BaseModel):
    """Represents a bounding box in relative coordinates (0-1 range).

    Attributes:
        id: Unique identifier.
        x_min: X-coordinate of top-left corner (relative).
        y_min: Y-coordinate of top-left corner (relative).
        x_max: X-coordinate of bottom-right corner (relative).
        y_max: Y-coordinate of bottom-right corner (relative).
        label: Class/category label.
        score: Detection confidence (0-1).
        estimated_width: Optional real-world width estimate.
        estimated_height: Optional real-world height estimate.
        estimated_area: Optional real-world area estimate.
    """

    id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label: str
    estimated_width: Optional[float] = None
    estimated_height: Optional[float] = None
    estimated_area: Optional[float] = None


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """Convert bounding box format from [x1, y1, x2, y2] to [x_center, y_center, w, h]"""
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding box format from [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Apply non-maximum suppression (NMS) on bounding boxes using NumPy.

    Args:
        boxes: Bounding boxes of shape (N, 4), where N is the number of boxes.
            Each box is represented as [x1, y1, x2, y2].
        scores: Array of shape (N,) containing the confidence scores of each bounding box.
        iou_threshold: Overlap threshold for NMS.

    Returns:
        List of indices of the boxes that have been kept after applying NMS.
    """
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Calculate the area of each bounding box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort box indices by their scores in ascending order
    idxs = scores.argsort()

    keep = []

    # While there are still boxes left...
    while len(idxs) > 0:
        # Index of the current box with the highest score
        i = idxs[-1]
        keep.append(i)

        # Remove the current box from the list of indices
        idxs = idxs[:-1]

        # Get the intersection coordinates
        xx1 = np.maximum(boxes[i, 0], boxes[idxs, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs, 3])

        # Calculate the width and height of the intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the area of the intersection
        intersection = w * h

        # Compute the union area by using: Union(A,B) = A + B - Inter(A,B)
        union = areas[i] + areas[idxs] - intersection

        # Compute the IoU value
        iou = intersection / (union + 1e-9)  # Adding a small value to prevent division by zero

        # Filter out boxes that have IoU value greater than the threshold
        idxs = idxs[iou <= iou_threshold]

    return keep


def scale_boxes(
    boxes: np.ndarray,
    target_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
    to_ratio: bool = False,
) -> np.ndarray:
    """Scale the bounding boxes from the original image to the new image.

    Args:
        boxes: An array containing bounding box in the format [x1, y1, x2, y2] with absolute values.
        target_shape: The height and width of the new image.
        original_shape: The height and width of the original image.
        to_ratio: If True, scales the boxes to the ratio of the new image to the original image.
                  Defaults to False.

    Returns:
        Scaled bounding boxes.
    """
    # If there are no boxes, return an empty array
    if len(boxes) == 0:
        return np.array([])

    # Calculate the scaling factor for both width and height
    gain = min(target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])

    # Calculate padding to maintain the aspect ratio while scaling
    padding = (
        (target_shape[1] - original_shape[1] * gain) / 2,  # width padding
        (target_shape[0] - original_shape[0] * gain) / 2,  # height padding
    )

    # Adjust bounding boxes according to padding and scaling factor
    boxes[:, [0, 2]] -= padding[0]  # Adjust x-coordinates
    boxes[:, [1, 3]] -= padding[1]  # Adjust y-coordinates
    boxes[:, :4] /= gain  # Rescale by gain

    # Ensure the coordinates are within the image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])

    if to_ratio:
        # Convert coordinates to ratio format
        boxes[:, [0, 2]] /= original_shape[1]
        boxes[:, [1, 3]] /= original_shape[0]
    else:
        # Round coordinates to integers
        boxes = np.round(boxes).astype(int)

    return boxes


def is_valid_box(d, valid_area: Tuple[float, float, float, float]):
    cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2  # centroid computation
    return valid_area[0] < cx < valid_area[2] and valid_area[1] < cy < valid_area[3]


def convert2boxes(
    detections: np.ndarray,
    classes: Optional[Dict[int, dict]] = None,
    valid_area: Optional[Tuple[float, float, float, float]] = None,
) -> List[Box]:
    """Convert the output of the model to a list of boxes.

    Args:
        detections: The output of the model [[x1, y1, x2, y2, score, class], ...].
        classes: The list of classes to filter by.
        valid_area: The area to filter by.

    Returns:
        A list of Box instances representing the detections.
    """
    out_boxes = []
    for det in detections:
        if valid_area and not is_valid_box(det, valid_area):
            continue
        out_boxes.append(
            Box(
                id=str(uuid4()),
                x_min=det[0],
                y_min=det[1],
                x_max=det[2],
                y_max=det[3],
                score=det[4],
                label=classes[det[5]]["name"] if classes else str(det[5]),
            )
        )

    return out_boxes
