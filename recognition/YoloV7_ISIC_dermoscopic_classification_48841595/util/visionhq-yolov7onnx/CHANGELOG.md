# Change Log

All notable changes to this package will be documented in this file.

## [2.1.4] - 2025-07-14

### Updated

- Updated `Box` model to remove unused attributes `x_center`, `y_center`, `width`, and `height`.
- Upgraded dependencies to latest compatible version.
  - from `pydantic~=2.7.1` to `pydantic~=2.10.6`.

## [2.1.3] - 2024-10-15

### Removed

- Removed `sliding_detection` feature

## [2.1.2] - 2024-09-24

### Added

- Added object threshold for NMS
- Added class-specific confidence thresholds

## [2.1.1] - 2024-08-20

### Added

- Added uuid for each Box object

## [2.1.0] - 2024-05-10

### Added

- Added agnostic parameter for NMS
- Added xywh properties for `Box` class as calculated values

### Changed

- Changed `Box` class to use Pydantic BaseModel

### Fixed

- Fixed `valid_area` parameter to be properly optional

## [2.0.0] - 2024-04-15

### Changed

- Renamed to class name to YOLOv7ONNX

## [2.0.0] - 2023-10-17

### Changed

- Integrated with utils 2.0 functions with modifications
- Improve functional code efficiency
- More customized sliding window feature
- Better time logging

### Added

- Batch inference

## [1.4.5] - 2023-03-24

### Updated

- Updated CV utilities version to 1.0.22

## [1.4.4] - 2022-12-13

### Added

- `img_size` parameter to specify model image size (pixels)

## [1.4.3] - 2022-10-30

### Added

- `execution_providers` parameter to specify onnxruntime Execution Providers

## [1.4.2] - 2022-10-28

### Changed

- Fixed CUDAExecutionProvider error from onnxruntime

## [1.4.1] - 2022-08-15

### Changed

- Reversed the order of `non_maximum_suppression` and `scale coords`
- Rename `non_maximum_suppression` to `nms_yolo`

## [1.4.0] - 2022-08-06

### Changed

- Changed `resources_path` to `model_path` to no longer supply classes.txt file
- Changed `cull_area` to `valid_area` and flips original functionality to accept deteciton in this area
- Changed `invert_cull` to `invert_valid` which flips the valid_area functionality
- Changed `split_indexes` to `sliding_detection_indexes` and will enable this funcitonality if a tuple is supplied, else do nothing
- Moved `scale_coords_xywh` and `non_max_suppression` helper functions into `visionhq_cv_utils`
- Switched all funcitonality needing `pytorch` in favour of `numpy` and `opencv`

### Added

- `classes` parameter to remove the need for a classes.txt file

### Removed

- `sliding_detection` parameter as redundant
-

## [1.3.1] - 2022-07-19

### Changed

- Switched inferencer from `cv2.dnn` to `onnxruntime`

## [1.3.0] - 2022-07-18

### Changed

- Removed redundant NMS function
- Switched inferencer from `onnxruntime` to `cv2.dnn`

### Added

- Added `cull_area` parameter to specify an area to exclude detections
- Added `invert_cull` parameter to flip the `cull_area` funcitonality
- Added `merge_overlapping` parameter to toggle merging bounding boxes that overlap each other

### Warnings

- Multiple threads of cv2.dnn.readNetFromONNX() may cause unexpected detections

## [1.2.0] - 2022-06-08

### Added

- Added `sliding_detection` parameter to toggle the sliding window feature
- Added `split_indexes` parameter to specify which of the 6 windows to detect on for sliding detection

## [1.1.2] - 2022-05-04

### Fixed

- Fixed inconsistent tensors on devices for GPU inferencing

## [1.1.0] - 2022-04-29

### Added

- Added Sliding window detection
  - Splits image into 6 sections and detects separately
  - Concatenates all detections afterwards

## [1.0.0]

First release of the package.

### Added

- Set up the project.
