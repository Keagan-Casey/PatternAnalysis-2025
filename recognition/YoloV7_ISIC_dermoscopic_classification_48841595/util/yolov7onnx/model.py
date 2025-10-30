from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from .helpers import filter_detections
from .utils import Logger, Timer
from .utils.box import Box, convert2boxes, scale_boxes
from .utils.image import letterbox

# Initialize logger
logger = Logger(debug_mode=False)


class CVWrapper(ABC):
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = self.from_disk(model_path)

    @abstractmethod
    def to_disk(self, model_path):
        """Serialisation for a specific classifier type."""
        pass

    @abstractmethod
    def from_disk(self, model_path):
        """Deserialisation for a specific classifier type."""
        pass

    @abstractmethod
    def forward(self, data):
        """Forward method for a specific classifier type."""
        pass


class YOLOv7ONNX(CVWrapper):
    def __init__(
        self,
        model_path: str,
        img_size: Tuple[int, int] = (640, 640),
        classes: Optional[Dict[int, dict]] = None,
        obj_thres: float = 0.25,
        conf_thres: float = 0.4,
        iou_thres: float = 0.45,
        batch_size: int = 1,
        agnostic: bool = False,
        valid_area: Optional[Tuple[float, float, float, float]] = None,
        execution_providers: Optional[List[str]] = None,
    ):
        """YOLOv7ONNX is a wrapper class for the YOLOv7 ONNX model.

        Args:
            model_path: Path to the ONNX model.
            img_size: Tuple of image size in the format (height, width).
            classes: Dictionary of classes in the format {class_idx: class_info}.
            obj_thres: Object threshold for filtering out low-confidence detections.
            conf_thres: Confidence threshold for filtering out low-confidence detections.
            iou_thres: IoU threshold for eliminating duplicate detections.
            batch_size: Batch size for running inference.
            agnostic: Whether to perform NMS agnostic of class or inclusive.
            valid_area: Tuple of valid area in the format (x_min, y_min, x_max, y_max).
            execution_providers: List of execution providers to use.
        """
        # Define the execution provider
        self.providers = (
            ort.get_available_providers() if execution_providers is None else execution_providers
        )

        # Initialization
        super().__init__(model_path)

        # Define the model resources
        self.classes = classes
        self.img_size = img_size
        self.obj_thres = obj_thres
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.agnostic = agnostic
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Define the detection refinement
        self.valid_area = valid_area

        logger.good(f"Initialized YOLOv5ONNX with batch size {self.batch_size}")

    def from_disk(self, model_path: str) -> ort.InferenceSession:
        """Load an ONNX model from disk and create an InferenceSession with the specified execution
        providers."""
        # Preparing resource path
        model = Path(model_path)
        if not model.exists() or not model.is_file() or not model.suffix == ".onnx":
            raise FileNotFoundError("Path to model is invalid or does not exist")

        # Setting up session options
        #sess_options = ort.SessionOptions()
        #sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #sess_options.intra_op_num_threads = 6
        #sess_options.add_session_config_entry('session.dynamic_block_base', '4')

        # Setting up session hardware to load model and weights from ONNX
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Create data input binding
        #self.io_binding = self.session.io_binding()

        return self.session

    def to_disk(self):
        pass

    def warmup(self, num_warmup_runs: int = 5, log_inference_time=False) -> None:
        """Performs warm-up runs to optimize the model for inference."""
        # Generate dummy input data
        dummy_input = np.random.rand(*self.img_size, 3).astype(np.float32)

        # Perform warm-up runs
        for _ in range(num_warmup_runs):
            self.forward(dummy_input, log_inference_time=log_inference_time)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocesses the input images to obtain the input tensor."""

        preprocessed_img = (
            letterbox(img, self.img_size).transpose((2, 0, 1)).astype(np.float32) / 255
        )

        return preprocessed_img

    def postprocess(
        self, predictions: np.ndarray, raw_image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Post-process detections from the YOLO model.

        This involves:
        1. Filtering out low-confidence detections.
        2. Applying non-maximum suppression (NMS) to eliminate overlapping detections.
        3. Rescaling bounding box coordinates to original image dimensions and format.

        Args:
            predictions: A tensor containing predictions from the model, with shape
                        (batch_size, num_anchors, num_features). Bounding box coordinates are in
                        [x_center, y_center, w, h] format.
            raw_image_shapes: A list containing the original shapes of each image in the format
                            [height, width].

        Returns:
            List of detections for each image. Each detection is a list in the format
            [x_min, y_min, x_max, y_max, confidence, class_index], where coordinates are relative
            ratios to the original image dimensions.
        """

        # Apply NMS and filter detections by confidence threshold and class
        filtered_dets = filter_detections(
            predictions,
            self.obj_thres,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic,
        )

        # Rescale boxes to match the original image sizes and convert to ratio format
        rescaled_dets = scale_boxes(filtered_dets, self.img_size, raw_image_shape, to_ratio=True)

        return rescaled_dets

    def forward(
        self, image_data: np.ndarray, log_inference_time: bool = False
    ) -> List[Box]:
        """Run inference on a list of images.

        Args:
            image_data: List of image arrays for inference.
            log_inference_time: Whether to log the time taken for inference.

        Returns:
            List of detected bounding boxes for each image.
        """

        # Initialize timers for different stages of the pipeline
        with Timer() as forward_timer:
            et = {"Preprocess": Timer(), "Inference": Timer(), "Postprocess": Timer()}

            imgsz = image_data.shape[:2]

            # Preprocess image
            with et["Preprocess"]:
                input_tensor = self.preprocess(image_data)

            # Run inference on the preprocessed batch
            with et["Inference"]:
                #ort_input_value = ort.OrtValue.ortvalue_from_numpy(input_tensor[None], 'cuda', 0)
                #self.io_binding.bind_input(
                #    name=self.input_name,
                #    device_type=ort_input_value.device_name(),
                #    device_id=0,
                #    element_type=np.float32,
                #    shape=ort_input_value.shape(),
                #    buffer_ptr=ort_input_value.data_ptr()
                #)
                #self.io_binding.bind_output(self.output_names[0])
                
                #self.session.run_with_iobinding(self.io_binding)
                #pred = self.io_binding.copy_outputs_to_cpu()[0]
    
                pred = self.session.run(
                    self.output_names, {self.input_name: input_tensor[None]}
                )[0]
            # Postprocess the inference results
            with et["Postprocess"]:
                pred = self.postprocess(pred, imgsz)

            # Convert postprocessed results to Box format and append to detections
            boxes = convert2boxes(pred, self.classes, self.valid_area)
            detections = boxes

        # Log the time taken for inference, if required
        if log_inference_time:
            total_t = f"Total inference takes {forward_timer.elapsed * 1000:.2f}ms"
            avg_prep_t = et["Preprocess"].elapsed * 1000
            avg_infer_t = et["Inference"].elapsed * 1000
            avg_post_t = et["Postprocess"].elapsed * 1000
            logger.info(f"{total_t} ({avg_prep_t:.0f}+{avg_infer_t:.0f}+{avg_post_t:.0f})")

        return detections
