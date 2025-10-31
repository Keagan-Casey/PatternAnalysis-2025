#“modules.py" containing the source code of the components of your model. Each component must be implemented as a class or a function 
import torch
from util.yolov7.models.yolo import Model

class YOLOv7SkinLesion(torch.nn.Module):
    def __init__(self, config_path='util/yolov7/cfg/training/yolov7.yaml', num_classes=2, pretrained=True):
        super().__init__()
        self.model = Model(config_path, ch=3, nc=num_classes, anchors=None)

        if pretrained:
            ckpt = torch.load('utils/yolov7/yolov7-tiny.pt', map_location='cpu')
            self.model.load_state_dict(ckpt['model'].state_dict(), strict=False)

    def forward(self, x):
        return self.model(x)


