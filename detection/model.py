from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN
)


def get_model(n_classes: int) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                                    pretrained_backbone=True,
                                    num_classes=n_classes)
    return model
