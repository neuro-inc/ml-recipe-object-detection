from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import nms

from detection.coco_subset import COLORS, CLS_SELECT
from detection.dataset import TTarget, get_mock_synt_datasets

TColor = Tuple[float, ...]


def predict_and_show(model: FasterRCNN,
                     im_pil: Image.Image,
                     im_tensor: Tensor,
                     score_th: float = 0.5
                     ) -> None:
    model.eval()

    pred = model([im_tensor])[0]

    # filter predicted boxes by scores and nms
    ii_select = nms(pred['boxes'], pred['scores'], iou_threshold=0.5).tolist()
    ii_select.extend(torch.nonzero(pred['scores'] > score_th)[:0].tolist())
    ii_select = list(set(ii_select))
    pred = {key: val[ii_select] for key, val in pred.items()}

    plt.figure(figsize=(14, 14))

    plt.subplot(1, 3, 1)
    plt.imshow(im_pil)
    plt.axis('image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(draw_boxes(img=im_pil, annot=pred))
    plt.title('predict')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    show_legend(CLS_SELECT, COLORS)
    plt.show()


def draw_boxes(img: Image.Image, annot: TTarget) -> np.ndarray:
    boxes = annot['boxes']
    labels = annot['labels'].tolist()
    n_obj = boxes.shape[0]

    for i in range(n_obj):
        x0, y0, x1, y1 = boxes[i, :].tolist()
        color = COLORS[labels[i] - 1]
        img = cv2.rectangle(img=np.array(img), color=color,
                            pt1=(int(x0), int(y0)),
                            pt2=(int(x1), int(y1)),
                            thickness=10)
    return img


def show_legend(names: List[str], colors: List[TColor]) -> None:
    plt.title('Legend')
    gap, n_row = 8, 15
    for i, (name, color) in enumerate(zip(names, colors)):
        x, y = gap * (i // n_row), n_row - i % n_row
        plt.scatter(x=x, y=y, color=[c / 255 for c in color])
        plt.text(x=x + 1, y=y, s=f'{i + 1} {name}')
    plt.axis('off')
    plt.axis('square')


def check() -> None:
    dataset, _, _ = get_mock_synt_datasets()
    _, target = dataset[0]

    img = dataset.get_image(0)

    img = draw_boxes(img, target)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    check()
