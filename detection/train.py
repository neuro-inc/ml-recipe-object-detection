from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.image as mpimg
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from detection.coco_subset import N_COCO_CLASSES
from detection.dataset import get_coco_datasets, CocoDataset
from detection.model import get_model, FasterRCNN
from detection.visualisation import draw_boxes
from pytorch_detection import utils
from pytorch_detection.engine import evaluate, train_one_epoch


def train(model: FasterRCNN, data_dir: Path,
          prev_ckpt: Optional[Path],
          n_epoch: int, batch_size: int,
          n_workers: int, ignore_labels: Tuple[int, ...],
          need_save: bool
          ) -> None:
    device = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')

    set_train, set_test, n_classes = get_coco_datasets(
        root_dir=data_dir, ignore_ids=ignore_labels)

    loader_train = DataLoader(dataset=set_train, batch_size=batch_size,
                              shuffle=True, num_workers=n_workers,
                              collate_fn=utils.collate_fn)

    loader_test = DataLoader(dataset=set_test, batch_size=batch_size,
                             shuffle=False, num_workers=n_workers,
                             collate_fn=utils.collate_fn)

    if prev_ckpt is not None:
        cur_epoch = int(Path(prev_ckpt).stem.split('_')[-1])
        state_dict = torch.load(prev_ckpt)
        model.load_state_dict(state_dict)
        print('Model was loaded from checkpoint.')
    else:
        cur_epoch = 0

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    weights_dir = Path(__file__).parent.parent / 'data' / 'weights'
    for epoch in range(cur_epoch, cur_epoch + n_epoch):
        train_one_epoch(model=model, optimizer=optimizer,
                        data_loader=loader_train, device=device,
                        epoch=epoch, print_freq=10)
        # lr_scheduler.step()
        evaluate(model=model, data_loader=loader_test, device=device)

        if need_save:
            save_example(model=model, dataset=set_test, idx=0, epoch=epoch)

            torch.save(model.state_dict(),
                       f=weights_dir / f'{data_dir.name}_{epoch}.ckpt')
            print('Model saved.')


def main(data_dir: Path, prev_ckpt: Optional[Path],
         n_epoch: int, batch_size: int,
         n_workers: int, ignore_labels=Tuple[int, ...]
         ) -> None:
    n_classes = N_COCO_CLASSES - len(ignore_labels)
    model = get_model(n_classes=n_classes)

    train(model=model, data_dir=data_dir, prev_ckpt=prev_ckpt,
          n_epoch=n_epoch, batch_size=batch_size,
          n_workers=n_workers, ignore_labels=ignore_labels,
          need_save=True)


def save_example(model: FasterRCNN,
                 dataset: CocoDataset,
                 idx: int,
                 epoch: int
                 ) -> None:
    model.eval()
    tensor, gt = dataset[idx]
    img = dataset.get_image(idx)
    pred = model([tensor.to(next(model.parameters()).device)])[0]
    img_pred = draw_boxes(img=img, annot=pred)
    img_gt = draw_boxes(img=img, annot=gt)

    save_dir = Path(__file__).parent.parent / 'data'
    mpimg.imsave(save_dir / f'pred_{epoch}.jpg', img_pred)
    mpimg.imsave(save_dir / 'gt.jpg', img_gt)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--prev_ckpt', type=Path, default=None)
    parser.add_argument('--n_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--ignore_labels', default=tuple(),
                        type=Tuple[int, ...])
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(data_dir=args.data_dir, prev_ckpt=args.prev_ckpt,
         n_epoch=args.n_epoch, batch_size=args.batch_size,
         n_workers=args.n_workers, ignore_labels=args.ignore_labels)
