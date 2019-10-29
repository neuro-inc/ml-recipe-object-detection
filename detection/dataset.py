import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Union, Callable

import torch
from PIL import Image
from torch import float32, int64, uint8
from torch.utils.data import Dataset

import pytorch_detection.transforms as t
from detection.coco_subset import CLS_SELECT, N_COCO_CLASSES

TTarget = Dict[str, Union[torch.FloatType, torch.IntType]]


class ObjDataset(Dataset, ABC):
    _ims: Tuple[Path, ...]
    _gts: Tuple[Path, ...]
    _transforms: t.Compose
    f_json_parse: Callable[[Path], TTarget]

    def __init__(self,
                 ims: Tuple[Path, ...],
                 gts: Tuple[Path, ...],
                 transforms: t.Compose,
                 ):
        self._ims = ims
        self._gts = gts
        self._transforms = transforms

        self.set_json_parser()

    def __len__(self) -> int:
        return len(self._ims)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, TTarget]:
        img = Image.open(self._ims[idx]).convert('RGB')
        target = self.f_json_parse(self._gts[idx])
        img, target = self._transforms(img, target)
        return img, target

    def get_image(self, idx: int) -> Image.Image:
        return Image.open(self._ims[idx]).convert('RGB')

    @abstractmethod
    def set_json_parser(self) -> None:
        raise NotImplementedError()


class SyntDataset(ObjDataset):

    def set_json_parser(self) -> None:
        self.f_json_parse = parse_synt_json


class CocoDataset(ObjDataset):
    def set_json_parser(self) -> None:
        self.f_json_parse = parse_coco_json


def get_transform(aug: bool) -> t.Compose:
    if aug:
        # todo looks like flip doesn't work well
        # transforms = [t.ToTensor(), t.RandomHorizontalFlip(0.5)]
        transforms = [t.ToTensor()]
    else:
        transforms = [t.ToTensor()]
    return t.Compose(transforms)


# COCO DATA


def parse_coco_json(json_path: Path,
                    ignore_ids: Tuple[int, ...] = ()
                    ) -> TTarget:
    boxes = torch.zeros([0, 4], dtype=float32)
    labels = torch.zeros([0], dtype=int64)
    areas = torch.zeros([0], dtype=float32)
    iscrowds = torch.zeros([0], dtype=uint8)

    with open(json_path, 'r') as f:
        for line in f:
            obj = json.loads(line)

            if obj['label'] not in ignore_ids:
                x0, y0, w, h = obj['bbox']
                x1, y1 = x0 + w + 1, y0 + h + 1
                box = torch.tensor([x0, y0, x1, y1], dtype=float32)
                label = torch.tensor(obj['label'], dtype=int64)
                area = torch.tensor(obj['area'], dtype=float32)
                iscrowd = torch.tensor(obj['is_crowd'], dtype=uint8)

                boxes = torch.cat([boxes, box.unsqueeze(0)], 0)
                labels = torch.cat([labels, label.unsqueeze(0)], 0)
                areas = torch.cat([areas, area.unsqueeze(0)], 0)
                iscrowds = torch.cat([iscrowds, iscrowd.unsqueeze(0)], 0)

    image_id = torch.tensor(obj['image_id'], dtype=int64)

    data = {'boxes': boxes, 'labels': labels, 'image_id': image_id,
            'area': areas, 'iscrowd': iscrowds}

    return data


def get_mock_coco_datasets() -> Tuple[CocoDataset, CocoDataset, int]:
    data_dir = Path(Path(__file__).parent.parent) / 'data' / 'coco_example'
    ims = tuple([data_dir / f'000000189078.jpg'])
    gts = tuple([Path(str(im).replace('.jpg', '.jsonl')) for im in ims])

    train_set = CocoDataset(ims, gts, transforms=get_transform(aug=True))
    test_set = CocoDataset(ims, gts, transforms=get_transform(aug=False))

    n_classes = len(CLS_SELECT) + 1
    return train_set, test_set, n_classes


def get_coco_datasets(root_dir: Path,
                      ignore_ids: Tuple[int, ...] = ()
                      ) -> Tuple[CocoDataset, CocoDataset, int]:
    def im_to_annot_name(im_name: Path) -> Path:
        return Path(
            str(im_name).replace('.jpg', '.jsonl').replace('images', 'annots'))

    ims_train = tuple((root_dir / 'train' / 'images').glob('**/*.jpg'))
    ims_test = tuple((root_dir / 'val' / 'images').glob('**/*.jpg'))

    annots_train = tuple([im_to_annot_name(im) for im in ims_train])
    annots_test = tuple([im_to_annot_name(im) for im in ims_test])

    transforms_train = get_transform(aug=True)
    transforms_test = get_transform(aug=False)

    train_set = CocoDataset(ims_train, annots_train, transforms_train)
    test_set = CocoDataset(ims_test, annots_test, transforms_test)

    train_set.f_json_parse = lambda path: parse_coco_json(path, ignore_ids)
    test_set.f_json_parse = lambda path: parse_coco_json(path, ignore_ids)

    n_classes = N_COCO_CLASSES - len(ignore_ids)
    return train_set, test_set, n_classes


def check_on_coco() -> None:
    dataset, _, _ = get_mock_coco_datasets()
    print(dataset[0])


# MOCK SYNT DATA

# class '0' reserved for background
sku_to_enum_synt = {
    4600494600579: 1,
    4690228016752: 2,
    7891024132470: 3,
    4607025398424: 4,
    5449000223609: 5,
    4607065730109: 6,
    4607065375966: 7
}


def parse_synt_json(json_path: Path) -> TTarget:
    with open(json_path, 'r') as j:
        content = json.load(j)

    boxes = torch.zeros([0, 4], dtype=float32)
    labels = torch.zeros([0], dtype=int64)
    images_id = torch.tensor(hash(json_path.stem), dtype=int64)
    areas = torch.zeros([0], dtype=float32)
    iscrowds = torch.zeros([0], dtype=uint8)

    for data_cls in content:

        if not any([c.isalpha() for c in data_cls['id']]):

            sku = int(data_cls['id'])
            label = torch.tensor(sku_to_enum_synt[sku], dtype=int64)

            for item in data_cls['data']:
                bb = item['boundingBox']
                x0, y0, h, w = bb['X'], bb['Y'], bb['Height'], bb['Width']
                x1 = x0 + w + 1
                y1 = y0 + h + 1
                box = torch.tensor([x0, y0, x1, y1], dtype=float32)
                area = torch.tensor(h * w, dtype=float32)
                iscrowd = torch.tensor(False, dtype=uint8)

                boxes = torch.cat([boxes, box.unsqueeze(0)], 0)
                labels = torch.cat([labels, label.unsqueeze(0)], 0)
                areas = torch.cat([areas, area.unsqueeze(0)], 0)
                iscrowds = torch.cat([iscrowds, iscrowd.unsqueeze(0)], 0)

    assert labels.size() == areas.size() == iscrowds.size()
    assert boxes.shape[0] == labels.size()[0]

    data = {'boxes': boxes, 'labels': labels, 'image_id': images_id,
            'area': areas, 'iscrowd': iscrowds}

    return data


def get_mock_synt_datasets() -> Tuple[SyntDataset, SyntDataset, int]:
    # mock dataset for tests, which includes only 1 labeled image

    data_dir = Path(Path(__file__).parent.parent) / 'data' / 'synt_example'
    ims = tuple([data_dir / f'sample1.jpg'])
    gts = tuple([Path(str(im).replace('.jpg', '.json')) for im in ims])

    train_set = SyntDataset(ims, gts, transforms=get_transform(aug=True))
    test_set = SyntDataset(ims, gts, transforms=get_transform(aug=False))

    n_classes = len(sku_to_enum_synt) + 1
    return train_set, test_set, n_classes


def check_on_synt() -> None:
    dataset, _, _ = get_mock_synt_datasets()
    print(dataset[0])


if __name__ == '__main__':
    check_on_synt()
    check_on_coco()
