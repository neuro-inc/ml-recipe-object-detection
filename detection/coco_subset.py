import json
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import random
from typing import Dict, Tuple, List

from bidict import bidict
from tqdm import tqdm

TColor = Tuple[float, ...]


def rand_color() -> TColor:
    return tuple([int(255 * random()) for _ in range(3)])


def rand_colors(n_colors: int) -> List[TColor]:
    colors = [rand_color() for _ in range(n_colors)]
    return colors


CLS_SELECT = {
    'book': 1, 'vase': 2, 'scissors': 3,
    'teddy bear': 4, 'hair drier': 5,
    'toothbrush': 6, 'potted plant': 7,
    'apple': 8, 'orange': 9, 'carrot': 10,
    'banana': 11, 'sandwich': 12, 'broccoli': 13,
    'hot dog': 14, 'pizza': 15, 'cake': 16,
    'donut': 17, 'wine glass': 18, 'bottle': 19,
    'cup': 20, 'fork': 21, 'spoon': 22,
    'knife': 23, 'bowl': 24, 'sports ball': 25
}
COLORS = rand_colors(len(CLS_SELECT))
N_COCO_CLASSES = len(CLS_SELECT) + 1


def get_coco_mapping(annot_file: Path) -> Dict[str, int]:
    with open(annot_file, 'r') as j:
        data = json.load(j)

    mapping = {cls['name']: cls['id'] for cls in data['categories']}
    return mapping


def pad_image_id(image_id: str) -> str:
    name_len = 12  # standart for coco_example
    n_pad = name_len - len(image_id)
    name = '0' * n_pad + image_id
    return name


def convert_and_save(annot_file: Path,
                     coco_im_dir: Path,
                     save_dir: Path
                     ) -> None:
    coco_mapping = bidict(get_coco_mapping(annot_file))

    with open(annot_file, 'r') as j:
        data = json.load(j)

    for n, obj in enumerate(tqdm(data['annotations'])):
        name = coco_mapping.inv[obj['category_id']]
        image_id = obj['image_id']
        image_id_pad = pad_image_id(str(image_id))

        if name in CLS_SELECT.keys():
            with open(save_dir / 'annot' / f'{image_id_pad}.jsonl', 'w') as out:
                annot = {
                    'bbox': [int(x) for x in obj['bbox']],
                    'label': CLS_SELECT[name], 'image_id': image_id,
                    'area': round(obj['area']), 'is_crowd': obj['iscrowd'],
                }

                out.write(json.dumps(annot) + '\n')

                im_name = f'{image_id_pad}.jpg'
                if not (save_dir / im_name).is_file():
                    shutil.copy(src=coco_im_dir / im_name,
                                dst=save_dir / 'images' / im_name)


def main(args: Namespace) -> None:
    im_dir = args.save_dir / 'images'
    annot_dir = args.save_dir / 'annots'

    im_dir.mkdir(exist_ok=True)
    annot_dir.mkdir(exist_ok=True)

    for fold in ['train2017', 'val2017']:
        im_dir = args.coco_dir / fold
        annot_file = args.coco_dir / 'annotations' / f'instances_{fold}.json'
        if im_dir.is_dir() and annot_file.is_file():
            print(fold)
            convert_and_save(annot_file=annot_file, coco_im_dir=im_dir,
                             save_dir=args.save_dir)

    n_im = len(list((args.save_dir / 'images').glob('*.jpg')))
    n_annot = len(list(annot_dir.glob('*.jsonl')))
    assert n_im == n_annot, f'num im: {n_im}, num annot: {n_annot}'


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--coco_dir', type=Path)
    parser.add_argument('--save_dir', type=Path)
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
