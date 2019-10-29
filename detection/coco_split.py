import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path,
                        help='fold contains images and annotations')
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--val_fraq', type=float, default=0.1)
    return parser


def main(args: Namespace) -> None:
    im_dir = args.data_dir / 'images'
    ims = list(im_dir.glob('**/*.jpg'))
    annots = [Path(str(p).replace('.jpg', '.jsonl').replace('images', 'annot'))
              for p in ims]

    ims_train, ims_val, annots_train, annots_val = train_test_split(
        ims, annots, test_size=args.val_fraq, random_state=42)

    folds = (('train', ims_train, annots_train),
             ('val', ims_val, annots_val))

    for label, ims, annots in folds:
        print(f'Work with {label}.')

        root_dir = args.save_dir / label
        im_dir = root_dir / 'images'
        annot_dir = root_dir / 'annot'

        root_dir.mkdir(exist_ok=True)
        im_dir.mkdir(exist_ok=True)
        annot_dir.mkdir(exist_ok=True)

        for im, annot in tqdm(zip(ims, annots), total=len(ims)):
            shutil.copyfile(src=im, dst=im_dir / im.name)
            shutil.copyfile(src=annot, dst=annot_dir / annot.name)

        n_im = len(list(im_dir.glob('**/*.jpg')))
        n_annot = len(list(annot_dir.glob('**/*.jsonl')))

        assert n_im == n_annot, f'num im: {n_im}, num annot: {n_annot}'


if __name__ == '__main__':
    main(args=get_parser().parse_args())
