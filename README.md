# Object Detection

This kit has code and procedures necessary to add new object classes into object detection model. Adding new classes in object recognition is computationally expensive and involves model fine-tuning to ensure it still performs well on old and new classes. Usually, it's a trade-off between cost of labeling new datasets (new objects in different environments and combinations), model training time (cost of GPU), and ML Engineering time spent on experimentation and development. To make this example more interactive we are providing a reduced version of [Common Objects in Context (COCO)](http://cocodataset.org) dataset that contains only classes that represent retail items. This dataset is further reduced to 100 images to reduce re-training time and make this example more interactive.

# Development Environment

This project is designed to run on [Neuro Platform](https://neu.ro), so you can jump into problem-solving right away.

## Directory Structure

| Mount Point              | Description                       | Storage URI                     |
|:------------------------ |:--------------------------------- |:------------------------------- |
|`/project/data/`          | Test images                       | `storage:detection-kit/data/`         |
|`/project/detection_kit/` | Python modules                    | `storage:detection-kit/detection_kit/`      |
|`/project/notebooks/`     | Jupyter notebooks                 | `storage:detection-kit/notebooks/`    |
|`/project/results/`       | Logs and results                  | `storage:detection-kit/results/`      |


## Development

Follow the instructions below to setup the environment and start Jupyter development session backed by CPU or GPU.

## Neuro Platform

* Setup development environment `make setup`
* Run Jupyter with GPU: `make jupyter`
* Kill Jupyter: `make kill_jupyter`

# Data

## Uploading via Web UI

From local machine run `make filebrowser` and open job's URL from your mobile device or desktop. Through a simple file explorer interface you can upload test images and perform file operations.

## Uploading via CLI

From local machine run `make upload_data`. This will push local files stored in `./data` into `storage:detection-kit/data` mounted to your development environment's `/project/data`.

## Uploading Dataset and Checkpoints

* Download [data](https://drive.google.com/a/neuromation.io/file/d/1WWQ33zM23udAPnTO6_Y-6w_P_Dgq7x0N/view?usp=sharing) [338 MB]
and unzip it into folder `data`. Please, check, that files structures
after this step looks like this:
```
data
    coco_example
        ...
    synt_example
        ...
    coco
        weights
            24_classes.ckpt
            25_classes.ckpt
        mini_coco
            train
                annots
                    1.jsonl
                    ...
                images
                    1.jpg
                    ...
            val
                annots
                    1.jsonl
                    ...
                images
                    1.jpg
                    ...
```
* `make setup`.
* `make upload-data`.
* `make jupter`.
