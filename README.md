# Object Detection

This recipe contains the code and procedures that are required in order to add new object classes into the object detection model. In object recognition, the addition of new classes  is computationally expensive and involves fine-tuning to ensure the model performs well on both old and new classes. Usually, this involves a trade-off between the cost of labeling new datasets (new objects in different environments and combinations), model training time (cost of GPU), and ML Engineering time spent on experimentation and development. To make this recipe interactive, we are providing a reduced version of the [Common Objects in Context (COCO)](http://cocodataset.org) dataset that contains only classes that represent retail items. This dataset is further reduced to 100 images to reduce re-training time. 

# Quick Start

##### 0. Sign up at [neu.ro](https://neu.ro)

##### 1. Install CLI and log in
```shell
pip install -U neuromation
neuro login
```

##### 2. Run the recipe

```shell
git clone git@github.com:neuromation/ml-recipe-object-detection.git
cd ml-recipe-object-detection
make setup
make jupyter
```

# Development Environment

This project is designed to run on [Neuro Platform](https://neu.ro), so you can dive into problem-solving right away.

## Directory structure

| Local directory                      | Description       | Storage URI                                                                  | Environment mounting point |
|:------------------------------------ |:----------------- |:---------------------------------------------------------------------------- |:-------------------------- | 
| `data/`                              | Data              | `storage:ml-recipe-object-detection/data/`                              | `/ml-recipe-object-detection/data/` | 
| `modules/` | Python modules    | `storage:ml-recipe-object-detection/modules/` | `/ml-recipe-object-detection/modules/` |
| `notebooks/`                         | Jupyter notebooks | `storage:ml-recipe-object-detection/notebooks/`                         | `/ml-recipe-object-detection/notebooks/` |
| No directory                         | Logs and results  | `storage:ml-recipe-object-detection/results/`                           | `/ml-recipe-object-detection/results/` |

## Development

Follow the instructions below to set up the environment and start your Jupyter Notebook development session.

### Setup development environment

`make setup`

* Several files from the local project upload to the platform’s storage (namely, `requirements.txt`, `apt.txt`, `setup.cfg`).
* A new job starts in our base environment.
* Pip requirements from `requirements.txt` and apt applications from `apt.txt` install in this environment.
* The updated environment is saved under a new project-dependent name and will be used later on.

### Run Jupyter with GPU

`make jupyter`

* The content of `modules` and `notebook` directories upload to the platform’s storage.
* A job with Jupyter is started, and its web interface opens in the local web browser window.

### Kill Jupyter

`make kill-jupyter`

This command terminates the job with Jupyter Notebooks. The notebooks remain saved on the platform’s storage. If you’d like  to download them to the local `notebooks/` directory, just run `make download-notebooks`.

### Help

`make help`

## Data

### Uploading via Web UI

On your local machine, run `make filebrowser` and open the job's URL on your mobile device or desktop. Through a simple file explorer interface, you can upload test images and perform file operations.

### Uploading via CLI

On your local machine, run `make upload-data`. This command pushes local files from `./data` into `storage:ml-recipe-object-detection/data` and mounts them to your development environment's `/project/data`.

## Customization

Several variables in `Makefile` are intended to be modified according to the project’s specifics. To change them, find the corresponding line in `Makefile` and update it.

### Data location

`DATA_DIR_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_DIR)`

This project template implies that your data is stored alongside the project. If this is the case, you don't need to change this variable. However, if your data is shared between several projects on the platform, you will need to change the following line to point to its location. For example:

`DATA_DIR_STORAGE?=storage:datasets/cifar10`

### Training machine type

`TRAINING_MACHINE_TYPE?=gpu-small`

There are several machine types supported on the platform. Run `neuro config show` to see the list.

### HTTP authentication

`HTTP_AUTH?=--http-auth`

When jobs with HTTP interface are executed (for example, with Jupyter Notebooks or TensorBoard), this interface requires that the user be authenticated on the platform. However, if you want to share the link with someone who is not registered on the platform, you may disable the authentication requirement by updating this line to `HTTP_AUTH?=--no-http-auth`.

### Training command

`TRAINING_COMMAND?='echo "Replace this placeholder with a training script execution"'`

If you want to train some models from code instead of from Jupyter Notebook, you need to update this line. For example:

`TRAINING_COMMAND="bash -c 'cd $(PROJECT_PATH_ENV) && python -u $(CODE_DIR)/train.py --data $(DATA_DIR)'"`
