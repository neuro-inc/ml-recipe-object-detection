# Object Detection

This recipe has code and procedures necessary to add new object classes into object detection model. Adding new classes in object recognition is computationally expensive and involves model fine-tuning to ensure it still performs well on old and new classes. Usually, it's a trade-off between cost of labeling new datasets (new objects in different environments and combinations), model training time (cost of GPU), and ML Engineering time spent on experimentation and development. To make this example more interactive we are providing a reduced version of [Common Objects in Context (COCO)](http://cocodataset.org) dataset that contains only classes that represent retail items. This dataset is further reduced to 100 images to reduce re-training time and make this example more interactive.

# Quick Start

##### Sign up at [neu.ro](https://neu.ro).
##### Install CLI and log in:
```shell
pip install -U neuromation
neuro login
```
##### Run the recipe:
```shell
git clone git@github.com:neuromation/ml-recipe-object-detection.git
cd ml-recipe-object-detection
make setup
make jupyter
```

# Development Environment

This project is designed to run on [Neuro Platform](https://neu.ro), so you can jump into problem-solving right away.

## Directory structure

| Local directory                      | Description       | Storage URI                                                                  | Environment mounting point |
|:------------------------------------ |:----------------- |:---------------------------------------------------------------------------- |:-------------------------- | 
| `data/`                              | Data              | `storage:ml-recipe-object-detection/data/`                              | `/ml-recipe-object-detection/data/` | 
| `modules/` | Python modules    | `storage:ml-recipe-object-detection/modules/` | `/ml-recipe-object-detection/modules/` |
| `notebooks/`                         | Jupyter notebooks | `storage:ml-recipe-object-detection/notebooks/`                         | `/ml-recipe-object-detection/notebooks/` |
| No directory                         | Logs and results  | `storage:ml-recipe-object-detection/results/`                           | `/ml-recipe-object-detection/results/` |

## Development

Follow the instructions below to set up the environment and start Jupyter development session.

### Setup development environment 

`make setup`

* Several files from the local project are uploaded to the platform storage (namely, `requirements.txt`, 
  `apt.txt`, `setup.cfg`).
* A new job is started in our [base environment](https://hub.docker.com/r/neuromation/base). 
* Pip requirements from `requirements.txt` and apt applications from `apt.txt` are installed in this environment.
* The updated environment is saved under a new project-dependent name and is used further on.

### Run Jupyter with GPU 

`make jupyter`

* The content of `modules` and `notebooks` directories is uploaded to the platform storage.
* A job with Jupyter is started, and its web interface is opened in the local web browser window.

### Kill Jupyter

`make kill-jupyter`

* The job with Jupyter Notebooks is terminated. The notebooks are saved on the platform storage. You may run 
  `make download-notebooks` to download them to the local `notebooks/` directory.

### Help

`make help`

## Data

### Uploading via Web UI

On local machine run `make filebrowser` and open job's URL on your mobile device or desktop.
Through a simple file explorer interface, you can upload test images and perform file operations.

### Uploading via CLI

On local machine run `make upload-data`. This command pushes local files stored in `./data`
into `storage:ml-recipe-object-detection/data` mounted to your development environment's `/project/data`.

## Customization

Several variables in `Makefile` are intended to be modified according to the project specifics. 
To change them, find the corresponding line in `Makefile` and update.

### Data location

`DATA_DIR_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_DIR)`

This project template implies that your data is stored alongside the project. If this is the case, you don't 
have to change this variable. However, if your data is shared between several projects on the platform, 
you need to change the following line to point to its location. For example:

`DATA_DIR_STORAGE?=storage:datasets/cifar10`

### Training machine type

`TRAINING_MACHINE_TYPE?=gpu-small`

There are several machine types supported on the platform. Run `neuro config show` to see the list.

### HTTP authentication

`HTTP_AUTH?=--http-auth`

When jobs with HTTP interface are executed (for example, with Jupyter Notebooks or TensorBoard), this interface requires
a user to be authenticated on the platform. However, if you want to share the link with someone who is not registered on
the platform, you may disable the authentication updating this line to `HTTP_AUTH?=--no-http-auth`.

### Training command

`TRAINING_COMMAND?='echo "Replace this placeholder with a training script execution"'`

If you want to train some models from code instead of Jupyter Notebooks, you need to update this line. For example:

`TRAINING_COMMAND="bash -c 'cd $(PROJECT_PATH_ENV) && python -u $(CODE_DIR)/train.py --data $(DATA_DIR)'"`

