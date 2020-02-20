#!/usr/bin/env bash
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root     --NotebookApp.token= --notebook-dir=/project/notebooks             --NotebookApp.default_url=/project/notebooks/demo.ipynb
