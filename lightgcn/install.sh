#!/bin/bash
pip install pandas
pip install sklearn
pip uninstall torch, torch-sparse, torch-scatter, torch-geometric
pip3 install torch==1.10.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-geometric
