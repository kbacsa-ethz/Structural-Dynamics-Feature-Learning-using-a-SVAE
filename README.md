# Structural Dynamics Feature Learning using a SVAE

Repository to reproduce results of Structural Dynamics Feature Learning using a SVAE.

## Setup

Install python requirement:

```pip install -r requirements.txt```

## Create dataset

Create dataset (modify parameters of dataset accordingly in script):

```python simulate_structure.py```

## Train model

Train model using main script:

```python main.py```

All experiments will be saved in the ```experiments``` folder. To restore training, use the resume option:

```python main.py --resume name_of_checkpoint```
