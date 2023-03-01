# MLink

code for paper "MLink: Linking Black-box Models for Collaborative Multi-model Inference"

## Installation

Create a virtual environment and install python packages using pip:
```bash
conda create --name mlink python=3.9
conda activate mlink
pip install tensorflow-federated
# tensorflow_federated-0.50.0, 2023/3/1

# test tensorflow federated (https://github.com/tensorflow/federated/blob/main/docs/install.md)
python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"
# b'Hello World'
```

To run jupyter notebooks:
```bash
pip install ipykernel
ipython kernel install --user --name=mlink
jupyter-lab
```