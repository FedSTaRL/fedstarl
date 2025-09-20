# 📝 FedSTaRL: Leveraging Federated Learning for Decentralized Semi-Supervised Task-Informed Representation Learning on Sequential Data

This repository is the official implementation of [Leveraging Federated Learning for Decentralized Semi-Supervised Task-Informed Representation Learning on Sequential Data](). Check out the official [Project Page](https://fedstarl.github.io/) for more information.


**Note**: While we are unable to publish all relevant code for this project, we have provided a framework that facilitates federated learning experiments for sequential data. This framework, adapted from the [pfl-source-code](https://apple.github.io/pfl-research/), also leverages the [Hydra configuration package](https://hydra.cc/).
To utilize this repository, simply add your dataset and models, then customize the configurations found in `/benchmark`.

## 📋 Requirements

We use [Poetry](https://python-poetry.org/) to manage our environment. We have provided the necessary `poetry.lock` and `pyproject.toml` files for this code-base, allowing you to install all required dependencies and set up the virtual environment once you have installed `Poetry`.

### Setup Process:
1. To install Poetry, please follow the official [guidance](https://python-poetry.org/docs/#installation).

2. Navigate to the project directory:
```
cd path/to/your/project
```

3. Install dependencies:
```
poetry install
```

4. Initiate poetry shell (virtual environment managed by poetry):
```
poetry shell
```
This command will activate a virtual environment where you can run your project with the dependencies specified in the lock file.

#### Update PyTorch Version:
If you would like to use newer Pytorch version, such as [PyTorch-2.4.1](https://pytorch.org/get-started/previous-versions/#v241) for example, you can update it using the following command (for Linux & Cuda 118):
```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

### Setup Horovod - Install from Source

The [pfl](https://apple.github.io/pfl-research/) framework use [Horovod](https://horovod.ai/) to perform distributed, multi-gpu training. To default installation might fail, hence we provide a guide that has worked for us. This guide is for linux based OS.

**Install openmpi**:
- Download: [openmpi-4.1.7](https://www.open-mpi.org/software/ompi/v4.1/)<br>
- Documentation: [openmpi-4.1.7-documentation](https://www.open-mpi.org/faq/?category=building#easy-build)

**Install NCCL**:
- Download: [nccl2](https://developer.nvidia.com/nccl/nccl-download)<br>
- Documentation: [nccl2-documentation](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)

**Install Horovod (for PyTorch)**:

*Download Horovod*:
```
git clone https://github.com/horovod/horovod
```
*Install Horovod*: \
Ensure your are in your virtual environment. Then navigate to the just downloaded horovod directory.
```
cd horovod
```
*Execute installation procedure* (for linux):
```
HOROVOD_NCCL_HOME=<path_NCCL_HOME> HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_DEBUG=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_GLOO=1 pip install -v -e .
```
I.e., `<path_NCCL_HOME>` = '/usr/share/doc/libnccl2'
```
HOROVOD_NCCL_HOME=/usr/share/doc/libnccl2 HOROVOD_CMAK=/home/ubuntu/.cache/pypoetry/virtualenvs/<virtual-environment-name>/bin/cmake  HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_DEBUG=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_GLOO=1 pip install -v -e .
```
*Check if build was successful*:
```
horovodrun --check-build
```
If the build was successful, you should see the following output:

```
Horovod v0.28.1:

Available Frameworks:
    [ ] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [ ] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [ ] Gloo
```

## 🏋️‍♂️ Training
We provide the code for the RNN-based model in this version of the code. 
Set up a project directory with a `train.py` script and add the configurations for your dataset and your models in the `benchmark/configs` directory, for example. 

To train the model(s) in the paper, run this command. 
```
CUDA_VISIBLE_DEVICES=0 python benchmarks/<project>/train.py +experiment=<project>/<experiment_config>
```

To use `horovod` for distrubted, multi-gpu training, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1 horovodrun --gloo -np 2 -H localhost:2 python benchmarks/<project>/train.py +experiment=<project>/<experiment_config>
```

**Note**: We provide an example on how to use this repo. See `/benchmarks/image_classification`.

## 📈 Evaluation 
Automatically evaluate the central performance for the federated experiments if a central dataset is given (required).

## 📊 Results
See the [Project Page](https://fedstarl.github.io/) for more details.

## 🤝 Contributing
All contributions are welcome! All content in this repository is licensed under the MIT license.





