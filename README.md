## Probabilistic Federated Neural Matching


This is the code accompanying the ICML 2019 paper "Bayesian Nonparametric Federated Learning of Neural Networks"

##### Requirements to run the code:
---
1. Python 3.6
2. PyTorch 0.4
3. Scikit-learn
4. Matplotlib
5. Numpy


##### Important source files:
---

1. `experiment.py`: Main entryway to the code. Used to run all experiments
2. `matching/pfnm.py`: Contains the PFNM matching code for single communication federated learning
3. `matching/pfnm_communication.py`: Contains the PFNM matching code for multiple communication federated learning


##### Sample Commands:
---

1. MNIST Heterogenous 10 batches
    ```python experiment.py --logdir "logs/mnist_test" --dataset "mnist" --datadir "data/mnist/" --net_config "784, 100, 10" --n_nets 10 --partition "hetero-dir" --experiment "u-ensemble,pdm,pdm_iterative" --lr 0.01 --epochs 10 --reg 1e-6 --communication_rounds 5 --lr_decay 0.99 --iter_epochs 5```

2. CIFAR-10 Heterogenous 10 batches
    `python experiment.py --logdir "logs/cifar10_test" --dataset "cifar10" --datadir "data/cifar10/" --net_config "3072, 100, 10" --n_nets 10 --partition "hetero-dir" --experiment "u-ensemble,pdm,pdm_iterative" --lr 0.001 --epochs 10 --reg 1e-5 --communication_rounds 5 --lr_decay 0.99 --iter_epochs 5`


##### Important arguments:
---

The following arguments to the PFNM file control the important parameters of the experiment

1. `net_config`: Defines the local network architecture. CSV of sizes. Ex: "784, 100, 100, 10" defines a 2-layer network with 100 neurons in each layer.
2. `n_nets`: Number of local networks. This is denoted by "J" in the paper
3. `partition`: Kind of data partition. Values: homo, hetero-dir
4. `experiments`: Defines which experiments will be executed. Values:  u-ensemble (Uniform ensemble), pdm (PFNM matching), pdm_iterative (PFNM with extra communications)
5. `communication_rounds`: How many rounds of communication between the local learner and the master network in the case of PFNM with multiple communications.


##### Output:
---

Some of the output is printed on the terminal. However, majority of the information is logged to a log file in the specified log folder.
