## Probabilistic Federated Neural Matching


This is the code accompanying the ICML 2019 paper "Bayesian Nonparametric Federated Learning of Neural Networks"
Paper link: [http://proceedings.mlr.press/v97/yurochkin19a.html]

#### Requirements to run the code:
---

1. Python 3.6
2. PyTorch 0.4
3. Scikit-learn
4. Matplotlib
5. Numpy


#### Important source files:
---

1. `experiment.py`: Main entryway to the code. Used to run all experiments
2. `matching/pfnm.py`: Contains the PFNM matching code for single communication federated learning
3. `matching/pfnm_communication.py`: Contains the PFNM matching code for multiple communication federated learning


#### Sample Commands:
---

1. MNIST Heterogenous 10 batches

`python experiment.py --logdir "logs/mnist_test" --dataset "mnist" --datadir "data/mnist/" --net_config "784, 100, 10" --n_nets 10 --partition "hetero-dir" --experiment "u-ensemble,pdm,pdm_iterative" --lr 0.01 --epochs 10 --reg 1e-6 --communication_rounds 5 --lr_decay 0.99 --iter_epochs 5`

2. CIFAR-10 Heterogenous 10 batches
    
`python experiment.py --logdir "logs/cifar10_test" --dataset "cifar10" --datadir "data/cifar10/" --net_config "3072, 100, 10" --n_nets 10 --partition "hetero-dir" --experiment "u-ensemble,pdm,pdm_iterative" --lr 0.001 --epochs 10 --reg 1e-5 --communication_rounds 5 --lr_decay 0.99 --iter_epochs 5`


#### Important arguments:
---


The following arguments to the PFNM file control the important parameters of the experiment

1. `net_config`: Defines the local network architecture. CSV of sizes. Ex: "784, 100, 100, 10" defines a 2-layer network with 100 neurons in each layer.
2. `n_nets`: Number of local networks. This is denoted by "J" in the paper
3. `partition`: Kind of data partition. Values: homo, hetero-dir
4. `experiments`: Defines which experiments will be executed. Values:  u-ensemble (Uniform ensemble), pdm (PFNM matching), pdm_iterative (PFNM with extra communications)
5. `communication_rounds`: How many rounds of communication between the local learner and the master network in the case of PFNM with multiple communications.


#### Output:
---

Some of the output is printed on the terminal. However, majority of the information is logged to a log file in the specified log folder.


### Citing PFNM
---

```
@InProceedings{pmlr-v97-yurochkin19a,
  title = 	 {{B}ayesian Nonparametric Federated Learning of Neural Networks},
  author = 	 {Yurochkin, Mikhail and Agarwal, Mayank and Ghosh, Soumya and Greenewald, Kristjan and Hoang, Nghia and Khazaeni, Yasaman},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {7252--7261},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/yurochkin19a.html},
  abstract = 	 {In federated learning problems, data is scattered across different servers and exchanging or pooling it is often impractical or prohibited. We develop a Bayesian nonparametric framework for federated learning with neural networks. Each data server is assumed to provide local neural network weights, which are modeled through our framework. We then develop an inference approach that allows us to synthesize a more expressive global network without additional supervision, data pooling and with as few as a single communication round. We then demonstrate the efficacy of our approach on federated learning problems simulated from two popular image classification datasets.}
}
```
