import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import copy
from sklearn.metrics import confusion_matrix

from model import FcNet
from datasets import MNIST_truncated, CIFAR10_truncated

from combine_nets import compute_ensemble_accuracy, compute_pdm_matching_multilayer, compute_iterative_pdm_matching

def mkdirs(dirpath):
	try:
		os.makedirs(dirpath)
	except Exception as _:
		pass

def get_parser():

	parser = argparse.ArgumentParser()

	parser.add_argument('--logdir', type=str, required=True, help='Log directory path')
	parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
	parser.add_argument('--dataset', type=str, required=True, help="Dataset [mnist/cifar10]")
	parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
	parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

	parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

	parser.add_argument('--n_nets', type=int , required=True, help="Number of nets to initialize")
	parser.add_argument('--partition', type=str, required=True, help="Partition = homo/hetero/hetero-dir")
	parser.add_argument('--experiment', required=True, type=lambda s: s.split(','), help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
	parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
	
	parser.add_argument('--lr', type=float, required=True, help="Learning rate")
	parser.add_argument('--epochs', type=int, required=True, help="Epochs")
	parser.add_argument('--reg', type=float, required=True, help="L2 regularization strength")

	parser.add_argument('--alpha', type=float, required=False, default=0.5, help="Dirichlet distribution constant used for data partitioning")

	parser.add_argument('--communication_rounds', type=int, required=False, default=None, help="How many iterations of PDM matching should be done")
	parser.add_argument('--lr_decay', type=float, required=False, default=1.0, help="Decay LR after every PDM iterative communication")
	parser.add_argument('--iter_epochs', type=int, required=False, default=None, help="Epochs for PDM-iterative method")
	parser.add_argument('--reg_fac', type=float, required=False, default=0.0, help="Regularization factor for PDM Iter")

	parser.add_argument('--pdm_sig', type=float, required=False, default=1.0, help="PDM sigma param")
	parser.add_argument('--pdm_sig0', type=float, required=False, default=1.0, help="PDM sigma0 param")
	parser.add_argument('--pdm_gamma', type=float, required=False, default=1.0, help="PDM gamma param")

	return parser

def load_mnist_data(datadir):

	transform = transforms.Compose([transforms.ToTensor()])

	mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
	mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

	X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
	X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

	X_train = X_train.data.numpy()
	y_train = y_train.data.numpy()
	X_test = X_test.data.numpy()
	y_test = y_test.data.numpy()

	return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

	transform = transforms.Compose([transforms.ToTensor()])

	cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
	cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

	X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
	X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

	return (X_train, y_train, X_test, y_test)


def parse_class_dist(net_class_config):

	cls_net_map = {}

	for net_idx, net_classes in enumerate(net_class_config):
		for net_cls in net_classes:
			if net_cls not in cls_net_map:
				cls_net_map[net_cls] = []
			cls_net_map[net_cls].append(net_idx)

	return cls_net_map

def record_net_data_stats(y_train, net_dataidx_map, logdir):

	net_cls_counts = {}

	for net_i, dataidx in net_dataidx_map.items():
		unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
		tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
		net_cls_counts[net_i] = tmp

	logging.debug('Data statistics: %s' % str(net_cls_counts))

	return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_nets, alpha=0.5):

	if dataset == 'mnist':
		X_train, y_train, X_test, y_test = load_mnist_data(datadir)
	elif dataset == 'cifar10':
		X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

	n_train = X_train.shape[0]

	if partition == "homo":
		idxs = np.random.permutation(n_train)
		batch_idxs = np.array_split(idxs, n_nets)
		net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

	elif partition == "hetero-dir":
		min_size = 0
		K = 10
		N = y_train.shape[0]
		net_dataidx_map = {}

		while min_size < 10:
			idx_batch = [[] for _ in range(n_nets)]
			for k in range(K):
				idx_k = np.where(y_train == k)[0]
				np.random.shuffle(idx_k)
				proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
				## Balance
				proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
				proportions = proportions/proportions.sum()
				proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
				idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
				min_size = min([len(idx_j) for idx_j in idx_batch])

		for j in range(n_nets):
			np.random.shuffle(idx_batch[j])
			net_dataidx_map[j] = idx_batch[j]

	traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

	return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def init_nets(net_configs, dropout_p, n_nets):

	input_size = net_configs[0]
	output_size = net_configs[-1]
	hidden_sizes = net_configs[1:-1]

	nets = {net_i: None for net_i in range(n_nets)}

	for net_i in range(n_nets):
		net = FcNet(input_size, hidden_sizes, output_size, dropout_p)

		nets[net_i] = net

	return nets

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):

	if dataset == 'mnist':
		dl_obj = MNIST_truncated
	elif dataset == 'cifar10':
		dl_obj = CIFAR10_truncated

	transform = transforms.Compose([transforms.ToTensor()])

	train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform, download=True)
	test_ds = dl_obj(datadir, train=False, transform=transform, download=True)

	train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
	test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

	return train_dl, test_dl


def compute_accuracy(model, dataloader, get_confusion_matrix=False):

	was_training = False
	if model.training:
		model.eval()
		was_training = True

	true_labels_list, pred_labels_list = np.array([]), np.array([])

	correct, total = 0, 0
	with torch.no_grad():
		for batch_idx, (x, target) in enumerate(dataloader):
			out = model(x)
			_, pred_label = torch.max(out.data, 1)

			total += x.data.size()[0]
			correct += (pred_label == target.data).sum().item()

			pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
			true_labels_list = np.append(true_labels_list, target.data.numpy())

	if get_confusion_matrix:
		conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

	if was_training:
		model.train()

	if get_confusion_matrix:
		return correct/float(total), conf_matrix

	return correct/float(total)

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, reg, reg_base_weights=None):

	logging.debug('Training network %s' % str(net_id))
	logging.debug('n_training: %d' % len(train_dataloader))
	logging.debug('n_test: %d' % len(test_dataloader))

	train_acc = compute_accuracy(net, train_dataloader)
	test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True)

	logging.debug('>> Pre-Training Training accuracy: %f' % train_acc)
	logging.debug('>> Pre-Training Test accuracy: %f' % test_acc)

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0, amsgrad=True)		# L2_reg=0 because it's manually added later

	criterion = nn.CrossEntropyLoss()

	cnt = 0
	losses, running_losses = [], []

	for epoch in range(epochs):
		for batch_idx, (x, target) in enumerate(train_dataloader):

			l2_reg = torch.zeros(1)
			l2_reg.requires_grad = True

			optimizer.zero_grad()
			x.requires_grad = True
			target.requires_grad = False
			target = target.long()

			out = net(x)
			loss = criterion(out, target)

			if reg_base_weights is None:
				# Apply standard L2-regularization
				for param in net.parameters():
					l2_reg = l2_reg + 0.5 * torch.pow(param, 2).sum()
			else:
				# Apply Iterative PDM regularization
				for pname, param in net.named_parameters():
					if "bias" in pname:
						continue

					layer_i = int(pname.split('.')[1])

					if pname.split('.')[2] == "weight":
						weight_i = layer_i * 2
						transpose = True

					ref_param = reg_base_weights[weight_i]
					ref_param = ref_param.T if transpose else ref_param

					l2_reg = l2_reg + 0.5 * torch.pow((param - torch.from_numpy(ref_param).float()), 2).sum()

			loss = loss + reg * l2_reg

			loss.backward()
			optimizer.step()

			cnt += 1
			losses.append(loss.item())

		logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))

	train_acc = compute_accuracy(net, train_dataloader)
	test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True)

	logging.debug('>> Training accuracy: %f' % train_acc)
	logging.debug('>> Test accuracy: %f' % test_acc)

	logging.debug(' ** Training complete **')

	return train_acc, test_acc


def load_new_state(nets, new_weights):

	for netid, net in nets.items():

		statedict = net.state_dict()
		weights = new_weights[netid]

		# Load weight into the network
		i = 0
		layer_i = 0

		while i < len(weights):
			weight = weights[i]
			i += 1
			bias = weights[i]
			i += 1

			statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
			statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)
			layer_i += 1

		net.load_state_dict(statedict)

	return nets

def run_exp():

	parser = get_parser()
	args = parser.parse_args()

	mkdirs(args.logdir)
	with open(os.path.join(args.logdir, 'experiment_arguments.json'), 'w') as f:
		json.dump(str(args), f)

	logging.basicConfig(
		filename=os.path.join(args.logdir, 'experiment_log-%d-%d.log' % (args.init_seed, args.trials)),
		format='%(asctime)s %(levelname)-8s %(message)s',
		datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

	logging.debug("Experiment arguments: %s" % str(args))

	for trial in range(args.trials):

		seed = trial + args.init_seed

		print("Executing Trial %d " % trial)
		logging.debug("#" * 100)
		logging.debug("Executing Trial %d with seed %d" % (trial, seed))

		np.random.seed(seed)
		torch.manual_seed(seed)

		print("Partitioning data")
		X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
						args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)

		n_classes = len(np.unique(y_train))

		print("Initializing nets")
		nets = init_nets(args.net_config, args.dropout_p, args.n_nets)

		local_train_accs = []
		local_test_accs = []
		for net_id, net in nets.items():
			dataidxs = net_dataidx_map[net_id]
			print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

			train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
			trainacc, testacc = train_net(net_id, net, train_dl, test_dl, args.epochs, args.lr, args.reg)

			local_train_accs.append(trainacc)
			local_test_accs.append(testacc)

		train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)

		logging.debug("*"*50)
		logging.debug("Running experiments \n")

		nets_list = list(nets.values())

		if ("u-ensemble" in args.experiment) or ("all" in args.experiment):
			print("Computing Uniform ensemble accuracy")
			uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl, n_classes,  uniform_weights=True)
			uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl, n_classes, uniform_weights=True)

			logging.debug("Uniform ensemble (Train acc): %f" % uens_train_acc)
			logging.debug("Uniform ensemble (Test acc): %f" % uens_test_acc)

		if ("pdm" in args.experiment) or ("all" in args.experiment):
			print("Computing hungarian matching")
			best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res = compute_pdm_matching_multilayer(
				nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=5, sigma=args.pdm_sig, sigma0=args.pdm_sig0, gamma=args.pdm_gamma
			)

			logging.debug("****** PDM matching ******** ")
			logging.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
						  % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

			logging.debug("PDM log: %s " % str(res))

		if ("pdm_iterative" in args.experiment) or ("all" in args.experiment):
			print("Running Iterative PDM matching procedure")
			logging.debug("Running Iterative PDM matching procedure")

			sigma0s = [1.0]
			sigmas = [1.0]
			gammas = [1.0]

			for (sigma0, sigma, gamma) in product(sigma0s, sigmas, gammas):
				logging.debug("Parameter setting: sigma0 = %f, sigma = %f, gamma = %f" % (sigma0, sigma, gamma))

				iter_nets = copy.deepcopy(nets)
				assignment = None
				lr_iter = args.lr
				reg_iter = args.reg

				# Run for communication rounds iterations
				for i, comm_round in enumerate(range(args.communication_rounds)):

					it = 3

					iter_nets_list = list(iter_nets.values())

					net_weights_new, train_acc, test_acc, new_shape, assignment, hungarian_weights, \
					conf_matrix_train, conf_matrix_test = compute_iterative_pdm_matching(
						iter_nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1],
						sigma, sigma0, gamma, it, old_assignment=assignment
					)

					logging.debug("Communication: %d, Train acc: %f, Test acc: %f, Shapes: %s" % (comm_round, train_acc, test_acc, str(new_shape)))
					logging.debug('CENTRAL MODEL CONFUSION MATRIX')
					logging.debug('Train data confusion matrix: \n %s' % str(conf_matrix_train))
					logging.debug('Test data confusion matrix: \n %s' % str(conf_matrix_test))

					iter_nets = load_new_state(iter_nets, net_weights_new)

					expepochs = args.iter_epochs if args.iter_epochs is not None else args.epochs

					# Train these networks again
					for net_id, net in iter_nets.items():
						dataidxs = net_dataidx_map[net_id]
						print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

						net_train_dl, net_test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
						train_net(net_id, net, net_train_dl, net_test_dl, expepochs, lr_iter, reg_iter, net_weights_new[net_id])

					lr_iter *= args.lr_decay
					reg_iter *= args.reg_fac

		logging.debug("Trial %d completed" % trial)
		logging.debug("#"*100)

if __name__ == "__main__":
	run_exp()
