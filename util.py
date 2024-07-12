# import h5py
import json
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py
import os
from logger import create_logger
import datetime


def initialize_exp(path, name):
	# """
	# Experiment initialization.
	# """
	# # dump parameters
	# params.dump_path = get_dump_path(params)
	# pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

	# create a logger
	time_stamp = datetime.datetime.now()

	time = time_stamp.strftime('%Y%m%d%H%M%S')

	logger = create_logger(os.path.join(path, name + '_' + time + '.log'))
	print('log_name:',name + '_' + time + '.log')
	# logger = create_logger(os.path.join(path, name +'.log'))
	logger.info('============ Initialized logger ============')
	# logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
	#                       in sorted(dict(vars(params)).items())))
	return logger


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def map_label(label, classes):
	mapped_label = torch.LongTensor(label.size())
	for i in range(len(classes)):
		mapped_label[label == classes[i]] = i

	return mapped_label


class Logger(object):
	def __init__(self, filename):
		self.filename = filename
		f = open(self.filename + '.log', "a")
		f.close()

	def write(self, message):
		f = open(self.filename + '.log', "a")
		f.write(message)
		f.close()
  
  
class DATA_LOADER(object):
	def __init__(self, opt):
		if opt.matdataset:
			if opt.dataset == 'imagenet':
				self.read_matimagenet(opt)
			elif opt.dataset == 'ZDFY':
				self.read_turmor(opt);
			else:
				self.read_matdataset(opt)
		self.index_in_epoch = 0
		self.epochs_completed = 0
  
	def read_turmor(self, opt):
		self.train_class = [1, 2]
        
		# 加载训练特征和标签并转换为 Tensor
		self.train_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_features.npy'))  # (606, 2048)
		self.train_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_targets.npy'))
		self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32)
		self.train_label = torch.tensor(self.train_label, dtype=torch.long)

		self.ntrain = self.train_feature.shape[0]
		self.ntrain_class = 2
		self.ntest_class = 1

		# 加载测试特征和标签并转换为 Tensor
		self.test_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_features.npy'))  # (171, 2048)
		self.test_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_targets.npy'))
		self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32)
		self.test_label = torch.tensor(self.test_label, dtype=torch.long)

		# 加载属性嵌入并转换为 Tensor
		file_path = os.path.join(opt.dataroot, 'att', 'embeddings.json')
		with open(file_path, 'r') as f:
			data = json.load(f)

		attribute = {}
		for key, value in data.items():
			attribute[key] = np.array(value)

		categories = list(attribute.keys())
		embedding_list = [attribute[category] for category in categories]
		self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32)

		self.allclasses = [0, 1, 2]
		self.seenclasses = [1, 2]
		self.unseenclasses = [0]
		self.attribute_seen = self.attribute[self.seenclasses, :]

		# 提取标签为 1 和 2 的测试数据
		indices_seen = (self.test_label == 1) | (self.test_label == 2)
		self.test_seen_feature = self.test_feature[indices_seen]
		self.test_seen_label = self.test_label[indices_seen]

		# 提取标签为 0 的测试数据
		indices_unseen = self.test_label == 0
		self.test_unseen_feature = self.test_feature[indices_unseen]
		self.test_unseen_label = self.test_label[indices_unseen]

		# 计算每个类别的样本数量
		self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])

		# 打印结果以确认
		print("训练集特征形状:", self.train_feature.shape)
		print("训练集标签形状:", self.train_label.shape)
		print("测试集特征形状:", self.test_feature.shape)
		print("测试集标签形状:", self.test_label.shape)
		print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
		print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
		print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
		print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
		print("每个类别的样本数量:", self.train_samples_class_index)
     

	def next_batch_one_class(self, batch_size):
		if self.index_in_epoch == self.ntrain_class:
			self.index_in_epoch = 0
			perm = torch.randperm(self.ntrain_class)
			self.train_class[perm] = self.train_class[perm]

		iclass = self.train_class[self.index_in_epoch]
		idx = self.train_label.eq(iclass).nonzero().squeeze()
		perm = torch.randperm(idx.size(0))
		idx = idx[perm]
		iclass_feature = self.train_feature[idx]
		iclass_label = self.train_label[idx]
		self.index_in_epoch += 1
		return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

	def next_batch(self, batch_size):
		idx = torch.randperm(self.ntrain)[0:batch_size]
		batch_feature = self.train_feature[idx]
		batch_label = self.train_label[idx]
		batch_att = self.attribute[batch_label]
		return batch_feature, batch_label, batch_att

	# select batch samples by randomly drawing batch_size classes
	def next_batch_uniform_class(self, batch_size):
		batch_class = torch.LongTensor(batch_size)
		for i in range(batch_size):
			idx = torch.randperm(self.ntrain_class)[0]
			batch_class[i] = self.train_class[idx]

		batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
		batch_label = torch.LongTensor(batch_size)
		batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
		for i in range(batch_size):
			iclass = batch_class[i]
			idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
			idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
			idx_file = idx_iclass[idx_in_iclass]
			batch_feature[i] = self.train_feature[idx_file]
			batch_label[i] = self.train_label[idx_file]
			batch_att[i] = self.attribute[batch_label[i]]
		return batch_feature, batch_label, batch_att


def save_models(epoch, netG, netD, netMap, F_ha, optimizerG, optimizerD, opt, model_path, kind="seen"):
    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'netMap_state_dict': netMap.state_dict(),
        'F_ha_state_dict': F_ha.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'opt': opt,
    }, os.path.join(model_path, f'model_epoch_{epoch}_{kind}.pth'))
    
    
def load_models(epoch, netG, netD, netMap, F_ha, optimizerG, optimizerD, model_path, kind):
    checkpoint = torch.load(os.path.join(model_path, f'model_epoch_{epoch}_{kind}.pth'))
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    netMap.load_state_dict(checkpoint['netMap_state_dict'])
    F_ha.load_state_dict(checkpoint['F_ha_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    epoch = checkpoint['epoch']
    opt = checkpoint['opt']
    return epoch, opt