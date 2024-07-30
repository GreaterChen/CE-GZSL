import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys

class CLASSIFIER:
    def __init__(self, _train_X, _train_Y, map_net, embed_size, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, cur_epoch=10):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.MapNet = map_net
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.cur_epoch = cur_epoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()

        self.input = torch.FloatTensor(_batch_size, _train_X.size(1))
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _ = self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()

            acc_seen, probs_seen, preds_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, probs_unseen, preds_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if (acc_seen + acc_unseen) == 0:
                print('a bug')
                H = 0
            else:
                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                # Save predictions and probabilities
                seen_df = pd.DataFrame(probs_seen, columns=[f'class_{i}' for i in range(probs_seen.shape[1])])
                seen_df['true_label'] = self.test_seen_label.numpy()
                seen_df['predicted_label'] = preds_seen

                unseen_df = pd.DataFrame(probs_unseen, columns=[f'class_{i}' for i in range(probs_unseen.shape[1])])
                unseen_df['true_label'] = self.test_unseen_label.numpy()
                unseen_df['predicted_label'] = preds_unseen
                all_df = pd.concat([seen_df, unseen_df])
        
        if best_H > 0:
            filename = f'/home/LAB/chenlb24/compare_model/CE-GZSL/models/testADNI/results_acc_{best_seen:.4f}_unseen_acc_{best_unseen:.4f}_h_{best_H:.4f}_epoch_{self.cur_epoch}.csv'
            all_df.to_csv(filename, index=False)

        return best_seen, best_unseen, best_H

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_probs = []
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output = self.model(embed)
            probs = torch.exp(output)  # Convert log probabilities to probabilities
            all_probs.append(probs.cpu().numpy())
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        all_probs = np.concatenate(all_probs, axis=0)
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, all_probs, predicted_label.numpy()

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= len(target_classes)
        return acc_per_class

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
