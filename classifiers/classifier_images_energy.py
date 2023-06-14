import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.special import logsumexp
import datasets.image_util as util
from sklearn.preprocessing import MinMaxScaler
import sys
import copy
import pdb

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netDec=None, dec_size=4096, dec_hidden_size=4096, usesoftmax = True, test_openset = True, temperature = 1, m_in=-23, m_out=-5, gamma1  = 0.1):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label
        if test_openset:
            self.test_unknown_feature = data_loader.test_unknown_feature.clone()   
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        if test_openset:
            self.unknownclasses = data_loader.unknownclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netDec = netDec
        self.usesoftmax = usesoftmax
        self.test_openset = test_openset
        self.ROC = None
        self.Precision = None
        self.Recall = None
        self.m_in = m_in
        self.m_out = m_out
        self.temperature = temperature
        self.gamma1 = gamma1
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            if test_openset:
                self.test_unknown_feature = self.compute_dec_out(self.test_unknown_feature, self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(util.weights_init)
        # self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            if self.test_openset:
                self.acc_seen, self.acc_unseen, self.H, self.epoch, self.best_model, self.auroc, self.fpr95 = self.fit()
            else:
                self.acc_seen, self.acc_unseen, self.H, self.epoch, self.best_model= self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            if self.test_openset:
                self.acc, self.auroc, self.best_model, self.fpr95 = self.fit_zsl() 
            else:
                self.acc, _, self.best_model = self.fit_zsl()
            #print('acc=%.4f' % (self.acc))
    def fit_zsl(self):
        best_acc = 0
        best_auroc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_model = copy.deepcopy(self.model.state_dict())
        train_set = torch.utils.data.TensorDataset(self.train_X, self.train_Y)
        train_loader = DataLoader(dataset=train_set, batch_size=int(self.batch_size), shuffle=True)
        for epoch in range(self.nepoch):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()  
                self.model.zero_grad()
                # data = in_set[0]
                output, logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                # mean_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            if self.test_openset:
                acc, auroc, ROC, Precision, Recall, fpr95 = self.val_auroc(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, self.test_unknown_feature)
                if acc > best_acc:
                    best_acc = acc
                    best_auroc = auroc
                    best_fpr95 = fpr95
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.ROC = ROC
                    self.Precision = Precision
                    self.Recall = Recall
            #print('acc %.4f' % (acc))
            else:
                acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(self.model.state_dict())
        if self.test_openset:
            return best_acc, best_auroc, best_model, best_fpr95
        else:
            return best_acc, None, best_model 
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_auroc = 0
        addd = 0
        best_ROC = None
        best_model = copy.deepcopy(self.model.state_dict())
        known_classes = torch.cat((self.seenclasses, self.unseenclasses), 0)
        train_set = torch.utils.data.TensorDataset(self.train_X, self.train_Y)
        train_loader = DataLoader(dataset=train_set, batch_size=int(self.batch_size), shuffle=True)
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()      
                self.model.zero_grad()
                output, logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
            
            if self.test_openset:
                acc_seen, acc_unseen, H, auroc, ROC, fpr95 = self.val_auroc_gzsl(self.test_seen_feature, util.map_label(self.test_seen_label, known_classes), \
                                                                    self.test_unseen_feature, util.map_label(self.test_unseen_label, known_classes),  \
                                                                    util.map_label(self.seenclasses, known_classes),util.map_label(self.unseenclasses, known_classes), \
                                                                    self.test_unknown_feature)
                if H > best_H:
                    best_H = H
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_auroc = auroc
                    best_ROC = ROC
                    best_fpr95 = fpr95
                    best_model = copy.deepcopy(self.model.state_dict())
            else:                
                acc_seen = self.val_gzsl(self.test_seen_feature, util.map_label(self.test_seen_label, known_classes), util.map_label(self.seenclasses, known_classes))
                acc_unseen = self.val_gzsl(self.test_unseen_feature, util.map_label(self.test_unseen_label, known_classes), util.map_label(self.unseenclasses, known_classes))
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
                if H > best_H:
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_H = H
                    best_model = copy.deepcopy(self.model.state_dict())
        if self.test_openset:
            self.ROC = ROC
            return best_seen, best_unseen, best_H, epoch, best_model, best_auroc, best_fpr95
        else:
            return best_seen, best_unseen, best_H, epoch, best_model
    def criterion(self, logits, target):
        loss = 0
        in_len = target.size(0)
        loss += F.cross_entropy(logits[:in_len], target)
        # Ec_out = -torch.logsumexp(logits[in_len:], dim=1)
        # Ec_in = -torch.logsumexp(logits[:in_len], dim=1)
        # Ec_out = -1 * self.temperature * torch.logsumexp(logits[in_len:] / self.temperature, axis=1)
        # Ec_in = -1 * self.temperature * torch.logsumexp(logits[:in_len] / self.temperature, axis=1)
        # loss += self.gamma1*(torch.pow(F.relu(Ec_in-self.m_in), 2).mean() + torch.pow(F.relu(self.m_out - Ec_out), 2).mean())
        return loss        

    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            output, _ = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        # acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        nclass = target_classes.size(0)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            output, _ = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data[:, :nclass], 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() 
    def val_auroc_gzsl(self, test_seen_feature, test_seen_label, test_unseen_feature, test_unseen_label, seenclasses, unseenclasses, test_unknown_feature):
        ntest_seen = test_seen_feature.size()[0]
        ntest_unseen = test_unseen_feature.size()[0]
        ntest_u = test_unknown_feature.size()[0]
        predicted_seen = torch.LongTensor(test_seen_label.size())
        predicted_unseen = torch.LongTensor(test_unseen_label.size())
        nclass_seen = seenclasses.size(0)
        nclass_unseen = unseenclasses.size(0)
        # TP = 0
        # FP = 0
        # TP_FN = test_unknown_feature.size()[0]
        # Precision = 0
        # Recall = 0
        pred_k, pred_u, labels = [], [], []
        start = 0
        for i in range(0, ntest_seen, self.batch_size):
            end = min(ntest_seen, start+self.batch_size)
            inputX = Variable(test_seen_feature[start:end].cuda())
            output, logits = self.model(inputX)
            if self.usesoftmax:
                logits = torch.nn.Softmax(dim=-1)(logits)
            pred_k.extend(logits.data.cpu().numpy())
            _, predicted_seen[start:end] = torch.max(output.data[:, :nclass_seen+nclass_unseen], 1)
            start = end
        start = 0
        for i in range(0, ntest_unseen, self.batch_size):
            end = min(ntest_unseen, start+self.batch_size)
            inputX = Variable(test_unseen_feature[start:end].cuda())
            output, logits = self.model(inputX)
            if self.usesoftmax:
                logits = torch.nn.Softmax(dim=-1)(logits)
            pred_k.extend(logits.data.cpu().numpy())
            _, predicted_unseen[start:end] = torch.max(output.data[:, :nclass_seen+nclass_unseen], 1)
            start = end
        start = 0
        for i in range(0, ntest_u, self.batch_size):
            end = min(ntest_u, start+self.batch_size)
            inputX = Variable(test_unknown_feature[start:end].cuda())
            output, logits = self.model(inputX)
            if self.usesoftmax:
                logits = torch.nn.Softmax(dim=-1)(logits)
            pred_u.extend(logits.data.cpu().numpy())
        acc_seen = self.compute_per_class_acc_gzsl(test_seen_label, predicted_seen, seenclasses)   
        acc_unseen = self.compute_per_class_acc_gzsl(test_unseen_label, predicted_unseen, unseenclasses)
        H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)     
        pred_k = np.array(pred_k)
        pred_u = np.array(pred_u)
        labels = torch.cat((test_seen_label, test_unseen_label), 0).data.cpu().numpy()
        auroc , ROC = self.compute_oscr(pred_k, pred_u, labels, nclass_seen+nclass_unseen)
        fpr95 = self.compute_fpr(pred_k, pred_u, labels, nclass_seen+nclass_unseen)
        return acc_seen, acc_unseen, H, auroc, ROC, fpr95
    def val_auroc(self, test_unseen_feature, test_unseen_label, unseenclasses, test_unknown_feature):
        start = 0
        TP = 0
        FP = 0
        TP_FN = test_unknown_feature.size()[0]
        Precision = 0
        Recall = 0
        ntest = test_unseen_feature.size()[0]
        ntest_u = test_unknown_feature.size()[0]
        predicted_known = torch.LongTensor(test_unseen_label.size())
        nclass = unseenclasses.size(0)
        pred_k,pred_u,labels = [],[],[]
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_unseen_feature[start:end].cuda())
            else:
                inputX = Variable(test_unseen_feature[start:end])
            output, logits = self.model(inputX) 
            if self.usesoftmax:
                logits = torch.nn.Softmax(dim=-1)(logits)
            # logits = logits.data[:, :nclass]    
            pred_k.extend(logits.data.cpu().numpy())
            _, predicted_known[start:end] = torch.max(output.data[:, :nclass], 1)
            start = end
        start = 0
        for i in range(0, ntest_u, self.batch_size):
            end = min(ntest_u, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_unknown_feature[start:end].cuda())
            else:
                inputX = Variable(test_unknown_feature[start:end])
            output, logits = self.model(inputX)
            if self.usesoftmax:
                logits = torch.nn.Softmax(dim=-1)(logits)
            pred_u.extend(logits.data.cpu().numpy())
            
            
        acc = self.compute_per_class_acc(util.map_label(test_unseen_label, unseenclasses), predicted_known, unseenclasses.size(0))
        if(TP + FP > 0):
            Precision = TP / (TP + FP)
            Recall = TP / TP_FN
        pred_k = np.array(pred_k)
        pred_u = np.array(pred_u)
        labels = np.array(util.map_label(test_unseen_label, unseenclasses))
        auroc , ROC = self.compute_oscr(pred_k, pred_u, labels, unseenclasses.size(0))
        fpr95 = self.compute_fpr(pred_k, pred_u, labels, unseenclasses.size(0))
        return acc, auroc, ROC, Precision, Recall, fpr95
    
    def compute_oscr(self, pred_k, pred_u, labels, nclasses):
        if pred_k.shape[1] == nclasses + 1:
            x1, x2 = pred_k[:, -1],  pred_u[:, -1]
        else:
            x1 = -1 * self.temperature * logsumexp(pred_k / self.temperature, axis=1)
            x2 = -1 * self.temperature * logsumexp(pred_u / self.temperature, axis=1)
            # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
        self.known_score = x1
        self.unknown_score = x2
        k_target = np.concatenate((np.ones(len(x1)), np.zeros(len(x2))), axis=0)
        u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
        predict = np.concatenate((x1, x2), axis=0)
        n = len(predict)

        # Cutoffs are of prediction values
        
        CCR = [0 for x in range(n+2)]
        FPR = [0 for x in range(n+2)] 

        auxa = np.random.random(predict.size)
        idx = np.lexsort((auxa, predict))
        # idx = predict.argsort()

        s_k_target = k_target[idx]
        s_u_target = u_target[idx]
        for k in range(n-1):
            CC = s_u_target[k+1:].sum()
            FP = s_k_target[k:].sum()

            # True	Positive Rate
            CCR[k] = float(CC) / float(len(x2))
            # False Positive Rate
            FPR[k] = float(FP) / float(len(x1))

        CCR[n] = 0.0
        FPR[n] = 0.0
        CCR[n+1] = 1.0
        FPR[n+1] = 1.0

        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(FPR, CCR), reverse=True)

        OSCR = 0

        # Compute AUROC Using Trapezoidal Rule
        for j in range(n+1):
            h =   ROC[j][0] - ROC[j+1][0]
            w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

            OSCR = OSCR + h*w

        return OSCR, ROC

    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            # if self.cuda:
            #     inputX = Variable(test_X[start:end].cuda())
            # else:
            inputX = Variable(test_X[start:end]).cuda()
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X
    def compute_fpr(self, pred_k, pred_u, labels, nclasses):
        if pred_k.shape[1] == nclasses + 1:
            x1, x2 = pred_k[:, -1],  pred_u[:, -1]
            pred = np.argmax(pred_k[:, :-1], axis=1)
        else:
            x1, x2 = -1 * np.max(pred_k, axis=1), -1 * np.max(pred_u, axis=1)
            pred = np.argmax(pred_k, axis=1)
        pos = np.array(x2[:]).reshape((-1, 1))
        neg = np.array(x1[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1
        
        fpr95 = self.fpr_and_fdr_at_recall(labels, examples)
        
        
        # fpr,tpr,thresh = roc_curve(labels, examples, pos_label=1)
        # fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        return fpr95
        
    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                            'its last element does not correspond to sum')
        return out
    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=0.95, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                        np.array_equal(classes, [-1, 1]) or
                        np.array_equal(classes, [0]) or
                        np.array_equal(classes, [-1]) or
                        np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)      # [last_ind::-1]
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        cutoff = np.argmin(np.abs(recall - recall_level))

        return fps[cutoff] / (np.sum(np.logical_not(y_true))) 

class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        logits = self.fc(x)
        o = self.logic(logits)
        return o, logits
