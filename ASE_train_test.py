
from __future__ import print_function
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import functions
import networks.TFVAEGAN_model as model
import datasets.image_util as util
import classifiers.classifier_images as classifier
from config_images import opt
import os, time
import numpy as np
from torch.nn import functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
# load data
data = util.DATA_LOADER(opt)

# load model
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)

netD.cuda()
netE.cuda()
netF.cuda()
netG.cuda()
netDec.cuda()


# load pretrained model
zsl_setting = ""
if opt.gzsl:
    zsl_setting = "gzsl"
else:
    zsl_setting = "zsl"
netG.load_state_dict(torch.load('{}/{}/{}/netG_best.pth'.format(opt.model_dir, opt.dataset, zsl_setting)))
netE.load_state_dict(torch.load('{}/{}/{}/netE_best.pth'.format(opt.model_dir, opt.dataset, zsl_setting)))
netD.load_state_dict(torch.load('{}/{}/{}/netD_best.pth'.format(opt.model_dir, opt.dataset, zsl_setting)))
netF.load_state_dict(torch.load('{}/{}/{}/netF_best.pth'.format(opt.model_dir, opt.dataset, zsl_setting)))
netDec.load_state_dict(torch.load('{}/{}/{}/netDec_best.pth'.format(opt.model_dir, opt.dataset, zsl_setting)))
netE.eval()
netG.eval()
netD.eval()
netF.eval()
netDec.eval()

if opt.gzsl:
    known_classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
else:
    known_classes = data.unseenclasses
index = torch.LongTensor(random.choices(range(known_classes.size(0)), 
                                       k = opt.count_ASE-known_classes.size(0)))
center_classes = torch.cat((known_classes, known_classes[index]), 0)
center_attr = data.attribute[center_classes]
center_label = util.map_label(center_classes, known_classes)
perm = torch.randperm(center_label.size(0))
center_attr = center_attr[perm]
center_label = center_label[perm]
index_in_epoch = 0
def next_batch(batch_size):
    global index_in_epoch, center_attr, center_label
    start = index_in_epoch
    if start + batch_size > center_attr.size(0):
        X_rest = center_attr[start:center_attr.size(0)]
        Y_rest = center_label[start:center_label.size(0)]
        perm = torch.randperm(center_label.size(0))
        center_attr = center_attr[perm]
        center_label = center_label[perm]
        index_in_epoch = batch_size - (center_attr.size(0) - start)
        X_new = center_attr[:index_in_epoch]
        Y_new = center_label[:index_in_epoch]
        if X_rest.size(0) > 0:
            return torch.cat((X_rest, X_new), 0), torch.cat((Y_rest, Y_new), 0)
        else:
            return X_new, Y_new
    else:
        index_in_epoch += batch_size
        return center_attr[start:start+batch_size], center_label[start:start+batch_size]
    
def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()

def compute_dec_out(netDec, test_X, new_size):
    start = 0
    ntest = test_X.size()[0]
    feat1 = netDec(test_X)
    feat2 = netDec.getLayersOutDet()
    new_test_X = torch.cat([test_X,feat1,feat2],dim=1)
    return new_test_X
# generate the unknown attribute vectors
index_in_epoch = 0
dec_size=opt.attSize
dec_hidden_size = 4096
nclass = known_classes.size(0)
unknown_attribute = None
input_dim = opt.resSize
if netDec:
    input_dim += dec_hidden_size + dec_size
def generate_syn_feature(generator, classes, attribute, num, netF=None, netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise)
        syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label
unseen_feature, unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, netF=netF,  netDec=netDec)
if opt.gzsl:
    train_X = torch.cat((data.train_feature, unseen_feature), 0)
    train_Y = torch.cat((data.train_label, unseen_label), 0)
    train_Y = util.map_label(train_Y, known_classes)
    # Train GZSL classifier
    zsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, known_classes.size(0), opt.cuda, opt.classifier_lr, 0.5, \
            25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096, test_openset = True)
else:
    zsl_cls = classifier.CLASSIFIER(unseen_feature, util.map_label(unseen_label, data.unseenclasses), data,
                                    data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                    opt.syn_num, False, netDec = netDec, dec_size = opt.attSize, dec_hidden_size=4096, test_openset = True)

# input_dim = opt.resSize
first_classifier = classifier.LINEAR_LOGSOFTMAX_CLASSIFIER(input_dim,known_classes.size(0))
first_classifier.load_state_dict(zsl_cls.best_model)
first_classifier.cuda()

for i in range(0, opt.count_ASE, opt.batch_size):
    z_value, z_label = next_batch(opt.batch_size)
    z_0_value = z_value
    z_0 = to_torch(z_0_value)
    z_label = z_label.cuda()
    for j in range(opt.iter_num):
        z = to_torch(z_value, requires_grad=True)
        syn_noise = torch.FloatTensor(opt.batch_size, opt.nz)
        syn_noise = syn_noise.cuda()
        syn_noise.normal_(0, 1)
        syn_noisev = syn_noise
        syn_noisev.requires_grad = False
        syn_noisev = syn_noisev.cuda()
        fake = netG(syn_noisev, c = z)

        if netF is not None:
            dec_out = netDec(fake)
            dec_hidden_feat = netDec.getLayersOutDet()
            feedback_out = netF(dec_hidden_feat)
            fake = netG(syn_noisev, a1 = opt.a2, c = z, feedback_layers = feedback_out)
        if netDec:
            netDec.eval()
            fake = compute_dec_out(netDec, fake, input_dim)
        output, logits = first_classifier(fake)
        augmented_logits = F.pad(logits, pad=(0,1))
        
        sum_of_log = torch.sum(torch.exp(augmented_logits), dim = 1)
        cf_loss = opt.energy_temp * torch.logsumexp(logits / opt.energy_temp, dim=1)
        distance_loss = torch.norm(z - z_0, dim = 1) * opt.distance_weight
            
        total_loss = cf_loss + distance_loss
        scores = F.softmax(augmented_logits, dim=1)
        dc_dz = autograd.grad(total_loss, z, torch.ones(total_loss.size()).cuda())[0]
        z = z - dc_dz * opt.ASE_speed
        z_value = z.data.cpu().numpy()
        del z
    if unknown_attribute is None:
        unknown_attribute = z_value
    else:
        unknown_attribute = np.concatenate((unknown_attribute, z_value), axis=0)
unknown_attribute = torch.from_numpy(unknown_attribute[:opt.count_ASE]).cuda()
# generate synthetic unknown features
def generate_syn_unknown_feature(generator, unknown_attribute, num, netF=None, netDec=None):
    nclass = unknown_attribute.shape[0]
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = i
        iclass_att = unknown_attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise)
        syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(-1)
    frame = syn_feature.cpu().detach().numpy()
    ori_label = syn_label.cpu().detach().numpy()
    nan_row = np.isnan(frame[:,0]) #找到对应的行
    row = np.where(nan_row==True) #行号
    frame_fix = np.delete(frame, row, axis=0) #删除行
    label_fix = np.delete(ori_label, row, axis=0)
    frame_fix = torch.Tensor(frame_fix)
    label_fix = torch.Tensor(label_fix)
    return frame_fix, label_fix
    return syn_feature, syn_label



unknown_feature, unknown_label = generate_syn_unknown_feature(netG, unknown_attribute, opt.unknown_num, netF=netF, netDec=netDec)
# train classifier
if opt.gzsl:
    known_classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
    
    ASE_X = torch.cat((data.train_feature, unseen_feature, unknown_feature), 0)
    ASE_Y = torch.cat((data.train_label, unseen_label, unknown_label), 0)
    ASE_Y = util.map_label(ASE_Y, known_classes)
    gzsl_X = torch.cat((data.train_feature, unseen_feature), 0)
    gzsl_Y = torch.cat((data.train_label, unseen_label), 0)
    gzsl_Y = util.map_label(gzsl_Y, known_classes)
    
    ASE_cls = classifier.CLASSIFIER(ASE_X, ASE_Y, data, known_classes.size(0)+1, opt.cuda, opt.classifier_lr, 0.5, \
                50, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    gzsl_cls = classifier.CLASSIFIER(gzsl_X, gzsl_Y, data, known_classes.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                50, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
else:
    train_X = torch.cat((unseen_feature, unknown_feature), 0)
    train_Y = torch.cat((unseen_label, unknown_label), 0)
    train_Y = util.map_label(train_Y, data.unseenclasses)
    ASE_cls = classifier.CLASSIFIER(train_X, train_Y, data, data.unseenclasses.size(0)+1, opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    zsl_cls = classifier.CLASSIFIER(unseen_feature, util.map_label(unseen_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
if opt.gzsl:
    print("gzsl_cls H is: ", gzsl_cls.H)
    print("gzsl_cls acc_seen is: ", gzsl_cls.acc_seen)
    print("gzsl_cls acc_unseen is: ", gzsl_cls.acc_unseen)
    print("ASE_cls H is: ", ASE_cls.H)
    print("ASE_cls acc_seen is: ", ASE_cls.acc_seen)
    print("ASE_cls acc_unseen is: ", ASE_cls.acc_unseen)
    print("gzsl_cls fpr95 is: ", gzsl_cls.fpr95)
    print("ASE_cls fpr95     is: ", ASE_cls.fpr95)
    print("gzsl_cls auroc is: ", gzsl_cls.auroc)
    print("ASE_cls auroc is: ", ASE_cls.auroc)
else:
    print("zsl_cls acc : {}".format(zsl_cls.acc))
    print("ASE_cls acc : {}".format(ASE_cls.acc))
    print("zsl_cls fpr95 : {}".format(zsl_cls.fpr95))
    print("ASE_cls fpr95 : {}".format(ASE_cls.fpr95))
    print("zsl_cls auroc : {}".format(zsl_cls.auroc))
    print("ASE_cls auroc : {}".format(ASE_cls.auroc))