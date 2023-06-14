
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
import classifiers.classifier_images_odin as classifier_odin
import classifiers.classifier_images as classifier

from config_images import opt
import os, time
import numpy as np
from torch.nn import functional as F


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
# train classifier
if opt.gzsl:
    known_classes = torch.cat((data.seenclasses, data.unseenclasses), 0)
    gzsl_X = torch.cat((data.train_feature, unseen_feature), 0)
    gzsl_Y = torch.cat((data.train_label, unseen_label), 0)
    gzsl_Y = util.map_label(gzsl_Y, known_classes)
    gzsl_cls_odin = classifier_odin.CLASSIFIER(gzsl_X, gzsl_Y, data, known_classes.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096, temperature = opt.odin_temperature,  NoiseMagnitude = opt.odin_magnitude)
else:
    zsl_cls_odin = classifier_odin.CLASSIFIER(unseen_feature, util.map_label(unseen_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096, temperature = opt.odin_temperature,  NoiseMagnitude = opt.odin_magnitude)

if opt.gzsl:
    print("gzsl_cls H is: ", gzsl_cls_odin.H)
    print("gzsl_cls acc_seen is: ", gzsl_cls_odin.acc_seen)
    print("gzsl_cls acc_unseen is: ", gzsl_cls_odin.acc_unseen)
    print("gzsl_cls fpr95 is: ", gzsl_cls_odin.fpr95)
    print("gzsl_cls auroc is: ", gzsl_cls_odin.auroc)
else:
    print("zsl_cls_ODIN acc : {}".format(zsl_cls_odin.acc))
    print("zsl_cls_ODIN fpr95 : {}".format(zsl_cls_odin.fpr95))
    print("zsl_cls_ODIN auroc : {}".format(zsl_cls_odin.auroc))
