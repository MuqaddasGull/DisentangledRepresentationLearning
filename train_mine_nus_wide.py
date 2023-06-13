#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import random
import os
import csv
import numpy as np
import warnings
import networks.CLF_model as model
import classifier as classifier
from config import opt
import util as util
#import util_mscoco as util
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
########################################################

#setting up seeds
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
torch.set_default_tensor_type('torch.FloatTensor')
cudnn.benchmark = True  # For speed i.e, cudnn autotuner
########################################################

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#calling the dataloader
data = util.DATA_LOADER(opt)
print("training samples: ", data.ntrain)

############## MODEL INITIALIZATION #############
netE = model.Encoder(opt)
netG = model.CLF(opt)
netD = model.Discriminator(opt)

#print(netE)
#print(netG)
#print(netD)
################################################
#init tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_labels = torch.LongTensor(opt.fake_batch_size, opt.nclass_all)
input_labels = torch.LongTensor(opt.batch_size, opt.nseen_class)
input_train_early_fusion_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_test_early_fusion_att = torch.FloatTensor(opt.fake_batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.attSize)

one = torch.tensor(1, dtype=torch.float)

#one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netE=netE.cuda()
    netG=netG.cuda()
    netD=netD.cuda()
    input_res = input_res.cuda()
    input_labels = input_labels.cuda()
    input_train_early_fusion_att = input_train_early_fusion_att.cuda()
    input_test_labels = input_test_labels.cuda()
    input_test_early_fusion_att = input_test_early_fusion_att.cuda()
    noise = noise.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    ## BCE+KL divergence loss  (64,4096)----(64,4096)===BCE===Muli-label
                     #          (64, 2048)---(64,2048)---L2/L1===single-label
    
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD),BCE,KLD

def loss_fn1(recon_x, x):
    ## BCE+KL divergence loss
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    return BCE 

def sample():
    #train dataloader
    batch_labels, batch_feature, late_fusion_train_batch_att, early_fusion_train_batch_att = data.next_train_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    #print("input_res",input_res.shape)   [64,4096]
    input_train_early_fusion_att.copy_(early_fusion_train_batch_att)
    #print("input_train_early_fusion_att",input_train_early_fusion_att.shape)   [64,300]
    input_labels.copy_(batch_labels) 
    #print("input_labels",input_labels.shape)    [64,925]
    return late_fusion_train_batch_att

def fake_sample(batch_size):
    #fake data synthesis dataloader
    batch_test_labels, late_fusion_test_batch_att, early_fusion_test_batch_att = data.next_test_batch(batch_size)
    input_test_labels.copy_(batch_test_labels)
    input_test_early_fusion_att.copy_(early_fusion_test_batch_att)
    return late_fusion_test_batch_att

def generate_syn_feature(netG, classes, batch_size):
    ## SYNTHESIS MULTI LABEL FEATURES
    nsample = classes.shape[0]  # zsl_classes or gzsl_classes
    if not nsample % batch_size == 0:
        nsample = nsample + (batch_size - (nsample % batch_size))
    nclass = classes.shape[1]
    syn_noise = torch.FloatTensor(batch_size, opt.attSize)
    syn_feature = torch.FloatTensor(nsample, opt.resSize)
    syn_label = torch.LongTensor(nsample, classes.shape[1])
    
    if opt.cuda:
        syn_noise = syn_noise.cuda()
    for k, i in enumerate(range(0, nsample, batch_size)):
        late_fusion_test_batch_att = fake_sample(batch_size)
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, att=late_fusion_test_batch_att, avg_att=input_test_early_fusion_att)
        syn_feature.narrow(0, k*batch_size, batch_size).copy_(output)
        syn_label.narrow(0, k*batch_size, batch_size).copy_(input_test_labels)
    return syn_feature, syn_label

# setup optimizer
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#######################
##Checkpoints##########
#######################
model_dir = 'models/NUS_WIDE/'
model_path= model_dir+'checkpoint/model_bs_orig{}.pth'.format(opt.batch_size)
model_final_path= model_dir+'/model_final_bs{}.pth'.format(opt.batch_size)

Resume = False # Training starts from 0 epoch
#Resume = True # Load Checkpoint

start_epoch = 0
if Resume:
    print('Resuming model from {}'.format(model_path))
    checkpoint = torch.load(model_path)
    netE.load_state_dict(checkpoint['model_netE'])
    optimizerE.load_state_dict(checkpoint['optimizer_netE'])
    netG.load_state_dict(checkpoint['model_netG'])
    optimizerG.load_state_dict(checkpoint['optimizer_netG'])
    start_epoch = checkpoint['epoch']+1
    errG = checkpoint['loss']        
    print('Resume training from epoch {}'.format(start_epoch))
else:
    print('Start training from epoch 0')
#######################


def calc_gradient_penalty(netD, real_data, fake_data, input_att=None):
    alpha = torch.rand(opt.batch_size, 1) 
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad = True
    if input_att is None:
        disc_interpolates = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates, att=input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)** 2).mean() * opt.lambda1
    return gradient_penalty


f1_best_GZSL_AP = 0
f1_best_GZSL_F1_5 = 0
f1_best_GZSL_F1_3 = 0
f1_best_ZSL_F1_5 = 0
f1_best_ZSL_F1_3 = 0

sum_f1_best_GZSL_F1 = 0
sum_f1_best_ZSL_F1 = 0

gzsl_best_epoch=0
zsl_best_epoch=0


tic1 = time.time()
###############################

#training loop
for epoch in range(0, opt.nepoch+1):
    img_loss = 0
    att_loss = 0
    comb_loss = 0
    
    # Here I added training of these models
    #netE.train()
    #netG.train()
    
    tic = time.time()
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netE.parameters():
            p.requires_grad = True
            
        for p in netG.parameters():
            p.requires_grad = True

        for param in netE.parameters():
            param.grad = None
        for param in netG.parameters():
            param.grad = None
        
        late_fusion_train_batch_att = sample()
        
        means_img, log_var_img = netE(x=input_res,att=None)   # Encoder (x,ALF)
        std_img = torch.exp(0.5 * log_var_img)
        eps = torch.randn([opt.batch_size, opt.attSize])
        if opt.cuda: eps=eps.cuda()
        z_from_img = eps * std_img + means_img
        
        recon_img = netG(z_from_img, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att)  # CLF_Generator ( z, FLF, ALF)
        #print("recon_img",recon_img.shape)  (64,4096)
        vae_loss_img,BC_img,KL_img = loss_fn(recon_img, input_res, means_img, log_var_img) #BCE+KL divergence loss
        img_loss += vae_loss_img.item()
        errG = vae_loss_img
        
        means_comb, log_var_comb = netE(x=input_res,att=input_train_early_fusion_att)   # Encoder (x,ALF)
        std_comb = torch.exp(0.5 * log_var_comb)
        eps = torch.randn([opt.batch_size, opt.attSize])
        if opt.cuda: eps=eps.cuda()
        z_from_comb = eps * std_comb + means_comb
        
        recon_comb = netG(z_from_comb, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att) # CLF_Generator ( z, FLF, ALF)
        vae_loss_comb,BC_comb,KL_comb = loss_fn(recon_comb, input_res, means_comb, log_var_comb) #BCE+KL divergence loss
        comb_loss += vae_loss_comb.item()
        errG += vae_loss_comb
        
        means_att, log_var_att = netE(x=None, att=input_train_early_fusion_att)   # Encoder (x,ALF)
        std_att = torch.exp(0.5 * log_var_att)
        eps = torch.randn([opt.batch_size, opt.attSize])
        if opt.cuda: eps=eps.cuda()
        z_from_att = eps * std_att + means_att
        
        recon_att = netG(z_from_att, att=late_fusion_train_batch_att, avg_att=input_train_early_fusion_att) # CLF_Generator ( z, FLF, ALF)
        vae_loss_att,BC_att,KL_att = loss_fn(recon_att, input_res, means_att, log_var_att) #BCE+KL divergence loss
        att_loss += vae_loss_att.item()
        errG += vae_loss_att
        ################# Distance Alignment ############
        #distance = torch.sqrt(torch.sum((means_img - means_att) ** 2, dim=1) + \
         #                     torch.sum((torch.sqrt(torch.sqrt(log_var_img.exp())-torch.sqrt(log_var_att.exp())) ** 2, 
          #                              dim=1))
        #distance = distance.sum()
        #errG += distance
        ######################################################
        optimizerE.zero_grad()
        optimizerG.zero_grad()
        errG.backward()
        optimizerE.step()
        optimizerG.step()
    
    img_loss /= data.ntrain / opt.batch_size
    comb_loss /= data.ntrain / opt.batch_size
    att_loss /= data.ntrain / opt.batch_size

    print('[%d/%d] Loss_img: %.4f Loss_comb: %.4f, Loss_att: %.4f' %
            (epoch, opt.nepoch, img_loss, comb_loss, att_loss))
    
    print("Generator {}th finished time taken {}".format(epoch, time.time()-tic))
    netG.eval()
    gzsl_syn_feature, gzsl_syn_label = generate_syn_feature(netG, data.GZSL_fake_test_labels, opt.fake_batch_size)
    #print("#########gzsl_syn_feature#########",gzsl_syn_feature.shape)   #(282900,4096)
    #print("#########gzsl_syn_label##########",gzsl_syn_label.shape)      # (282900,1006)
    
    if opt.gzsl:
        nclass = opt.nclass_all
        train_X = gzsl_syn_feature
        train_Y = gzsl_syn_label

        tic = time.time()
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass,
                                        opt.cuda, opt, opt.classifier_lr, 0.5, opt.classifier_epoch,
                                        opt.classifier_batch_size, True)

        sum_GZSL_F1_5 = gzsl_cls.sum_F1_scores_seen_unseen[4]*100 + gzsl_cls.sum_F1_scores_seen_unseen[0]*100
        if sum_f1_best_GZSL_F1 < sum_GZSL_F1_5:
            gzsl_best_epoch = epoch
            sum_f1_best_GZSL_F1 = sum_GZSL_F1_5
            sum_f1_best_GZSL_AP = gzsl_cls.sum_F1_scores_seen_unseen[0]
            sum_f1_best_GZSL_F1_3 = gzsl_cls.sum_F1_scores_seen_unseen[1]
            sum_f1_best_GZSL_P_3 = gzsl_cls.sum_F1_scores_seen_unseen[2]
            sum_f1_best_GZSL_R_3 = gzsl_cls.sum_F1_scores_seen_unseen[3]
            sum_f1_best_GZSL_F1_5 = gzsl_cls.sum_F1_scores_seen_unseen[4]
            sum_f1_best_GZSL_P_5 = gzsl_cls.sum_F1_scores_seen_unseen[5]
            sum_f1_best_GZSL_R_5 = gzsl_cls.sum_F1_scores_seen_unseen[6]
            
        print('GZSL: AP=%.4f' % (gzsl_cls.sum_F1_scores_seen_unseen[0]))
        print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' % (
            gzsl_cls.sum_F1_scores_seen_unseen[4], gzsl_cls.sum_F1_scores_seen_unseen[5], gzsl_cls.sum_F1_scores_seen_unseen[6]))
        print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' % (
            gzsl_cls.sum_F1_scores_seen_unseen[1], gzsl_cls.sum_F1_scores_seen_unseen[2], gzsl_cls.sum_F1_scores_seen_unseen[3]))
        print("GZSL classification finished time taken {}".format(time.time()-tic))
    
    ######### FETCHING ZSL CLASSIFIER TRAINING DATA ########################
    temp_label = gzsl_syn_label[:,:len(data.seenclasses)].sum(1)
    zsl_syn_label = gzsl_syn_label[temp_label==0][:,len(data.seenclasses):]
    zsl_syn_feature = gzsl_syn_feature[temp_label==0]
    
    ###############################################3########################

    tic = time.time()
    zsl_cls = classifier.CLASSIFIER(zsl_syn_feature, zsl_syn_label, data,
                                        data.unseenclasses.size(0), opt.cuda, opt, opt.classifier_lr,
                                        0.5, opt.classifier_epoch, opt.classifier_batch_size, False)

    sum_ZSL_F1 = zsl_cls.sum_F1_scores[4]*100 + zsl_cls.sum_F1_scores[0]*100
       
    if sum_f1_best_ZSL_F1 < sum_ZSL_F1:
        zsl_best_epoch = epoch
        sum_f1_best_ZSL_F1 = sum_ZSL_F1
        sum_f1_best_ZSL_AP = zsl_cls.sum_F1_scores[0]
        sum_f1_best_ZSL_F1_3 = zsl_cls.sum_F1_scores[1]
        sum_f1_best_ZSL_P_3 = zsl_cls.sum_F1_scores[2]
        sum_f1_best_ZSL_R_3 = zsl_cls.sum_F1_scores[3]
        sum_f1_best_ZSL_F1_5 = zsl_cls.sum_F1_scores[4]
        sum_f1_best_ZSL_P_5 = zsl_cls.sum_F1_scores[5]
        sum_f1_best_ZSL_R_5 = zsl_cls.sum_F1_scores[6]
       
    print('ZSL: AP=%.4f' % (zsl_cls.sum_F1_scores[0]))
    print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[4], zsl_cls.sum_F1_scores[5], zsl_cls.sum_F1_scores[6]))
    print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' % (zsl_cls.sum_F1_scores[1], zsl_cls.sum_F1_scores[2], zsl_cls.sum_F1_scores[3]))
    print("ZSL classification finished time taken {}".format(time.time()-tic))

    if epoch % 3 == 0 and epoch > 0: ## PRINT BEST EPOCH AFTER EVERY 3 EPOCHS
        print("LAST GZSL BEST EPOCH", gzsl_best_epoch)
        print('GZSL: AP=%.4f' % (sum_f1_best_GZSL_AP))
        print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5))
        print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, sum_f1_best_GZSL_R_3))
        print("LAST ZSL BEST EPOCH", zsl_best_epoch)
        print('ZSL: AP=%.4f' % (sum_f1_best_ZSL_AP))
        print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_ZSL_F1_5, sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5))
        print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
                (sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3))
    
    # reset G to training mode
    netG.train()
    torch.save({'epoch': epoch,
            'model_netE': netE.state_dict(),
            'optimizer_netE': optimizerE.state_dict(),
            'model_netG': netG.state_dict(),
            'optimizer_netG': optimizerG.state_dict(),
            'loss': errG}, model_path)
    print("Checkpoint Saved.", model_path)


print(" Total time taken {} ".format(time.time()-tic1))

print("GZSL BEST EPOCH", gzsl_best_epoch)
print('GZSL: AP=%.4f' % (sum_f1_best_GZSL_AP))
print('GZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5))
print('GZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, sum_f1_best_GZSL_R_3))

print("ZSL BEST EPOCH", zsl_best_epoch)
print('ZSL: AP=%.4f' % (sum_f1_best_ZSL_AP))
print('ZSL K=5 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_ZSL_F1_5, sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5))
print('ZSL K=3 : f1=%.4f,P=%.4f,R=%.4f' %
        (sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3))

#saving final model
torch.save({'epoch': opt.nepoch,
        'model_netE': netE.state_dict(),
        'optimizer_netE': optimizerE.state_dict(),
        'model_netG': netG.state_dict(),
        'optimizer_netG': optimizerG.state_dict(),
        'loss': errG}, model_final_path)

##saving results to csv file
fname = 'CLF_result_F1.csv'
row = [opt.nepoch, sum_f1_best_GZSL_AP, sum_f1_best_ZSL_AP, sum_f1_best_GZSL_F1_3, sum_f1_best_GZSL_P_3, 
        sum_f1_best_GZSL_R_3, sum_f1_best_ZSL_F1_3, sum_f1_best_ZSL_P_3, sum_f1_best_ZSL_R_3,
        sum_f1_best_GZSL_F1_5, sum_f1_best_GZSL_P_5, sum_f1_best_GZSL_R_5, sum_f1_best_ZSL_F1_5, 
        sum_f1_best_ZSL_P_5, sum_f1_best_ZSL_R_5, opt.summary]

with open(fname, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()
