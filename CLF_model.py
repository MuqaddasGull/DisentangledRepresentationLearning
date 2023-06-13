#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init1(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
def reduce_with_choice(mu1, mu2, var1, var2, choice = None, eta = 0):
    term1 = torch.pow(mu1-mu2,2)
    term2 = torch.div(term1,eta+var1+var2)
    term3 = torch.mul(term2,-0.5)
    term4 = torch.exp(term3)
    term5 = torch.sqrt(var1+var2) + eta
    res = torch.div(term4, term5)
    return torch.mean(res) if choice == 'mean' else torch.sum(res)

class Gaussian_Distance(nn.Module):
    def __init__(self,kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern=kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)


    def forward(self, mu_a,logvar_a,mu_b,logvar_b,reduce='mean'):

        var_a = torch.exp(logvar_a)
        var_b = torch.exp(logvar_b)

        mu_a1 = mu_a.view(mu_a.size(0),1,-1)
        mu_a2 = mu_a.view(1,mu_a.size(0),-1)
        var_a1 = var_a.view(var_a.size(0),1,-1)
        var_a2 = var_a.view(1,var_a.size(0),-1)

        mu_b1 = mu_b.view(mu_b.size(0),1,-1)
        mu_b2 = mu_b.view(1,mu_b.size(0),-1)
        var_b1 = var_b.view(var_b.size(0),1,-1)
        var_b2 = var_b.view(1,var_b.size(0),-1)

        if reduce == 'mean':
            vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice = 'mean')
            vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice = 'mean')
            vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice = 'mean')
        
        else:
            vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice = 'sum')
            vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice = 'sum')
            vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice = 'sum')
        loss = vaa+vbb-torch.mul(vab,2.0)

        return loss

class ClsModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModel, self).__init__()
        self.fc1 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        x = self.lsm(x)
        return x
 
class ClsUnseen(torch.nn.Module):
    def __init__(self, att):
        super(ClsUnseen, self).__init__()
        self.W = att.type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=1024, out_features=300, bias=True)
        self.lsm = nn.LogSoftmax(dim=1)
        #print(f"__init__ {self.W.shape}")

    def forward(self, feats=None, classifier_only=False):
        f = self.fc1(feats)
        x = f.mm(self.W.transpose(1,0))
        x = self.lsm(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.attSize
        #print("latent_size",latent_size) 300
        in_c = layer_sizes[0] + latent_size
        #print("layer_sizes[-1]",layer_sizes[-1])  # 4096
        self.fc1 = nn.Linear(in_c, layer_sizes[-1])
        self.fc2 = nn.Linear(4096, layer_sizes[-1])
        
        self.fc4 = nn.Linear(opt.attSize, layer_sizes[-1])
        
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x=None, att=None):
        
        if x is None: 
            x = self.lrelu(self.fc4(att))
        elif att is None:
            x = self.lrelu(self.fc2(x))
            #print("x2",x.shape)  (32,4096)
        else:
            x = torch.cat((x, att), dim=-1)  
            x = self.lrelu(self.fc1(x))
            #print("x1",x.shape)  (32,4096)

        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars
    
class Encoder1(nn.Module):
    def __init__(self, opt):
        super(Encoder1, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.attSize
        in_c = layer_sizes[0] + latent_size
        self.fc1 = nn.Linear(in_c, layer_sizes[-1])
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, att=None):
        
        #if x is None:
        #    x = torch.zeros(att.shape[0], 0)
        #    x = x.cuda()
        
        if att is not None: x = torch.cat((x, att), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

# Generator body used for late and hybrid fusion
class generator(nn.Module):
    def __init__(self, opt, att_size):
        super(generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        input_size = att_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        x = torch.cat((x, att), 1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x

## HYBRID FUSION SELF ATTENTION ###
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear  = nn.Linear(d_model, d_model, bias=False)
        self.v_linear  = nn.Linear(d_model, d_model, bias=False)
        self.k_linear  = nn.Linear(d_model, d_model, bias=False)
        self.out       = nn.Linear(d_model, d_model, bias=False)
        self.dropout_1   = nn.Dropout(dropout)
        self.dropout_2   = nn.Dropout(dropout)

    def forward(self, q, k, v):
        bs = q.size(0)
        residual = q
        # perform linear operation and split into h heads ## transpose to get dimensions bs * h * sl * d_model
        k = self.k_linear(k).view(bs, self.h, -1, self.d_k)
        q = self.q_linear(q).view(bs, self.h, -1, self.d_k)
        v = self.v_linear(v).view(bs, self.h, -1, self.d_k)
        scores = attention(q, k, v, self.d_k, self.dropout_1)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.dropout_2(self.out(concat))
        output += residual
        return output

def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = scores.masked_fill(scores == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores) 
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.apply(weights_init)

    def forward(self, x):
        residual = x
        x = self.linear_2(F.leaky_relu(self.linear_1(x), 0.2, True))
        x += residual
        return x

class Fusion_Attention(nn.Module):
    def __init__(self, heads, d_model, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.feedforward_net = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = self.feedforward_net(x)
        return x

## self.early_fusion -> 'ALF', self.late_fusion -> 'FLF' ##
class CLF(nn.Module):
    def __init__(self, opt):
        super(CLF, self).__init__()
        self.resSize, self.N, self.hiddensize, late_heads  = opt.resSize, opt.N, opt.hiddensize, 8     # 8 
        self.early_fusion = generator(opt, opt.attSize)
        self.late_fusion = generator(opt, opt.attSize)
        self.attn = Fusion_Attention(late_heads, self.resSize, self.hiddensize)

    def forward(self, noise, att, avg_att):
        late_out = torch.zeros(len(att),self.resSize).cuda()  #(64,4096)
        #print("late_out",late_out.shape)
        idx_count = torch.zeros(len(att)).cuda()      #(64)
        #print("idx_count",idx_count.shape)
        for j in range(att.size(1)):
            idx = [i for i in range(len(att)) if att[i,j].abs().sum() > 0]
            idx_count[idx] += 1
            late_out[idx] += torch.sigmoid(self.late_fusion(noise[idx],att[idx,j].cuda()))        
        late_out = late_out/idx_count.unsqueeze(1).clamp(min=1)
        early_out = torch.sigmoid(self.early_fusion(noise, avg_att))
        temp_out = torch.stack((late_out,early_out),1)
        #print("temp_out",temp_out.shape)  (64,2,4096)
        out = self.attn(temp_out)
        #print("out",out.shape)     (64,2,4096)
        out = torch.sigmoid(torch.mean(out,1))
        #print("outttt",out.shape)   (64,4096)
        return out

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att=None):
        if att is not None: x = torch.cat((x, att), 1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x

class generator1(nn.Module):
    def __init__(self, opt):
        super(generator1, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        #input_size = att_size * 2
        
        self.fc1 = nn.Linear(opt.attSize + opt.attSize, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, noise, avg_att):
        x = torch.cat((noise, avg_att), 1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid((x))
        
        return x
