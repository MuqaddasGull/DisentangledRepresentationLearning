#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import h5py
import time
import numpy as np
import random
random.seed(3483)
np.random.seed(3483)

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
            #ans[key] = item.value [()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# SYNTHESIS LABELS FROM TRAIN DATA 
def generate_fake_test_from_train_labels(train_seen_label, attribute, seenclasses, unseenclasses, num, per_seen=0.10, \
                                        per_unseen=0.40, per_seen_unseen= 0.50):
    """
    Input:
        train_seen_label-> images with labels containing objects less than opt.N
        attribute-> array containing word embeddings
        seenclasses-> array containing seen class indices
        unseenclasses-> array containing unseen class indices
        num-> number of generated synthetic labels
    Output:
        gzsl -> tensor containing synthetic labels of only unseen, seen and seen-unseen classes.  
    
    """
    if train_seen_label.min() == 0:
        print("Training data already trimmed and converted") ##########
    else:
        print("original training data received (-1,1)'s ")
        train_seen_label = torch.clamp(train_seen_label,0,1)

    #remove all zero labeled images while training
    train_seen_label = train_seen_label[(train_seen_label.sum(1) != 0).nonzero().flatten()]
    seen_attributes = attribute[seenclasses]
    unseen_attributes = attribute[unseenclasses]
    seen_percent, unseen_percent, seen_unseen_percent = per_seen , per_unseen, per_seen_unseen

    print("seen={}, unseen={}, seen-unseen={}".format(seen_percent, unseen_percent, seen_unseen_percent))
    print("syn num={}".format(num))  # 1
    gzsl = []
    for i in range(0, num):
        new_gzsl_syn_list = []
        seen_unseen_label_pairs = {}
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(unseen_attributes)
        for seen_idx, seen_att in zip(seenclasses,seen_attributes):
            _, indices = nbrs.kneighbors(seen_att[None,:])
            seen_unseen_label_pairs[seen_idx.tolist()] = unseenclasses[indices[0][0]].tolist()

        #ADDING ONLY SEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*seen_percent)]
        seen_labels = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(seen_labels.shape[0], attribute.shape[0])
        _new_gzsl_syn_list[:,:len(seenclasses)] = seen_labels
        new_gzsl_syn_list.append(_new_gzsl_syn_list)

        #ADDING ONLY UNSEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*unseen_percent)]
        temp_label = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(temp_label.shape[0], attribute.shape[0])
        for m,lab in enumerate(temp_label):
            new_lab = torch.zeros(attribute.shape[0])
            unseen_lab = lab.nonzero().flatten()
            u=[]
            for i in unseen_lab:
                u.append(seen_unseen_label_pairs[i.tolist()])
            new_lab[u]=1
            _new_gzsl_syn_list[m,:] = new_lab
        unseen_labels = _new_gzsl_syn_list
        new_gzsl_syn_list.append(unseen_labels)

        #ADDING BOTH SEEN AND UNSEEN LABELS 50% OF THE SELECTED SEEN LABELS IS MAPPED TO UNSEEN LABELS
        idx = torch.randperm(len(train_seen_label))[0:int(len(train_seen_label)*seen_unseen_percent)]
        temp_label = train_seen_label[idx]
        _new_gzsl_syn_list = torch.zeros(temp_label.shape[0], attribute.shape[0])
        for m,lab in enumerate(temp_label):
            u = []
            new_lab = torch.zeros(attribute.shape[0])
            seen_unseen_lab = lab.nonzero().flatten()
            temp_seen_label = np.random.choice(seen_unseen_lab,int(len(seen_unseen_lab)*0.50))
            u.extend(temp_seen_label)
            rem_seen_label =  np.setxor1d(temp_seen_label,seen_unseen_lab)
            for i in rem_seen_label:
                u.append(seen_unseen_label_pairs[i.tolist()])
            new_lab[u]=1
            _new_gzsl_syn_list[m,:] = new_lab
        seen_unseen_labels = _new_gzsl_syn_list
        new_gzsl_syn_list.append(seen_unseen_labels)

        new_gzsl_syn_list = torch.cat(new_gzsl_syn_list)
        gzsl.append(new_gzsl_syn_list)
    
    gzsl = torch.cat(gzsl)
    tmp_list = gzsl.sum(0)
    ## To make sure every unseen label gets covered
    empty_lab = torch.arange(tmp_list.numel())[tmp_list==0]
    min_uc = int(tmp_list[len(seenclasses):][tmp_list[len(seenclasses):]>0].min().item())
    for el in empty_lab:
        idx = torch.randperm(gzsl.size(0))[:min_uc]
        gzsl[idx,el] = 1
    gzsl = gzsl.long()
    print("GZSL TEST LABELS:",gzsl.shape)
    return gzsl

def get_seen_unseen_classes(file_tag_all, file_tag_unseen):
    """
    Input:
        file_tag_all -> All Categories categories.
        file_tag_unseen -> Unseen categories.
        
    Output:
        seen_cls_idx -> selected seen class indices
        unseen_cls_idx -> selected unseen class indices
    """
    with open(file_tag_all, "r") as file:
        tag_all = np.array(file.read().splitlines())
    with open(file_tag_unseen, "r") as file:
        tag_unseen = np.array(file.read().splitlines())
    seen_cls_idx = np.array(
        [i for i in range(len(tag_all)) if tag_all[i] not in tag_unseen])
    unseen_cls_idx = np.array(
        [i for i in range(len(tag_all)) if tag_all[i] in tag_unseen])
    return seen_cls_idx, unseen_cls_idx


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        tic = time.time()
        src = "/muqadas/deeplearning/torch/Generative_MLZSL/datasets/MS_COCO" #folder for path containing features
        att_path = os.path.join(src,'word_embedding','word_glo.csv')
        file_tag_all = os.path.join(src,'ms_coco_Tags','TagList80.txt')
        file_tag_unseen = os.path.join(src,'ConceptsList','Concepts15.txt')
        self.seen_cls_idx, self.unseen_cls_idx = get_seen_unseen_classes(file_tag_all, file_tag_unseen)
        #src_att = pickle.load(open(att_path, 'rb'))
        
        src_att = np.loadtxt(att_path,
                 delimiter=",", dtype=float) # (300, 80)
        src_att = src_att.transpose()
        #print("src_att",src_att.shape)   #(80,300)
        print("attributes are combined in this order-> seen+unseen")
        
        print ("src_att[self.seen_cls_idx]", src_att[0].shape)
        
        self.attribute = torch.from_numpy(normalize(np.concatenate((src_att[self.seen_cls_idx],src_att[self.unseen_cls_idx]),axis=0)))
        #self.attribute = torch.from_numpy(normalize(src_att))

        #print("self.attribute",self.attribute.shape)  #(1006,300)
                
        #VGG features path   
        #import pdb;pdb.set_trace()
        train_loc = load_dict_from_hdf5(os.path.join(src, 'ms_coco_vgg_features','mscoco_train_pl_vgg.h5')) 
        test_unseen_loc=load_dict_from_hdf5(os.path.join(src,'ms_coco_vgg_features','mscoco_test_unseen_pl_vgg.h5')) 
        test_seen_unseen_loc=load_dict_from_hdf5(os.path.join(src,'ms_coco_vgg_features','mscoco_test_gzsl_pl_vgg.h5'))

        #train_loc = load_dict_from_hdf5(os.path.join(src, 'ms_coco_vgg_features','mscoco_seen_train_vgg19.h5')) # 
        #test_unseen_loc=load_dict_from_hdf5(os.path.join(src,'ms_coco_vgg_features','mscoco_zsl_test_vgg19.h5')) # 
        #test_seen_unseen_loc=load_dict_from_hdf5(os.path.join(src,'ms_coco_vgg_features','mscoco_test_gzsl_vgg19.h5')) #


        feature_train_loc = train_loc['features']
        #print("feature_train_loc",feature_train_loc.shape)  (161789,4096)   nus_seen_train_vgg19.h5
        label_train_loc = train_loc['labels']    
        # print("label_train_loc",label_train_loc)    #(161789,925)
        # Get the index of elements with value 1
        #result = np.where(label_train_loc[0] == 1)
        #print("Labels of Image 0", result[0], sep='\n')


        feature_test_unseen_loc = test_unseen_loc['features']                  
        #print("feature_test_unseen_loc",feature_test_unseen_loc.shape)   #(107859, 4096)  nus_zsl_test_vgg19.h5    
        label_test_unseen_loc = test_unseen_loc['labels']
        #print("label_test_unseen_loc",label_test_unseen_loc.shape)    #(107859, 81)
        feature_test_seen_unseen_loc = test_seen_unseen_loc['features']   
        #print("feature_test_seen_unseen_loc",feature_test_seen_unseen_loc.shape)    #(107859, 4096) nus_gzsl_test_vgg19.h5
        label_test_seen_unseen_loc = test_seen_unseen_loc['labels']
        #print("label_test_seen_unseen_loc",label_test_seen_unseen_loc.shape)   # (107859, 1006)
        print("Data loading finished, Time taken: {}".format(time.time()-tic))

        tic = time.time()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature_train_loc)
                _test_unseen_feature = scaler.transform(feature_test_unseen_loc)
                _test_seen_unseen_feature = scaler.transform(feature_test_seen_unseen_loc)

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label_train_loc).long()

                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label_test_unseen_loc).long()

                self.test_seen_unseen_feature = torch.from_numpy(_test_seen_unseen_feature).float()
                self.test_seen_unseen_feature.mul_(1/mx)
                self.test_seen_unseen_label = torch.from_numpy(label_test_seen_unseen_loc).long()
            else:
                self.train_feature = torch.from_numpy(feature_train_loc).float()
                self.train_label = torch.from_numpy(label_train_loc).long()
                self.test_unseen_feature = torch.from_numpy(feature_test_unseen_loc).float()
                self.test_unseen_label = torch.from_numpy(label_test_unseen_loc).long()

        print("REMOVING ZEROS LABELS")
        #print("train_label",self.train_label.shape) #  [61598, 65]
        temp_label = torch.clamp(self.train_label,0,1)  # put datat in a range like here in the range of [0,1]
        #print("temp_label b4",temp_label.shape)  # [61598, 65]
        temp_seen_labels = temp_label.sum(1)
        temp_label = temp_label[temp_seen_labels>0]  # remove those examples where no labels exist.
        #print("temp_label after",temp_label.shape) #  ([61598, 65]

        self.train_label           = temp_label     #[61598, 925]
        self.train_feature         = self.train_feature[temp_seen_labels>0]  

        self.train_trimmed_label   = self.train_label[temp_label.sum(1)<=opt.N]
        self.train_trimmed_feature = self.train_feature[temp_label.sum(1)<=opt.N]

        print("Data with N={} labels={}".format(opt.N,self.train_trimmed_label.shape)) #([61492, 65])
        print("Full Data labels={} with min label/feature = {} and max label/feature = {}".format(self.train_label.shape, temp_label.sum(1).min(), temp_label.sum(1).max()))  #min label/feature = 1 and max label/feature = 16

        self.seenclasses = torch.from_numpy(np.arange(0, self.seen_cls_idx.shape[-1]))  # [0-64]
        self.unseenclasses = torch.from_numpy(np.arange(0+self.seen_cls_idx.shape[-1], len(self.attribute)))  # [65-79]
                
        self.N = opt.N
        self.syn_num = opt.syn_num
        self.per_seen = opt.per_seen
        self.per_unseen = opt.per_unseen
        self.per_seen_unseen = opt.per_seen_unseen

        print("USING TRAIN FEATURES WITH <=N")
        self.ntrain = self.train_trimmed_feature.size()[0]
        train_labels = self.train_trimmed_label  #([61492, 65])

        self.ntest_unseen = self.test_unseen_feature.size()[0]
        #print("self.ntest_unseen",self.ntest_unseen) # 107859
        self.ntrain_class = self.seenclasses.size(0)
        #print("self.ntrain_class",self.ntrain_class) # 65
        self.ntest_class = self.unseenclasses.size(0)
        #print("self.ntest_class",self.ntest_class) # 15
        self.train_class = self.seenclasses.clone()       # clone function return a copy of input
        #print("self.train_class",self.train_class.shape) # 65
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        #print("self.allclasses",self.allclasses.shape)  # 80
        self.GZSL_fake_test_labels = generate_fake_test_from_train_labels(train_labels, self.attribute, self.seenclasses, \
                                        self.unseenclasses, self.syn_num, self.per_seen, self.per_unseen, self.per_seen_unseen)
       
        #print("self.GZSL_fake_test_labels",self.GZSL_fake_test_labels.shape)  [282708, 1006]
        print("Data preprocssing finished,Time taken: {}".format(time.time()-tic))
    
    def _average(self, lab, attribute):
        return torch.mean(attribute[lab], 0)

    def ALF_preprocess_att(self, labels, attribute):
        new_seen_attribute = torch.zeros(labels.shape[0], attribute.shape[-1])
        for i in range(len(labels)):
            lab = labels[i].nonzero().flatten()
            if len(lab) == 0: continue
            new_seen_attribute[i, :] = self._average(lab, attribute)  
        return new_seen_attribute

    def FLF_preprocess_att(self, labels, attribute):
        new_attributes = torch.zeros(labels.shape[0], self.N, attribute.shape[-1]) #new attributes [BS X 10 X 925]
        for i in range(len(labels)):
            lab = labels[i].nonzero().flatten()
            if len(lab) == self.N: new_attributes[i,:,:] = attribute[lab]
            elif len(lab) < self.N: 
                arg1 = torch.tensor(attribute[lab], dtype=torch.float64)
                arg2 = torch.tensor(torch.zeros((self.N - len(lab)), attribute.shape[-1]), dtype=torch.float64)

                #new_attributes[i,:,:] = torch.cat((attribute[lab],torch.zeros((self.N - len(lab)), attribute.shape[-1])))
                new_attributes[i,:,:] = torch.cat((arg1,arg2))     
        return new_attributes

    ## Training Dataloader
    def next_train_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        #print("self.train_trimmed_feature",self.train_trimmed_feature.shape) [141354, 4096]
        feature = self.train_trimmed_feature
        labels = self.train_trimmed_label
        batch_feature = feature[idx]
        batch_labels = labels[idx]
        
        early_fusion_train_batch_att = self.ALF_preprocess_att(batch_labels, self.attribute)  #att avg as attributes level fusions (64,300)
        late_fusion_train_batch_att = self.FLF_preprocess_att(batch_labels, self.attribute)  # feat fusion (64,10,300)
        return batch_labels, batch_feature, late_fusion_train_batch_att, early_fusion_train_batch_att

    ## Testing Dataloader
    def next_test_batch(self, batch_size):
        idx = torch.randperm(len(self.GZSL_fake_test_labels))[0:batch_size]
        batch_labels = self.GZSL_fake_test_labels[idx]
        early_fusion_test_batch_att = self.ALF_preprocess_att(batch_labels, self.attribute)
        late_fusion_test_batch_att = self.FLF_preprocess_att(batch_labels, self.attribute)

        return batch_labels, late_fusion_test_batch_att, early_fusion_test_batch_att

