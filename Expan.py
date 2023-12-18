import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from Trans import Transformer
from utils import *
from HCL import get_cl_dataset, cl_collate_fn, cl_criterion, cluster_criterion
import time
import pickle
import numpy as np
from tqdm import tqdm


class Model(nn.Module):

    def __init__(self, len_vocab, cluster_num=41, bert_model='bert-base-uncased', dim=768, num_patch=36):
        super().__init__()
        self.num_patch = num_patch

        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        self.mask_token_id = self.tokenizer.mask_token_id

        self.visual_linear = nn.Sequential(nn.Linear(2048, dim),
                                           nn.Dropout(0.2),
                                           nn.LayerNorm(dim),
                                           nn.GELU())
        self.visual_pos_embedding = nn.Parameter(torch.randn(1, num_patch, dim))
        self.visual_transformer = Transformer(dim, depth=3, dropout=0.1)
        
        self.cross_transformer = Transformer(dim, depth=3, dropout=0.1)
        # classification head f
        self.head = nn.Sequential(nn.Linear(dim, dim),
                                  nn.Dropout(0.2),
                                  nn.LayerNorm(dim),
                                  nn.GELU(),
                                  nn.Linear(dim, len_vocab),
                                  nn.Dropout(0.2),
                                  nn.LogSoftmax(dim=-1))
        
        # projection head p
        self.projection_head = nn.Sequential(nn.Linear(dim, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, 128, bias=True))
        
        # cluster head c
        self.cluster_head = nn.Sequential(nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim),
                                               nn.ReLU(), nn.Linear(dim, cluster_num), nn.Softmax(dim=1))

        # momentum distillation, self-training
        self.momentum = 0.995
        self.bert_m = BertModel.from_pretrained(bert_model)
        self.visual_linear_m = nn.Sequential(nn.Linear(2048, dim),
                                            nn.Dropout(0.2),
                                            nn.LayerNorm(dim),
                                            nn.GELU())
        self.visual_pos_embedding_m = nn.Parameter(torch.randn(1, num_patch, dim))
        self.visual_transformer_m = Transformer(dim, depth=3, dropout=0.1)
        
        self.cross_transformer_m = Transformer(dim, depth=3, dropout=0.1)
        self.head_m = nn.Sequential(nn.Linear(dim, dim),
                                  nn.Dropout(0.2),
                                  nn.LayerNorm(dim),
                                  nn.GELU(),
                                  nn.Linear(dim, len_vocab),
                                  nn.Dropout(0.2),
                                  nn.LogSoftmax(dim=-1))
        self.model_pairs = [[self.bert, self.bert_m],
                            [self.visual_linear, self.visual_linear_m],
                            [self.visual_transformer, self.visual_transformer_m],
                            [self.cross_transformer, self.cross_transformer_m],
                            [self.head, self.head_m]]
        self.copy_params()
    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = True
    
    def momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    def forward(self, x, y, mode='0'):            
        mask_pos = (x == self.mask_token_id).nonzero(as_tuple=True)
        x_mask = (x != 0).long()
        text_features = self.bert(x, x_mask)[0]

        visual_features = self.visual_linear(y) + self.visual_pos_embedding
        visual_features = self.visual_transformer(visual_features)

        cross_mask = torch.concat((x_mask, torch.ones(x.shape[0], self.num_patch).cuda()), 1)
        hidden_stats = self.cross_transformer(torch.concat((text_features, visual_features), 1), mask=cross_mask)

        set_embeddings = hidden_stats[mask_pos]

        if mode == '0':
            set_distributions = self.head(set_embeddings)
            projection = None
            cluster = None
            momentum_dist = None
        if mode == '1':
            set_distributions = None
            projection = self.projection_head(set_embeddings)
            projection = F.normalize(projection, dim=-1)
            cluster = None
            momentum_dist = None
        if mode == '2':
            set_distributions = None
            projection = None
            cluster = self.cluster_head(set_embeddings)
            momentum_dist = None

        if mode == '3' or mode == '4':  # momentum distillation
            set_distributions = self.head(set_embeddings)
            projection = None
            cluster = None

            self.momentum_update()
            text_features_m = self.bert_m(x, x_mask)[0]
            visual_features_m = self.visual_linear_m(y) + self.visual_pos_embedding_m
            visual_features_m = self.visual_transformer_m(visual_features_m)
            hidden_stats_m = self.cross_transformer_m(torch.concat((text_features_m, visual_features_m), 1), mask=cross_mask)
            set_embeddings_m = hidden_stats_m[mask_pos]
            momentum_dist = self.head_m(set_embeddings_m)
        return set_distributions, projection, cluster, momentum_dist


class Eid2Data(Dataset):
    def __init__(self, eid, eid2sents_folder, label_indexs, siz=None):
        self.eid = eid
        self.data = pickle.load(open(os.path.join(eid2sents_folder, str(eid)+'.pkl'), 'rb'))
        self.sents = self.data[0]
        self.imgs = self.data[1]

        self.label_indexs = label_indexs

        if siz is not None:
            if siz <= len(self.sents):
                indexs = np.random.choice(len(self.sents), siz, replace=False)
                self.sents = [self.sents[i] for i in indexs]
                self.imgs = [self.imgs[i] for i in indexs]
            else:
                indexs = np.random.choice(len(self.sents), siz, replace=True)
                self.sents = [self.sents[i] for i in indexs]
                self.imgs = [self.imgs[i] for i in indexs]

        self.num_sents = len(self.sents)

    def __len__(self):
        return self.num_sents

    def __getitem__(self, index):
        token_ids = self.sents[index]
        image_features = self.imgs[index]
        labels = self.label_indexs
        return token_ids, image_features, labels


def collate_fn(batch):
    batch_ids, batch_features, batch_labels = zip(*batch)

    batch_max_length = max(len(ids) for ids in batch_ids)
    batch_ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()

    batch_features = torch.concat([features.unsqueeze(0) for features in batch_features])

    return batch_ids, batch_features, batch_labels


def run_epoch(model, mode, data_iter, loss_compute, optimizer, log_step=100):
    total_loss_predict = 0
    if mode == '0':
        weight = 1.0
    if mode == '1':
        weight = 0.5
    if mode == '2':
        weight = 0.8
    if mode == '3' or mode == '4':
        weight = 1.0
    if mode == '1' or mode == '2':
        mode = '0'  # get [MASK] distribution from Model when cl and cluster
    # masked enity prediction task
    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        out, _, _, out_m = model.forward(batch[0].cuda(), batch[1].cuda(), mode)
        optimizer.zero_grad()
        loss_predict = loss_compute(mode, out, out_m, batch[2]) * weight
        loss_predict.backward()
        optimizer.step()

        total_loss_predict += loss_predict.item() / weight
        if (i + 1) % log_step == 0:
            print("Step: %4d        Loss: %.4f" % (i + 1, total_loss_predict / log_step))
            total_loss_predict = 0


def run_epoch_cl(model, cl_data_iter, cl_loss_compute, optimizer, batchsize, device_count, log_step=100):
    total_loss_cl = 0
    # contrastive learning
    for i, batch in tqdm(enumerate(cl_data_iter), total=len(cl_data_iter)):
        _, out_1, _, _ = model.forward(batch[0].cuda(), batch[1].cuda(), mode='1')
        _, out_2, _, _ = model.forward(batch[2].cuda(), batch[3].cuda(), mode='1')

        optimizer.zero_grad()
        loss_cl = cl_loss_compute(out_1, out_2, device_count, tau_plus=0.05, batch_size=batchsize, beta=1) * 0.5
        loss_cl.backward()
        optimizer.step()

        total_loss_cl += loss_cl.item() / 0.5
        if (i + 1) % log_step == 0:
            print("CL Step: %4d     CL Loss: %.4f" % (i + 1, total_loss_cl / log_step))
            total_loss_cl = 0


def run_epoch_cluster(model, cluster_data_iter, cluster_loss_criterion, optimizer, cluster_num, log_step=100):
    total_loss_cluster = 0
    # contrastive learning
    for i, batch in tqdm(enumerate(cluster_data_iter), total=len(cluster_data_iter)):
        _, _, out_1, _ = model.forward(batch[0].cuda(), batch[1].cuda(), mode='2')
        _, _, out_2, _ = model.forward(batch[2].cuda(), batch[3].cuda(), mode='2')

        optimizer.zero_grad()
        loss_cluster = cluster_loss_criterion(out_1, out_2, tau_plus=0.05, cluster_num=cluster_num, beta=1) * 0.2
        loss_cluster.backward()
        optimizer.step()

        total_loss_cluster += loss_cluster.item() / 0.2
        if (i + 1) % log_step == 0:
            print("CL Step: %4d     CL Loss: %.4f" % (i + 1, total_loss_cluster / log_step))
            total_loss_cluster = 0


class Loss_Compute(nn.Module):
    def __init__(self, criterion, len_vocab, smoothing=0):
        super(Loss_Compute, self).__init__()
        self.criterion = criterion
        self.len_vocab = len_vocab
        self.smoothing = smoothing

    def forward(self, mode, output, output_m, batch_labels, alpha=0.02):
        dists = []
        for labels in batch_labels:
            len_set = len(labels)
            dist = torch.zeros(self.len_vocab)
            dist.fill_(self.smoothing / (self.len_vocab - len_set))
            dist.scatter_(0, torch.tensor(labels), (1 - self.smoothing) / len_set)
            dists.append(dist)
        tensor_dists = torch.stack(dists).cuda()

        if mode == '3' or mode == '4':  # out_m is not None
            return (1 - alpha) * self.criterion(output, tensor_dists) + alpha * self.criterion(torch.exp(output), torch.exp(output_m))
        else:
            return self.criterion(output, tensor_dists)

class Expan(object):

    def __init__(self, args, cls_names, device_count):
        self.device_ids = range(device_count)
        # dict of entity names, list of entity ids, dict of line index
        self.eid2name, _, _ = load_vocab(os.path.join(args.dataset, args.vocab))

        self.eid2sents_folder = os.path.join(args.dataset, args.pkl_e2s)
        self.list_eids = []
        for file in os.listdir(self.eid2sents_folder):
            self.list_eids.append(int(file.split('.')[0]))
        self.len_vocab = len(self.list_eids)
        self.eid2index = {eid: i for i, eid in enumerate(self.list_eids)}


        self.cls_names = cls_names
        self.num_cls = len(cls_names)
        self.cluster_num = self.num_cls + 1

        self.model = Model(self.len_vocab, self.cluster_num)
        
        self.pkl_path_e2d = os.path.join(args.dataset, args.pkl_e2d)
        self.pkl_path_e2logd = os.path.join(args.dataset, args.pkl_e2d + '_log')
        self.eindex2dist = None
        self.eindex2logdist = None
        
        self.mode = args.mode
        self.cls2eids = None
        if os.path.exists(os.path.join(args.dataset, args.pkl_cls2eids)):
            self.cls2eids = pickle.load(open(os.path.join(args.dataset, args.pkl_cls2eids), 'rb'))

    # Pretraining model with Loss
    def pretrain(self, save_path, lr=1e-5, epoch=5, batchsize=128, num_sen_per_entity=64, smoothing=0.1, decay=0.01):
        if len(self.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # freeze part of BERT
        unfreeze_layers = ['visual_pos_embedding', 'encoder.layer.11', 'transformer', 'visual_linear', 'head']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        # optimizer
        no_decay = ['bias', 'layerNorm.weight', 'norm.weight', 'visual_linear.2.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        loss_compute = Loss_Compute(nn.KLDivLoss(reduction='batchmean'), self.len_vocab,
                                    smoothing=smoothing)

        self.model.cuda()

        for i in range(0, epoch):
            print('\n[Epoch %d]' % (i + 1))
            if self.mode in ['1', '2', '4']:
                dataset_pairs = get_cl_dataset(self.cls2eids, self.eid2sents_folder, num_sen_per_entity)
            if self.mode == '1' or self.mode == '4':
                # Contrastive Learning
                print('Contrastive Learning:')
                for _ in range(2):
                    data_loader_cl = DataLoader(dataset_pairs, batch_size=batchsize*len(self.device_ids), shuffle=True, collate_fn=cl_collate_fn, drop_last=True)
                    run_epoch_cl(self.model, data_loader_cl, cl_criterion, optimizer, batchsize, len(self.device_ids), log_step=20)
                print('')
            
            if self.mode == '2' or self.mode == '4':
                # Cluster
                print('Cluster:')
                for _ in range(2):
                    data_loader_cluster = DataLoader(dataset_pairs, batch_size=batchsize*len(self.device_ids), shuffle=True, collate_fn=cl_collate_fn, drop_last=True)
                    run_epoch_cluster(self.model, data_loader_cluster, cluster_criterion, optimizer, self.cluster_num, log_step=20)
                print('')

            # rebuild dataset before each epoch
            list_dataset = []
            for eid in tqdm(self.list_eids):
                this_dataset = Eid2Data(eid, self.eid2sents_folder, [self.eid2index[eid]], num_sen_per_entity)
                list_dataset.append(this_dataset)
            dataset = ConcatDataset(list_dataset)
            data_loader = DataLoader(dataset, batch_size=batchsize*len(self.device_ids), shuffle=True, collate_fn=collate_fn)
            # Masked Entity Prediction Learning
            run_epoch(self.model, self.mode, data_loader, loss_compute, optimizer, log_step=100)
            
            # Save Model Parameters
            model_pkl_name = "epoch_%d.pkl" % (i + 1)

            if len(self.device_ids) == 1:
                torch.save(self.model.state_dict(), os.path.join(save_path, model_pkl_name))
            else:
                torch.save(self.model.module.state_dict(), os.path.join(save_path, model_pkl_name))


    def expand(self, query_sets, target_size=103, ranking=True, mu=9,
               init_win_size=1, win_grow_rate=2.5, win_grow_step=20, total_iter=1):
        pre_expanded_sets = [None for _ in range(self.num_cls)]
        expanded_sets = query_sets
        cnt_iter = 0
        flag_stop = False
        pre_cursor = 13

        while cnt_iter < total_iter and flag_stop is False:
            flag_stop = True
            seed_sets = []
            cursor = target_size

            # check whether the expanded_set of each class is changed in last iteration
            # if so, renew seed set
            for i, expanded_set in enumerate(expanded_sets):
                changed = False
                if cnt_iter == 0:
                    seed_set = expanded_set
                    changed = True
                elif cnt_iter == 1:
                    seed_set = expanded_set[:13]
                    changed = True
                else:
                    # seed set is updated as the longest common set between pre_expanded_set and expanded_set
                    for j in range(pre_cursor, target_size):
                        for k in range(3, j):
                            if pre_expanded_sets[i][k] not in expanded_set[:j]:
                                changed = True
                                break
                        if changed and j < cursor:
                            cursor = j
                            pre_cursor = cursor
                            break
                    seed_set = expanded_set
                seed_sets.append(seed_set)

                if changed:
                    flag_stop = False
                else:
                    print(self.cls_names[i] + '  UNCHANGED')

            # truncate seed sets to same length
            if cnt_iter > 1:
                print('Cursor: ', cursor)
                print('')
                seed_sets = [seed_set[:cursor] for seed_set in seed_sets]

            pre_expanded_sets = expanded_sets
            expanded_sets = self.expand_(seed_sets, target_size, ranking, mu + cnt_iter * 2,
                                         init_win_size, win_grow_rate, win_grow_step)

            cnt_iter += 1

        return [eid_set[3:] for eid_set in expanded_sets]

    def expand_(self, seed_sets, target_size, ranking, mu, init_win_size, win_grow_rate, win_grow_step):
        expanded_sets = seed_sets

        eid_out_of_sets = set()
        for eid in self.list_eids:
            eid_out_of_sets.add(eid)
        for eid_set in seed_sets:
            for eid in eid_set:
                eid_out_of_sets.remove(eid)

        rounds = len(expanded_sets[0]) - 3
        toatl_eid_list = list(self.eid2index.keys())
        while len(expanded_sets[0]) < target_size:
            if rounds < win_grow_step:
                size_window = init_win_size
            elif rounds < 50:
                size_window = init_win_size + (rounds / win_grow_step) * win_grow_rate
            else:
                size_window = init_win_size + (rounds / win_grow_step) * win_grow_rate * 1.25
            if rounds >= 100:
                size_window = 1
            rounds += 1

            """ Expand """
            for i, cls in enumerate(self.cls_names):
                scores = np.zeros(self.len_vocab)
                eid_set = expanded_sets[i]
                for eid in eid_set:
                    mean_dist = self.get_mean_log_dist(eid)
                    scores += mean_dist

                indexs = np.argsort(-scores)

                """ Window Search """
                cnt = 0
                tgt_eid = None
                min_KL_div = float('inf')
                set_dist = np.zeros(self.len_vocab)
                len_set = len(expanded_sets[0])

                for index in indexs:
                    if cnt >= int(size_window):
                        break
                    eid = toatl_eid_list[index]
                    if eid in eid_out_of_sets:
                        cnt += 1
                        feature_dist = self.get_feature_dist(eid)
                        mean_prob = np.mean(feature_dist)
                        set_dist[:] = mean_prob
                        set_dist[index] = feature_dist[index]

                        if len_set <= 41:
                            set_dist[[self.eid2index[eid] for eid in eid_set[:len_set]]] = mean_prob * 1000
                        else:
                            set_dist[[self.eid2index[eid] for eid in eid_set[:41]]] = mean_prob * 1000
                            set_dist[[self.eid2index[eid] for eid in eid_set[41:min(len_set, 75)]]] = mean_prob * 500
                            if len_set > 75:
                                set_dist[[self.eid2index[eid] for eid in eid_set[75:len_set]]] = mean_prob * 300

                        KL_div = KL_divergence(set_dist, feature_dist)
                        if KL_div < min_KL_div:
                            min_KL_div = KL_div
                            tgt_eid = eid

                expanded_sets[i].append(tgt_eid)
                eid_out_of_sets.remove(tgt_eid)

        if ranking:
            """ Re-Ranking """
            ranked_expanded_sets = []
            dt = np.dtype([('eid', 'int'), ('rev_score', 'float32')])

            for i, eid_set in enumerate(expanded_sets):
                # rank-score on original entity set
                scores = [mu / r for r in range(1, target_size - 2)]

                # sort with KL-div bewtween set distribution and entity's feature distribution
                set_dist = np.zeros(self.len_vocab)

                eids = eid_set[3:]
                KLdivs = []
                for eid in eids:
                    feature_dist = self.get_feature_dist(eid)
                    mean_prob = np.mean(feature_dist)
                    set_dist[:] = mean_prob
                    set_dist[self.eid2index[eid]] = feature_dist[self.eid2index[eid]]

                    set_dist[[self.eid2index[eid] for eid in eid_set[:20]]] = mean_prob * 1600
                    set_dist[[self.eid2index[eid] for eid in eid_set[20:35]]] = mean_prob * 900
                    set_dist[[self.eid2index[eid] for eid in eid_set[35:70]]] = mean_prob * 600
                    set_dist[[self.eid2index[eid] for eid in eid_set[70:]]] = mean_prob * 150
                    diverg = KL_divergence(set_dist, feature_dist)
                    KLdivs.append(diverg)

                arg_sorted_z = np.argsort(KLdivs)
                for j, r in enumerate(arg_sorted_z):
                    scores[r] += 1 / (j + 1)

                z = list(zip(eids, -np.array(scores)))
                z = np.array(z, dtype=dt)
                sorted_z = np.sort(z, order='rev_score')
                ranked_set = eid_set[:3] + [x[0] for x in sorted_z]
                ranked_expanded_sets.append(ranked_set)
        else:
            ranked_expanded_sets = expanded_sets

        return ranked_expanded_sets

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def make_eindex2dists(self, batchsize=256):
        self.model.cuda()
        self.model.eval()
        eindex2logdist = []
        eindex2dist = []
        pkl_path_e2d = self.pkl_path_e2d
        pkl_path_e2logd = self.pkl_path_e2logd
        # if os.path.exists(self.pkl_path_e2d):
        #     self.eindex2dist = pickle.load(open(pkl_path_e2d, 'rb'))
        #     self.eindex2logdist = pickle.load(open(pkl_path_e2logd, 'rb'))
        #     return

        print('Total entities: %d' % len(self.list_eids))
        print('Making %s and %s ...' % (pkl_path_e2d, pkl_path_e2logd))
        for i, eid in tqdm(enumerate(self.list_eids), total=len(self.list_eids)):
            list_dists = []
            dataset = Eid2Data(eid, self.eid2sents_folder, [])
            data_loader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn)

            with torch.no_grad():
                for j, batch in enumerate(data_loader):
                    output, _, _, _ = self.model.forward(batch[0].cuda(), batch[1].cuda(), mode='0')
                    list_dists.append(output)

            log_dists = torch.cat(list_dists).cpu().numpy()
            eindex2logdist.append(np.mean(log_dists, axis=0))
            dists = np.exp(log_dists)
            eindex2dist.append(np.mean(dists, axis=0))

            torch.cuda.empty_cache()

        print('Writing to disk ...')
        # pickle.dump(eindex2logdist, open(pkl_path_e2logd, 'wb'))
        # pickle.dump(eindex2dist, open(pkl_path_e2d, 'wb'))
        self.eindex2logdist = eindex2logdist
        self.eindex2dist = eindex2dist

    def get_mean_log_dist(self, eid):
        mean_log_dist = self.eindex2logdist[self.eid2index[eid]]
        mean_log_dist = standardization(mean_log_dist)
        return mean_log_dist

    def get_feature_dist(self, eid):
        feature_dist = self.eindex2dist[self.eid2index[eid]]
        return feature_dist
