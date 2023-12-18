import os
import torch
import argparse
from utils import *
from Expan import Expan
import sys
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='../data', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-pkl_e2s', default='entity2sents', help='name of entity2sents folder')
    parser.add_argument('-pretrained_model', default=None, help='name of pretrained model parameters')
    parser.add_argument('-save_path', default='../model/ptm', help='path to place model parameters')
    parser.add_argument('-pkl_e2d', default='entity2dist', help='name of entity2dist pkl file')

    parser.add_argument('-pkl_eid2cls', default='eid2cls.pkl', help='name of eid2cls pkl file')
    parser.add_argument('-pkl_cls2eids', default='cls2eids.pkl', help='name of cls2eids pkl file')
    parser.add_argument('-query', default='query', help='name of query file')
    parser.add_argument('-mode', default='4', help='0 for common mode, 1 for cl mode, 2 for cluster mode, 3 for momentum distillation, 4 for four loss')
    
    parser.add_argument('-output', default='results', help='file name for output')
    parser.add_argument('-result', default='MultiExpan.txt', help='file name for epoch result')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device_count = 8
    print(args)

    class_names = []
    query_sets = dict()
    gt = dict()
    num_query_per_class = 0
    for file in os.listdir(os.path.join(args.dataset, args.query)):
        class_name = file.split('.')[0]
        class_names.append(class_name)
        query_sets[class_name] = []
        gt[class_name] = set()
        num_query_per_class = 0

        with open(os.path.join(args.dataset, args.query, file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n':
                    break
                num_query_per_class += 1
                temp = line.strip().split(' ')
                query_sets[class_name].append([int(eid) for eid in temp])

        with open(os.path.join(args.dataset, 'gt', file), encoding='utf-8') as f:
            for line in f:
                temp = line.strip().split('\t')
                eid = int(temp[0])
                if int(temp[2]) >= 1:
                    gt[class_name].add(eid)

    expan = Expan(args, class_names, device_count=device_count)

    if args.pretrained_model is None:
        if args.mode in ['0', '3', '4']:
            expan.pretrain(args.save_path, lr=1e-5, epoch=50, batchsize=256, num_sen_per_entity=64, smoothing=0.075, decay=0.01)
        if args.mode == '1':
            expan.pretrain(args.save_path, lr=1.5e-5, epoch=50, batchsize=256, num_sen_per_entity=64, smoothing=0.075, decay=0.01)
        if args.mode == '2':
            expan.pretrain(args.save_path, lr=1.5e-5, epoch=50, batchsize=256, num_sen_per_entity=64, smoothing=0.075, decay=0.01)
        exit(0)

    expan.load_model(os.path.join(args.save_path, args.pretrained_model))
    expan.make_eindex2dists(batchsize=128)

    '''
    Expanding and Evalutation
    '''
    if not os.path.exists(os.path.join(args.dataset, args.output)):
        os.mkdir(os.path.join(args.dataset, args.output))
    if not os.path.exists(os.path.join(args.dataset, args.output, args.pretrained_model.split('.')[0])):
        os.mkdir(os.path.join(args.dataset, args.output, args.pretrained_model.split('.')[0]))

    MAPs = [0, 0, 0, 0]
    Ps = [0, 0, 0, 0]
    num_class = len(class_names)
    target_size = 105
    if args.mode == '0':
        target_size = 203

    eid2cls = pickle.load(open(os.path.join(args.dataset, args.pkl_eid2cls), 'rb'))

    with open(os.path.join(args.dataset, args.output, args.pretrained_model.split('.')[0], 'summary.txt'), 'w') as file_summary:
        for i in tqdm(range(0, num_query_per_class), total=num_query_per_class):
            print('\n[Test %d]' % (i+1))
            query_set = [query_sets[cls_name][i] for cls_name in class_names]
            # Hyperparamter setting for wiki
            expanded = expan.expand(query_set, target_size=target_size, ranking=True, mu=9, win_grow_rate=2.5, win_grow_step=20)
            AP10s, AP20s, AP50s, AP100s = [[], [], [], []]
            P10s, P20s, P50s, P100s = [[], [], [], []]
            for j, cls in enumerate(class_names):
                with open(os.path.join(args.dataset, args.output, args.pretrained_model.split('.')[0], f'{i}_{cls}.txt'), 'w') as f:
                    AP10, AP20, AP50, AP100 = [apk(gt[cls], expanded[j], n) for n in [10, 20, 50, 100]]
                    P10, P20, P50, P100 = [recall(gt[cls], expanded[j], n) for n in [10, 20, 50, 100]]
                    AP10s.append(AP10)
                    AP20s.append(AP20)
                    AP50s.append(AP50)
                    AP100s.append(AP100)
                    P10s.append(P10)
                    P20s.append(P20)
                    P50s.append(P50)
                    P100s.append(P100)

                    print(AP10, AP20, AP50, AP100, P10, P20, P50, P100, file=f)
                    print('', file=f)
                    for eid in expanded[j]:
                        print(f'{eid}\t{expan.eid2name[eid]}\t\t{" & ".join(eid2cls[eid])}', file=f)

            MAPs[0] += sum(AP10s) / num_class
            MAPs[1] += sum(AP20s) / num_class
            MAPs[2] += sum(AP50s) / num_class
            MAPs[3] += sum(AP100s) / num_class
            Ps[0] += sum(P10s) / num_class
            Ps[1] += sum(P20s) / num_class
            Ps[2] += sum(P50s) / num_class
            Ps[3] += sum(P100s) / num_class
            print('[TEST %d]' % (i + 1), file=file_summary)
            print('MAP %.6f %.6f %.6f %.6f' %
                  (sum(AP10s) / num_class, sum(AP20s) / num_class, sum(AP50s) / num_class, sum(AP100s) / num_class),
                  file=file_summary)
            print('MAP %.6f %.6f %.6f %.6f' %
                  (sum(AP10s) / num_class, sum(AP20s) / num_class, sum(AP50s) / num_class, sum(AP100s) / num_class))
            print('P %.6f %.6f %.6f %.6f\n' %
                  (sum(P10s) / num_class, sum(P20s) / num_class, sum(P50s) / num_class, sum(P100s) / num_class),
                  file=file_summary)
            print('P %.6f %.6f %.6f %.6f\n' %
                  (sum(P10s) / num_class, sum(P20s) / num_class, sum(P50s) / num_class, sum(P100s) / num_class))

        print('\nTotal MAP %.6f %.6f %.6f %.6f' %
              (MAPs[0] / num_query_per_class, MAPs[1] / num_query_per_class,
               MAPs[2] / num_query_per_class, MAPs[3] / num_query_per_class), file=file_summary)
        print('\nTotal MAP %.6f %.6f %.6f %.6f' %
              (MAPs[0] / num_query_per_class, MAPs[1] / num_query_per_class,
               MAPs[2] / num_query_per_class, MAPs[3] / num_query_per_class))
        print('Total P %.6f %.6f %.6f %.6f\n' %
              (Ps[0] / num_query_per_class, Ps[1] / num_query_per_class,
               Ps[2] / num_query_per_class, Ps[3] / num_query_per_class), file=file_summary)
        print('Total P %.6f %.6f %.6f %.6f\n' %
              (Ps[0] / num_query_per_class, Ps[1] / num_query_per_class,
               Ps[2] / num_query_per_class, Ps[3] / num_query_per_class))

    result_file = '../result/{}'.format(args.result)
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            pass
    with open(result_file, 'a') as f:
        print(f'[{args.pretrained_model}]', file=f)
        print('Total MAP %.6f %.6f %.6f %.6f' %
              (MAPs[0] / num_query_per_class, MAPs[1] / num_query_per_class,
               MAPs[2] / num_query_per_class, MAPs[3] / num_query_per_class), file=f)
        print('Total P %.6f %.6f %.6f %.6f\n' %
              (Ps[0] / num_query_per_class, Ps[1] / num_query_per_class,
               Ps[2] / num_query_per_class, Ps[3] / num_query_per_class), file=f)