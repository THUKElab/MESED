import os
import argparse
import pickle
from utils import *
from PIL import Image
from transformers import BertTokenizer
import torch
import torchvision
import torchvision.transforms as transforms 
from einops import rearrange

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='../data', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-pkl_e2s', default='entity2sents', help='name of entity2sents folder')
    parser.add_argument('-path_num_sents', default='num_sents.txt')
    parser.add_argument('-max_len', default=150, help='max sentence len')
    args = parser.parse_args()

    model_resnet50 = torchvision.models.resnet50(pretrained=True)
    modules = list(model_resnet50.children())[:-1]      # delete the last fc layer.
    model_resnet50 = torch.nn.Sequential(*modules)

    model_resnet50.cuda()
    model_resnet50.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    mask_token = tokenizer.mask_token

    eid2name, vocab, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))

    entity2sents = dict()
    for eid in vocab:
        entity2sents[eid] = []

    filename = os.path.join(args.dataset, args.sent)
    total_line = get_num_lines(filename)
    image_size = 192
    patch_size = 32

    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_line):
            obj = json.loads(line)
            if len(obj['entityMentions']) == 0 or len(obj['tokens']) > args.max_len:
                continue
            raw_sent = [token.lower() for token in obj['tokens']]
            try:
                raw_image = obj['image']
                image = Image.open(raw_image).convert("RGB").resize((image_size, image_size))
                image = np.array(image)
                tf = transforms.Compose([transforms.ToTensor()])
                image = tf(image)
                image = rearrange(image, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=patch_size, p2=patch_size).cuda()
                image_feature = model_resnet50(image).detach().cpu()
                image_feature = rearrange(image_feature, 'n f 1 1 -> n f')

                for entity in obj['entityMentions']:
                    eid = entity['entityId']
                    sent = copy.deepcopy(raw_sent)
                    sent[entity['start']:entity['end'] + 1] = [mask_token]
                    entity2sents[eid].append((tokenizer.encode(sent), image_feature))
            except Exception as e:
                pass

    drop_eids = []
    with open(args.path_num_sents, 'w') as f:
        print(len(entity2sents), file=f)
        print('', file=f)

        total_size = 0
        cnt = 0
        for eid in entity2sents:
            siz = len(entity2sents[eid])
            if siz < 2:
                drop_eids.append(eid)
            else:
                total_size += siz
                print('%d\t%s\t%d' % (cnt, eid2name[eid], siz), file=f)
                cnt += 1

        print('\nTotal entities %d' % cnt, file=f)
        print('Total sents %d' % total_size, file=f)

    for eid in drop_eids:
        entity2sents.pop(eid)
    
    for eid in tqdm(entity2sents):
        sents = []
        images = []
        tuples = entity2sents[eid]
        for sent, image in tuples:
            sents.append(sent)
            images.append(image)
        entity2sents[eid] = (sents, images)

    print(len(entity2sents))

    if not os.path.exists(os.path.join(args.dataset, args.pkl_e2s)):
        os.mkdir(os.path.join(args.dataset, args.pkl_e2s))

    for eid in tqdm(entity2sents):
        sents = entity2sents[eid][0]
        images = entity2sents[eid][1]
        pickle.dump((sents, images), open(os.path.join(args.dataset, args.pkl_e2s, str(eid)+'.pkl'), 'wb'))

