import argparse
import json
import pickle
import cv2
import csv
import os
import numpy as np
import torch
from torch.nn.functional import adaptive_max_pool2d
from ipdb import set_trace
from networks.under_the_radar import UnderTheRadar
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath

'''
python3 desc.py --config ckpt-3/nuScenes.json --pretrain ckpt-3/latest.pt
'''

if __name__ == '__main__':
    # ---------------------------------------- load model ---------------------------------------- #
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    root = get_folder_from_file_path(args.pretrain)
    seq_nums = config['test_split']
    if config['model'] == 'UnderTheRadar':
        model = UnderTheRadar(config).to(config['gpuid'])
    assert (args.pretrain is not None)
    checkpoint = torch.load(args.pretrain, map_location=torch.device(config['gpuid']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # ------------------------------------- output embeddings ------------------------------------ #
    IMG_DIR = '/workspace/robocar/7n5s_xy11/img'
    CSV_DIR = '/workspace/robocar/7n5s_xy11'

    database_list = []
    csv_reader = csv.reader(open(os.path.join(CSV_DIR, 'database.csv')))
    for index, line in enumerate(csv_reader):
        if index == 0:
            continue
        database_list.append(os.path.join(IMG_DIR, '{:0>5d}.jpg'.format(int(line[0]))))

    query_list = []
    csv_reader = csv.reader(open(os.path.join(CSV_DIR, 'test.csv')))
    for index, line in enumerate(csv_reader):
        if index == 0:
            continue
        query_list.append(os.path.join(IMG_DIR, '{:0>5d}.jpg'.format(int(line[0]))))

    dbFeat = np.empty((len(database_list), 248))
    for index, img in enumerate(tqdm(database_list)):
        # set_trace()
        data = cv2.imread(img)[:, :, 0:1]
        data = np.transpose(data, [2, 0, 1])
        data = torch.from_numpy(data).type(torch.FloatTensor)
        data = torch.unsqueeze(data, 0)
        data = data.to(config['gpuid'])
        _, _, desc = model.unet(data)
        max_desc = adaptive_max_pool2d(input=desc, output_size=1)
        max_desc = torch.squeeze(max_desc, -1)
        max_desc = torch.squeeze(max_desc, -1)
        dbFeat[index, :] = max_desc.detach().cpu().numpy()
        dbFeat = dbFeat.astype('float32')

    qFeat = np.empty((len(query_list), 248))
    for index, img in enumerate(tqdm(query_list)):
        # set_trace()
        data = cv2.imread(img)[:, :, 0:1]
        data = np.transpose(data, [2, 0, 1])
        data = torch.from_numpy(data).type(torch.FloatTensor)
        data = torch.unsqueeze(data, 0)
        data = data.to(config['gpuid'])
        _, _, desc = model.unet(data)
        max_desc = adaptive_max_pool2d(input=desc, output_size=1)
        max_desc = torch.squeeze(max_desc, -1)
        max_desc = torch.squeeze(max_desc, -1)
        qFeat[index, :] = max_desc.detach().cpu().numpy()
        qFeat = qFeat.astype('float32')

    with open('utr_feature.pickle', 'wb') as f:
        feature = {'qFeat': qFeat, 'dbFeat': dbFeat}
        pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved pickle')
