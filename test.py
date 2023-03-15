import argparse, random, time, os, pdb
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

import np_transforms as NP_T
# from CrowdDataset import TestSeq
from dataset import CrowdSeq
from model import STGN
from sklearn.metrics import mean_squared_error,mean_absolute_error
import scipy.io as sio

def get_seq_class(seq, set):
    backlight = ['DJI_0021', 'DJI_0022', 'DJI_0032', 'DJI_0202', 'DJI_0339', 'DJI_0340']
    # cloudy = ['DJI_0519', 'DJI_0554']
    
    # uhd = ['DJI_0332', 'DJI_0334', 'DJI_0339', 'DJI_0340', 'DJI_0342', 'DJI_0343', 'DJI_345', 'DJI_0348', 'DJI_0519', 'DJI_0544']

    fly = ['DJI_0177', 'DJI_0174', 'DJI_0022', 'DJI_0180', 'DJI_0181', 'DJI_0200', 'DJI_0544', 'DJI_0012', 'DJI_0178', 'DJI_0343', 'DJI_0185', 'DJI_0195']

    angle_90 = ['DJI_0179', 'DJI_0186', 'DJI_0189', 'DJI_0191', 'DJI_0196', 'DJI_0190']

    mid_size = ['DJI_0012', 'DJI_0013', 'DJI_0014', 'DJI_0021', 'DJI_0022', 'DJI_0026', 'DJI_0028', 'DJI_0028', 'DJI_0030', 'DJI_0028', 'DJI_0030', 'DJI_0034','DJI_0200', 'DJI_0544']

    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'
    
    count = 'sparse'
    loca = sio.loadmat(os.path.join('../../ds/dronebird/', set, 'ground_truth', 'GT_img'+str(seq[-3:])+'000.mat'))['locations']
    if loca.shape[0] > 150:
        count = 'crowded'
    return light, angle, bird, size, count

def main():
    parser = argparse.ArgumentParser(
        description='Train CSRNet in Crowd dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='./models/dronebird/STGN.pth', type=str)
    parser.add_argument('--dataset', default='Mall', type=str)
    parser.add_argument('--valid', default=0, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=4, type=int)
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--shape', default=[360, 480], nargs='+', type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_all', action='store_true', help='')
    parser.add_argument('--adaptive', action='store_true', help='')
    parser.add_argument('--agg', action='store_true', help='')
    parser.add_argument('--use_cuda', default=True, type=bool)

    args = vars(parser.parse_args())
    
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    valid_transf = NP_T.ToTensor() 

    datasets = ['dronebird']
    for dataset in datasets:
        if dataset == 'UCSD':
            args['shape'] = [360, 480]
            args['max_len'] = 10
            args['channel'] = 128
        elif dataset == 'Mall':
            args['shape'] = [480, 640]
            args['max_len'] = 4
            args['channel'] = 128
        elif dataset == 'FDST':
            args['max_len'] = 4
            args['shape'] = [360, 640]
            args['channel'] = 128
        elif dataset == 'Venice':
            args['max_len'] = 8
            args['shape'] = [360, 640]
            args['channel'] = 128
        elif dataset == 'TRANCOS':
            args['max_len'] = 4
            args['shape'] = [360, 480]
            args['channel'] = 128
        elif dataset == 'dronebird':
            args['max_len'] = 4
            args['shape'] = [480, 640]
            args['channel'] = 128
            
        dataset_path = os.path.join('../../ds', dataset)
        valid_data = CrowdSeq(mode='test',
                             path=dataset_path,
                             out_shape=args['shape'],
                             transform=valid_transf,
                             gamma=args['gamma'],
                             max_len=args['max_len'], 
                             load_all=args['load_all'])
        valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

        model = STGN(args).to(device)
        model.eval()
        model_name = os.path.join('./models', dataset, 'STGN.pth')
        assert os.path.exists(model_name) is True
        model.load_state_dict(torch.load(model_name))
        print('Load pre-trained model')

        X, density, count = None, None, None
        
        # preds = {}
        predictions = []
        counts = []
        preds = [[] for i in range(10)]
        gts = [[] for i in range(10)]
        for i, (X, count, seq_len, names) in enumerate(valid_loader):
            X, count, seq_len = X.to(device), count.to(device), seq_len.to(device)

            with torch.no_grad():
                density_pred, count_pred = model(X)
        
            N = torch.sum(seq_len)
            count = count.sum(dim=[2,3,4])
            count_pred = count_pred.data.cpu().numpy()
            count = count.data.cpu().numpy()
            print("\r{}/{}".format(i, len(valid_loader)), end='')
            # print(names)
            for i, name in enumerate(names):
                seq = int(os.path.basename(name[0])[3:6])
                seq = 'DJI_' + str(seq).zfill(4)
                light, angle, bird, size, count_bird = get_seq_class(seq, 'test')
                # dir_name, img_name = name[0].split('&')
                pred_e = count_pred[0, i]
                gt_e = count[0, i]
                if light == 'sunny':
                    preds[0].append(pred_e)
                    gts[0].append(gt_e)
                elif light == 'backlight':
                    preds[1].append(pred_e)
                    gts[1].append(gt_e)
                if count_bird == 'crowded':
                    preds[2].append(pred_e)
                    gts[2].append(gt_e)
                else:
                    preds[3].append(pred_e)
                    gts[3].append(gt_e)
                if angle == '60':
                    preds[4].append(pred_e)
                    gts[4].append(gt_e)
                else:
                    preds[5].append(pred_e)
                    gts[5].append(gt_e)
                if bird == 'stand':
                    preds[6].append(pred_e)
                    gts[6].append(gt_e)
                else:
                    preds[7].append(pred_e)
                    gts[7].append(gt_e)
                if size == 'small':
                    preds[8].append(pred_e)
                    gts[8].append(gt_e)
                else:
                    preds[9].append(pred_e)
                    gts[9].append(gt_e)
                # count


                # preds[dir_name + '_' + img_name] = count[0, i]
                
                predictions.append(count_pred[0, i])
                counts.append(count[0, i])

        print()    
        mae = mean_absolute_error(predictions, counts)
        rmse = np.sqrt(mean_squared_error(predictions, counts))
        
        print('Dataset : {} MAE : {:.3f} MSE : {:.3f}'.format(dataset, mae, rmse))

        with open('result.txt', 'w') as f:
            # f.write('max: {}, min: {}\n'.format(max(np.abs(predictions-counts)), min(np.abs(predictions-counts))))
            # print('max: {}, min: {}'.format(max(np.abs(predictions-counts)), min(np.abs(predictions-counts))))
            log_str = 'mae {}, mse {}\n'.format(mae, rmse)
            print(log_str)
            f.write(log_str)
            attri = ['sunny', 'backlight', 'crowded', 'sparse', '60', '90', 'stand', 'fly', 'small', 'mid']
            for i in range(10):
                if len(preds[i]) == 0:
                    continue
                print('{}: MAE:{}. RMSE:{}.'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))
                f.write('{}: MAE:{}. RMSE:{}.\n'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))

        
if __name__ == '__main__':
    main()
