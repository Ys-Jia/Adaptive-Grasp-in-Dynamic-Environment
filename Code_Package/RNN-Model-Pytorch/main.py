#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from encoder import Encoder
#from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST, MovingObstacles, MovingObstaclesJSON
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
import random
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
import argparse
import json
from seq2seq import Encoder, Decoder

TIMESTAMP = "2021-12-17T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=3,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=1,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


save_dir = './save_model/' + TIMESTAMP

with open('../dataset4.json', 'r') as f:
    depth_map = json.load(f)

folder_list = list(depth_map.keys())
indices = [i for i in range(len(folder_list)) if folder_list[i].split('_')[1] != "block" and folder_list[i].split('_')[1] != "teddy"]
vector_list = list(depth_map.values())
vector_list = [vector_list[idx] for idx in indices]
train_list = vector_list[:600]
test_list = vector_list[600:]
del depth_map

diff = np.zeros((1, 10))
count = 0
for i in range(len(vector_list)):
    vectors = list(vector_list[i].values())
    for j in range(len(vectors) - 1):
        diff += (np.asarray(vectors[j + 1]) - np.asarray(vectors[j]))
        count += 1
ratio = np.abs(diff/count) * 1000

trainFolder = MovingObstaclesJSON(
                            train_list,
                            speed_scale=1,
                            is_train=True,
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output,
                            ratio=ratio
                            )

validFolder = MovingObstaclesJSON(
                            test_list,
                            speed_scale=1,
                            is_train=False,
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output,
                            ratio=ratio
                            )
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=True)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

out = next(iter(trainLoader))


def train():
    '''
    main function to run the training
    '''
    LR = 0.001
    DECODER_LEARNING_RATIO = 5.0
    
    encoder = Encoder(10, 512, args.frames_input, True)
    decoder = Decoder(10, 512 * 2)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    
    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        # model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        # net.load_state_dict(model_info['state_dict'])
        # optimizer = torch.optim.Adam(net.parameters())
        # optimizer.load_state_dict(model_info['optimizer'])
        # cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().to(device)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
    enc_pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(enc_optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)
    dec_pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(dec_optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            encoder.train()
            decoder.train()
            
            start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1)
            input_lengths = (torch.ones(inputs.size()[0])*inputs.size()[1]).cpu()
            target_lengths = (torch.ones(label.size()[0])*label.size()[1]).cpu()

            output, hidden_c = encoder(inputs, input_lengths)
            preds = decoder(start_decode, hidden_c, label.shape[1], output, None, is_training=True)
            
            loss = lossfunction(preds, label)
            loss.backward()
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            #torch.nn.utils.clip_grad_value_(.parameters(), clip_value=10.0)
            enc_optimizer.step()
            dec_optimizer.step()
            
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        #tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                encoder.train()
                decoder.train()
                
                start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1)
                input_lengths = (torch.ones(inputs.size()[0])*inputs.size()[1])
                target_lengths = (torch.ones(label.size()[0])*label.size()[1])

                output, hidden_c = encoder(inputs, input_lengths)
                preds = decoder(start_decode, hidden_c, label.shape[1], output, None, is_training=False)
                loss = lossfunction(preds, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        #tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        enc_pla_lr_scheduler.step(train_loss)
        dec_pla_lr_scheduler.step(train_loss)# lr_scheduler
        model_dict = {
            'epoch': epoch,
            'enc_state_dict': encoder.state_dict(),
            'dec_state_dict': decoder.state_dict(),
            'enc_optimizer': enc_optimizer.state_dict(),
            'dec_optimizer': dec_optimizer.state_dict()
        }
        early_stopping(train_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
