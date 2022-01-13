import os
import torch
from torch import nn
import torch.optim as optim
import sys
import random
import numpy as np
from seq2seq import Encoder, Decoder

def load_model(frames_input, save_dir='./save_model/'+"2021-12-06T00-00-00", model_name='checkpoint_296_0.000000.pth.tar'):
    encoder = Encoder(3, 512, frames_input, True)
    decoder = Decoder(3, 512 * 2)

    model_info = torch.load(os.path.join(save_dir, model_name))
    encoder.load_state_dict(model_info['enc_state_dict'])
    decoder.load_state_dict(model_info['dec_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


def inference(encoder, decoder, inputs, frames_output):
    inputs = torch.unsqueeze(torch.tensor(inputs).float(), dim=0)
    start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1)
    input_lengths = (torch.ones(inputs.size()[0])*inputs.size()[1]).cpu()

    output, hidden_c = encoder(inputs, input_lengths)
    preds = decoder(start_decode, hidden_c, frames_output, output, None, is_training=False)
    return torch.squeeze(preds + torch.unsqueeze(inputs[:, 0, :], dim=1), dim=0)

encoder, decoder = load_model(3, save_dir='./save_model/'+"2021-12-06T00-00-00", model_name='checkpoint_296_0.000000.pth.tar')

inputs = np.ones((3, 3))
pred = inference(encoder, decoder, inputs, 1)
print(pred.shape)
print(pred)