import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter, OrderedDict
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
random.seed(1024)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1,bidirec=False):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        if bidirec:
            self.n_direction = 2 
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
    
    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size))
        #return hidden.cuda()
        return hidden
    
    def init_weight(self):
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
    
    def forward(self, inputs, input_lengths):
        """
        inputs : B, T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs)
        
        #packed = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        outputs, hidden = self.gru(inputs, hidden)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # unpack (back to padded)
                
        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Define the layers
        self.dropout = nn.Dropout(dropout_p)
        
        self.gru = nn.GRU(input_size + hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) # Attention
    
    def init_hidden(self,inputs):
        hidden = Variable(torch.zeros(self.n_layers, inputs.size(0), self.hidden_size))
        #return hidden.cuda()
        return hidden
    
    
    def init_weight(self):
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform(self.attn.weight)
#         self.attn.bias.data.fill_(0)
    
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)
        
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len, -1) # B,T,D
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T
        
#         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
#             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking
        
        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D
        
        return context, alpha
    
    
    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        max_length : int, max length to decode # for batch
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        is_training : bool, this is because adapt dropout only training step.
        """
        # Get the embedding of the current input word
        embedded = inputs
        hidden = self.init_hidden(inputs)
        if is_training:
            embedded = self.dropout(embedded)
        
        decode = []
        # Apply GRU to the output so far
        for i in range(max_length):
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            score = torch.unsqueeze(score, dim=1)
            #softmaxed = F.log_softmax(score, 1)
            decode.append(score)
            embedded = score
            if is_training:
                embedded = self.dropout(embedded)
            
            # compute next context vector using attention
            context, alpha = self.Attention(hidden, encoder_outputs, encoder_maskings)
            
        #  column-wise concat, reshape!!
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0), max_length, -1)
    
    def decode(self, context, encoder_outputs):
        output_length = 5
        
        start_decode = Variable(torch.rand(1, 1, 15))
        hidden = self.init_hidden(start_decode)
        
        decodes = []
        attentions = []
        decoded = start_decode
        for i in range(output_length):
            _, hidden = self.gru(torch.cat((decoded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            context, alpha = self.Attention(hidden, encoder_outputs,None)
            attentions.append(alpha.squeeze(1))
        
        return torch.cat(decodes).max(1)[1], torch.cat(attentions)

# encoder = Encoder(15, 512, 3, True).cuda()
# decoder = Decoder(15, 512 * 2).cuda()
# encoder.init_weight()
# decoder.init_weight()

# inputs = torch.rand(16, 3, 15).cuda()
# targets = torch.rand(16, 3, 15).cuda()
# input_lengths = (torch.ones(16)*3)
# target_lengths = (torch.ones(16)*3)

# start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1).cuda()
# output, hidden_c = encoder(inputs, input_lengths)
# preds = decoder(start_decode, hidden_c, targets.shape[1], output, None, is_training=True)
# print(preds.shape)

