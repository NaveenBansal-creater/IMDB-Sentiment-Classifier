
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

class UserActivityModel(nn.Module):
    
    def __init__(self, n_vocab, embedding_size, num_layers,
                 hidden_state_size, output_size,drop_prob=0.3,use_gpu=True):
            
            super().__init__()
            self.use_gpu = use_gpu
            self.num_layers = num_layers
            self.drop_prob = drop_prob
            self.n_vocab = n_vocab
            self.hidden_state_size = hidden_state_size
            self.embedding = nn.Embedding(n_vocab, embedding_size)
            
            self.lstm = nn.LSTM(input_size = embedding_size,
                            hidden_size = hidden_state_size,
                            num_layers = num_layers,    
                            dropout=drop_prob,    
                            batch_first=True)
            self.dense = nn.Linear(hidden_state_size, output_size)
            self.softmax = nn.Softmax(dim=1)
     
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        lstm_output, state = self.lstm(embed, prev_state)
  
        # take the last lstm output after consuming all the word vectors
        x = lstm_output[:,-1,:]
        out = self.dense(x) 
        soft_out = self.softmax(out)
        return soft_out
    
        #return out, state, lstm_output[:,-1,:]
    
    def zero_state(self, batch_size):
        
        if self.use_gpu:
            return (torch.zeros(self.num_layers,batch_size,self.hidden_state_size).cuda(),
                     torch.zeros(self.num_layers,batch_size,self.hidden_state_size).cuda())
        else:    
            return (torch.zeros(self.num_layers,batch_size,self.hidden_state_size),
                    torch.zeros(self.num_layers,batch_size,self.hidden_state_size))


