#!/user/bin/env python
# yangyi
# Function: weighted doa estimation

import torch
import torch.nn as nn
import numpy as np
from sigprocess import STFT, ISTFT

class WCRNN(nn.Module):
    def __init__(self, FILTER_LENGTH = 512, WINDOW_LENGTH = 400, HOP_LENGTH = 100, channel = 7, dropout = 0):
        super(WCRNN, self).__init__()
        self.channel = channel
        self.stft = STFT(fftsize=FILTER_LENGTH, window_size=WINDOW_LENGTH, stride=HOP_LENGTH, trainable=False)

        self.input_conv_layer1 = nn.Sequential(
            nn.Conv2d(2*channel,64,[3,3],[1,1],padding=[1,1]),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d([8,1]),
            nn.Dropout(p=dropout),
        )
        self.input_conv_layer2 = nn.Sequential(
            nn.Conv2d(64,64,[3,3],[1,1],padding=[1,1]),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d([4,1]),
            nn.Dropout(p=dropout),
        )
        self.input_conv_layer3 = nn.Sequential(
            nn.Conv2d(64,64,[3,3],[1,1],padding=[1,1]),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d([4,1]),
            nn.Dropout(p=dropout),
        )
        self.blstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128,429),
            nn.Linear(429,4),
        )

    def forward(self, x):       # x:(b,c,n)
        xs = self.stft(x[:,[0],:])[...,1:,:].unsqueeze(1)
        for i in range(1,self.channel):
            xs = torch.cat((xs,self.stft(x[:,[i],:])[...,1:,:].unsqueeze(1)),1) # xs:(b,c,t,f,(real+imag))

        feat = torch.cat((xs[...,0], xs[...,1]), 1).permute(0,1,3,2)            # feat:(b,2c,f,t)

        x = self.input_conv_layer1(feat)
        x = self.input_conv_layer2(x)
        x = self.input_conv_layer3(x)
        x = x.permute(0,2,1,3).contiguous().view(x.size(0),-1,128)
        x, _ = self.blstm(x)

        y = self.fc_layer(x)   # y:(b,t,4)
        y = y[:,:,:3] * y[:,:,[-1]].softmax(dim=-2)

        return y.sum(1)        # (b,3)
        
