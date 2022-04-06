#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# yangyi
# Function: direction-aware blstm separation with IRM

import torch
import torch.nn as nn
from feature import FeatureExtractor
from wcrnn import WCRNN

class WDASnetSerial(nn.Module):
    def __init__(self):
        super(WDASnetSerial, self).__init__()
        self.model1 = WCRNN()
        self.model2 = BLSTM_MASK()
    def forward(self, x):
        est_doa = self.model1(x)
        output = self.model2(x, est_doa)
        return est_doa, output

class BLSTM_MASK(nn.Module):
    """The wrapper for the weighted-direction-aware speech separation model
        
    Forward Input Arguments
    ----------
    x: tensor, mixed speech, b x c x n
    doa: tensor, estimated doa of the main speaker, b x 3 x 2
        
    Example
    -----
    >>> model = BLSTM_MASK()
    >>> inp = torch.rand(1, 7, 16000)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 2, 16000])
    """
    
    def __init__(self, 
                 frame_len=512, 
                 frame_hop=128, 
                 channel=7, 
                 hidden_size=512, 
                 bidirectional=True, 
                 num_layers=3, 
                 do_doa=True
                ):
        super(BLSTM_MASK,self).__init__()
        self.channel = channel
        num_bins = frame_len // 2 + 1
        self.num_bins = num_bins
        self.do_doa = do_doa
        self.extractor = FeatureExtractor(frame_len=frame_len, frame_hop=frame_hop, do_ipd=True)
        if (do_doa):
            self.bn = nn.BatchNorm1d(num_bins * (channel+1))
        else:
            self.bn = nn.BatchNorm1d(num_bins * channel)
        self.ipd_layer = nn.LSTM(input_size=num_bins * 2, 
                                 hidden_size=num_bins, 
                                 bidirectional=bidirectional, 
                                 num_layers=1, 
                                 batch_first=True
                                )

        if (do_doa):
            self.lstm = nn.LSTM(input_size=num_bins * 3, 
                                hidden_size=hidden_size, 
                                dropout=0.4, 
                                bidirectional=bidirectional, 
                                num_layers=num_layers, 
                                batch_first=True
                               )
        else:
            self.lstm = nn.LSTM(input_size=num_bins * 2, 
                                hidden_size=hidden_size, 
                                dropout=0.4, 
                                bidirectional=bidirectional, 
                                num_layers=num_layers, 
                                batch_first=True
                               )

        self.mask_layer = nn.Linear(hidden_size * 2, num_bins * 2)
        self.mask_act = nn.Sigmoid()

        self.ipd_layer.flatten_parameters()
        self.lstm.flatten_parameters()

    def forward(self, x, doa):
        doa = torch.atan2(doa[:,1], doa[:,0])
        # mag & pha [b c f t] feat [b (c+1) t]
        mag, pha, feat = self.extractor(x, doa=doa)
        feat = self.bn(feat).permute(0,2,1)
        B, T, _ = feat.size()

        feats = []
        for i in range(self.channel-1):
            # [b t 2f]
            feats.append(torch.cat((feat[:,:,:self.num_bins], feat[:,:,(i+1)*self.num_bins:(i+2)*self.num_bins]), -1))
        ipd_layer_outs = []
        for j in range(self.channel-1):
            ipd_layer_out, _ = self.ipd_layer(feats[j])
            ipd_layer_outs.append(ipd_layer_out)
        # [b t 2f]
        ipd_layer_out = torch.stack(ipd_layer_outs, -1).mean(-1)
        if (self.do_doa):
            # [b t 3f]
            ipd_layer_out = torch.cat((ipd_layer_out, feat[:,:,self.channel*self.num_bins:]), -1)

        lstm_out, _ = self.lstm(ipd_layer_out)
        mask_out = self.mask_layer(lstm_out).view(B,T,2,self.num_bins)
        # [b s f t]
        mask_out = mask_out.permute(0,2,3,1)
        mask_out = self.mask_act(mask_out)

        # [b f t]
        mag_est1 = mask_out[:,0] * mag[:,0]
        mag_est2 = mask_out[:,1] * mag[:,0]
        x_est1 = self.extractor.istft(mag_est1, pha[:,0])
        x_est2 = self.extractor.istft(mag_est2, pha[:,0])

        output = torch.stack((x_est1, x_est2),1)
        output = nn.functional.pad(output,[0,x.shape[-1]-output.shape[-1]])

        return output

