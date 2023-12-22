from torch.autograd import Variable
import numpy as np
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = enc_in
        
        #attention score
        self._attention = torch.ones(self.channels,1)
        self._attention = Variable(self._attention, requires_grad=False)
        self.fs_attention = torch.nn.Parameter(self._attention.data)

        self.IsTest = False
        self.pretrain = False
        self.project = False

        #encoder
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        
        for i in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        #decoder
        self.Decoder_Seasonal = nn.ModuleList()
        self.Decoder_Trend = nn.ModuleList()
        
        for i in range(self.channels):
            self.Decoder_Seasonal.append(nn.Linear(self.pred_len,self.seq_len))
            self.Decoder_Trend.append(nn.Linear(self.pred_len,self.seq_len))  

        self.Decoder_Seasonal_pointwise = nn.Linear(self.seq_len * self.channels, 1)
        self.Decoder_Trend_pointwise = nn.Linear(self.seq_len * self.channels, 1)

        #projector
        self.Proj_Seasonal = nn.ModuleList()
        self.Proj_Trend = nn.ModuleList()
        self.Proj_Seasonal_2 = nn.ModuleList()
        self.Proj_Trend_2 = nn.ModuleList()
        self.activation = nn.PReLU()
        for i in range(self.channels):
            self.Proj_Seasonal.append(nn.Linear(self.pred_len,self.pred_len * 2))
            self.Proj_Trend.append(nn.Linear(self.pred_len,self.pred_len * 2))
            self.Proj_Seasonal_2.append(nn.Linear(self.pred_len * 2,self.pred_len))
            self.Proj_Trend_2.append(nn.Linear(self.pred_len * 2,self.pred_len))
        

    def forward(self, x):
        if self.pretrain:
            x = x.transpose(1, 2)
   
            seasonal_init, trend_init = self.decompsition(x)

            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to("cuda:0")
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to("cuda:0")
            
            if self.project:

                for i in range(self.channels):
                    seasonal_output[:,i,:] = self.Proj_Seasonal_2[i](self.activation(self.Proj_Seasonal[i](self.Linear_Seasonal[i](seasonal_init[:,i,:].clone()))))
                    trend_output[:,i,:] = self.Proj_Trend_2[i](self.activation(self.Proj_Trend[i](self.Linear_Trend[i](trend_init[:,i,:].clone()))))

                x = seasonal_output + trend_output
            else:
                with torch.no_grad():
                    for i in range(self.channels):
                        seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:].clone())
                        trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:].clone())

                    x = seasonal_output + trend_output

            return x.transpose(1, 2)

        # x: [Batch, Input length, Channel]
        x = x.transpose(1, 2)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to("cuda:0")
        trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to("cuda:0")
        
        seasonal_output_1 = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.seq_len],dtype=seasonal_init.dtype).to("cuda:0")
        trend_output_1 = torch.zeros([trend_init.size(0),trend_init.size(1),self.seq_len],dtype=trend_init.dtype).to("cuda:0")
        
        for i in range(self.channels):
            seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:].clone())
            trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:].clone())

        
        seasonal_output = seasonal_output *  F.softmax(self.fs_attention, dim=0)
        trend_output = trend_output * F.softmax(self.fs_attention, dim = 0)
        
        for i in range(self.channels):
            seasonal_output_1[:,i,:] = self.Decoder_Seasonal[i](seasonal_output[:,i,:].clone())
            trend_output_1[:,i,:] = self.Decoder_Trend[i](trend_output[:,i,:].clone())

        if self.IsTest:
            reshape_seasonal = torch.reshape(seasonal_output_1, (1, 1, 32*self.channels))
            reshape_trend = torch.reshape(trend_output_1, (1, 1, 32*self.channels))
        else:
            reshape_seasonal = torch.reshape(seasonal_output_1, (128, 1, 32*self.channels))
            reshape_trend = torch.reshape(trend_output_1, (128, 1, 32*self.channels))
        
        y1 = self.Decoder_Seasonal_pointwise(reshape_seasonal)
        y2 = self.Decoder_Trend_pointwise(reshape_trend)

        x = y1 + y2 
        x = x.transpose(1,2)

        return x

    def setPretrain(self, x):
        self.pretrain = x

    def setProj(self, x):
        self.project = x

    def setTest(self, x):
        self.IsTest = x