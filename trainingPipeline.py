import sys; sys.path.append("./"), sys.path.append("../")
import os
import gc
import time
import torch
import torchaudio
import numpy as np
import torchvision
import torch.nn as nn
from torch import optim 
from typing import Tuple
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributions import normal
from torch.utils.data import DataLoader
# from Architectures.LDMmodels import q_sample,sample

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.cuda.is_available()

class TrainingPipe:
  
    def __init__(self,
                 experiment_name,
                 dataset,
                 criterion,
                 speech2textModel,
                 textEncoder,
                 optimizer,
                 model,               
                 config:dict) -> None:
        
        self.experiment_name = experiment_name
        path = os.path.join(os.path.expanduser("~"), "Trained/Logs")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, f"{experiment_name}.txt")
        self.config = config
        self.device = self.config['device']
        self.model = model
        self.dataset = dataset
        self.speech2textModel = speech2textModel
        # self.Clayer = nn.ConvTranspose1d(80,80,60,dilation=6)
        self.textEncoder = textEncoder
        # self.flatten = nn.Flatten()
        self.criterion = criterion
        self.criterion1 = criterion
        self.optimizer = optimizer
        # self.extrapolate = nn.Linear(512,885)
        self.intrapolate = nn.LazyLinear(28)
 
    def log(self, *args):
        with open(self.log_path, "a") as F:
            F.write(" ".join([str(i) for i in args]))
            F.write("\n")
        F.close()    
    
    def p_losses(self,original, predicted, loss_type:str="l1"):

            if loss_type == 'l1':
                loss = F.l1_loss(original, predicted)
            elif loss_type == 'l2':
                loss = F.mse_loss(original, predicted)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(original, predicted)
            else:
                raise NotImplementedError()

            return loss
    
    def TrainingPipeline(self, minibatch):
        
        CropPic, DiseaseName, DiseaseDescription = minibatch
        #CropPic == (BS,28,28)
        textTokens = self.textEncoder(DiseaseDescription)
        #textTokens == (BS,textLength)
        textTokens = self.intrapolate(textTokens).unsqueeze(1)
        #textTokens == (BS,1,28)
        print(textTokens.shape)
        
        FusedPicText = torch.cat((CropPic,textTokens),dim=1)
        print(FusedPicText.shape) # == (BS,29,28)

        PredictedDisease = self.model(FusedPicText.unsqueeze(1)).squeeze()
            
        
        l1 =  self.p_losses(DiseaseName,PredictedDisease,'l2')
        print(l1)
        
        # # Architectures/GaussianNoiseonENgS_LDM_HinLossMSE
        # print(GaussianNoisedEng.shape,t.shape)
        # Output = self.unet(GaussianNoisedEng,t)
        # print(Output.shape)
        # l1 =  self.p_losses(Hdata,Output,'l2')
        # print(l1)
        
        # # Architectures/H_sample(x_start=Edata, t=t, noise=Hdata)#noising the english spectrogam with hindi spectrogram acc to random timestep 
        # print(HindiNoisedEng.shape,t.shape)
        # PredictedHinNoise = self.unet(HindiNoisedEng,t)
        # # print(Output.shape)indiNoiseonENgS_removeHinNoise_huber
        # HindiNoisedEng = q
        l2 =  self.p_losses(DiseaseName,PredictedDisease,'l2')
        print(l2)
        # OutputNoise = PredictedHinNoise        
                
        loss = l1+l2
        print(loss)

        loss = l1
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss#,audio
        
        
    def dataloader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.config["batch_size"],
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
        

    def train_1_epoch(self, epoch):

        self.unet.train()
     
        self.criterion.train()
        self.criterion1.train()
        dataloader = self.dataloader()
        
        start_time = time.time()
        epoch_loss = 0 
        
        for minibatch_idx, minibatch in enumerate(dataloader):
            
            loss = self.pipeline(minibatch)

            epoch_loss += loss.detach().item()
                
            torch.cuda.empty_cache()
            gc.collect()
            
            self.log(f"Epoch - {epoch}",
                    f"minibatch idx - {minibatch_idx}",
                    f"Loss - {(epoch_loss/len(dataloader)):.4f}",
                    f"Time Taken - {((time.time()-start_time)/36000):.4f}")       

            print(f"Epoch - {epoch} minibatch idx - {minibatch_idx} Loss - {(epoch_loss/len(dataloader)):.4f} Training - {(100 * (minibatch_idx/dataloader.__len__())):.4f}")
            
            samples = sample(self.unet, image_size=image_size, batch_size=6, channels=1)

            # show a random one
            random_index = 5
            plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
                        
                        
        for sno,i in enumerate(samples.unsqueeze(1)) : 
            path = '/home/earth/Architectures/'+ self.experiment_name + '/' +str(epoch)+ '/'
            if not os.path.exists(path):
                os.makedirs(path)
            print(i.shape,'{{{}}}')
        
        self.save_checkpoints(epoch)     
        return reconst_spec
# cropPic = torch.randn((5,80,400))
# textTokens = torch.randn((5,1,400))

# FusedPicText = torch.cat((cropPic,textTokens),dim=1)
# print(FusedPicText.shape)
