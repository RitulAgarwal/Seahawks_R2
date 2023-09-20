import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoTokenizer, RobertaModel
import warnings 
warnings.filterwarnings('ignore')
#Frame wise input to transformer block but with splits according to actual word length


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model= 256, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
       
        self.pe[:, 0::2] = torch.sin(position * div_term)#for even terms sin 
        self.pe[:, 1::2] = torch.cos(position * div_term)#for odd terms cos
       
        self.b = nn.Linear(in_features=d_model, out_features=d_model)
        self.c = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, t) :
        return self.c(self.b(self.pe[:t.size(0)]))

class MHAVEc(nn.Module):
    def __init__(self,embVecSize,cross= False):
        super().__init__()
        # self.state = nn.Linear(C,embVecSize)
        self.emb = embVecSize
        self.state = nn.LazyLinear(self.emb)
        self.cross = cross
        
    def forward(self,InputEmbedding,encoderOutput=None):
        query = self.state(InputEmbedding)
        key = self.state(InputEmbedding)
        value = self.state(InputEmbedding)
        if self.cross : 
            assert encoderOutput != None
            key = encoderOutput
            value = encoderOutput
            
        multihead_attn = nn.MultiheadAttention(self.emb, 8)
        attn_output, _ = multihead_attn(query, key, value)
        return attn_output
      
class Encoder(nn.Module):
    def __init__(self,embeddingSize):
        super().__init__()
        self.mha = MHAVEc(embeddingSize)
        self.AddNorm = nn.LayerNorm(embeddingSize)
        self.FeedForward = nn.Sequential(
            nn.Linear(embeddingSize,80),
            nn.ReLU(),
            nn.Linear(80,embeddingSize)
        )
    def forward(self,input):
        o1 = self.mha(input)
        o2 = self.AddNorm(o1+input)
        o3 = self.FeedForward(o2)
        o4 = self.AddNorm(o3+o2)
        return o4

class Decoder(nn.Module):
    def __init__(self ,embeddingSize):
        super().__init__()
        self.mhaCross = MHAVEc(embeddingSize,cross = True)
        self.mha = MHAVEc(embeddingSize)
        self.AddNorm = nn.LayerNorm(embeddingSize)
        self.FeedForward = nn.Sequential(
            nn.Linear(embeddingSize,80),
            nn.ReLU(),
            nn.Linear(80,embeddingSize)
        )
    def forward(self,x,encoderOuput):
        o1 = self.mha(x)
        # print(o1.shape)
        # print(x.shape)
        o2 = self.AddNorm(o1+x)
        # print(o2.shape)
        o3 = self.mhaCross(o2,encoderOuput)
        # print(o3.shape)
        o4 = self.AddNorm(o2+o3)
        # print(o4.shape)
        o5 = self.FeedForward(o4)
        # print(o5.shape)
        o6 = self.AddNorm(o5+o4)
        # print(o6.shape)
        return o6
    
class TransFrameWise(nn.Module):
    """ Frame wise input to transformer block but with splits according to actual word length """
    def __init__(self,embVecSize= 256,RobertaWordEmbSize = 768,Encoder=Encoder,Decoder=Decoder,training=True,inference = False):
        super().__init__()
        self.training = training 
        self.inference = inference
        self.embVecSize = embVecSize
        self.dense_1 = nn.LazyLinear(self.embVecSize)
        self.dense11 = nn.LazyLinear(self.embVecSize)
        self.dense_2 = nn.Linear(self.embVecSize,1)
        self.act_fn = nn.SiLU()
        self.RobertaWordEmbSize = RobertaWordEmbSize
        self.lossf = nn.MSELoss()
        self.EncSplitsReq = []
        self.dense_3 = nn.Linear(80,1)
        self.enc = Encoder(self.embVecSize)
        self.dec = Decoder(self.embVecSize)
        self.IPpositionalEncoding = PositionalEncoding(self.embVecSize)
        
    def forward(self,TargetSequnce,input):
        decoderInput = []
        for i in TargetSequnce:
            self.EncSplitsReq.append(len(i.split()))
            inputs = tokenizer(i, return_tensors="pt")
            outputs = model(**inputs) 
            decoderInput.append(outputs[1])
        decIn = torch.stack(decoderInput)
        decoderBatchedinput = []
        for ind,i in enumerate(decIn):
            words = self.EncSplitsReq[ind]
            print(words)
            outputEmbs =  torch.split(i,int(self.RobertaWordEmbSize/words),dim=1)
            o = torch.cat(outputEmbs)
            _,e = o.shape
            if e < self.embVecSize:
                o = F.pad(o,(0,self.embVecSize-e))
            else:
                o = self.dense_1(o)
            decoderBatchedinput.append(o)       
            print(o.shape,'decoder input for all sequences')
        bs,C,_ = input.shape
        inputs = []
        for ind,a in enumerate(input):
            #diving input spec to frames
            i = a.view(self.EncSplitsReq[ind],C,-1)
            _,_,e = i.shape
            # i = i.repeat(1,1,int(self.embVecSize/e))
            if e < self.embVecSize:
                i = F.pad(i,(0,self.embVecSize-e))
            else:
                i = self.dense11(i)
            i += self.dense_2(self.act_fn(self.IPpositionalEncoding(i)))[:, :,  None]
            b = self.enc(i) 
            print(b.shape,'encoder output for all squences')
            b = self.dense_3(b.permute(0,2,1)).squeeze()
            print(b.shape,'encoder hidden state for all squences')
            inputs.append(b)
        ##inputs mei encoder ki hidden states hai framewise and decoderBatchedinput mei decoder to be input word emebdding hai word by word        
        dec = Decoder(self.embVecSize)
        #for training time 
        for i in range(bs):
            words,_ = decoderBatchedinput[i].shape
            print(words)
            for j in range(words):
                if self.training:
                    TextEMbedding = decoderBatchedinput[i][j].unsqueeze(0)
                    EncOUT = inputs[i][j].unsqueeze(0)
                    predicted = dec(TextEMbedding,EncOUT)
                    print(predicted.shape)
                    if (j+1 < words):
                        actual = decoderBatchedinput[i][j+1]
                        loss = self.lossf(actual,predicted)
                        print(loss)
                    # loss.backward()
                # #for inference time 
                if self.inference:
                    sos = torch.randn((1,256))
                    EncOUT = inputs[i][j].unsqueeze(0)
                    pred1 = dec(sos,EncOUT)  
                    subPreds = pred1 
                    print(subPreds.shape)
                    # for i in range(words-1):
                    # EncOUT = EncoderHiddenState[i]   
                    # pred1 = dec(sos,EncOUT)    
                    # subPreds = pred1 
                    # print(subPreds.shape)
                    # for i in range(words-1):
                    #     EncOUT = EncoderHiddenState[i+1] #3,256
                    #     print(subPreds.shape,'-*syb')
                    #     pred = dec(subPreds,EncOUT)
                    #     print(pred.shape,'pred')
                    #     subPreds = torch.cat((subPreds,pred))
                

BatchTargetSequence = ['my name is ritul and i live here','i study math here','hi hello']
input = torch.randn((3,80,800))
        
