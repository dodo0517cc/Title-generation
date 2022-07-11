# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

import os
os.chdir('/content/drive/My Drive/Deep_Learning_final')

!pip install pytorch_transformers

!pip install transformers

import seaborn as sns;
import sys
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nltk
import re
import string
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
import transformers
from transformers import T5Tokenizer
from transformers import T5Model, T5Config,T5ForConditionalGeneration

class Dataset():
    def __init__(self,set_name,train=True):
        self.df = pd.read_csv(set_name)
        self.tokenizer = tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.train_,self.val_ = train_test_split(self.df,train_size=0.8,random_state=42)
        self.train = train
        self.max_length=700
        self.target_length=56

    def __getitem__(self,index): 
        if self.train:
            title,abstract,year = self.train_.iloc[index].values
        else:
            abstract,year = self.val_.iloc[index].values

        text = self.tokenizer.tokenize(abstract)
        if len(text) > self.max_length-2:
            text = text[:self.max_length-2]
        text = ['</s>'] + text 

        ids = self.tokenizer.convert_tokens_to_ids(text)
        text_tensor = torch.tensor(ids)

        labels = self.tokenizer.tokenize(title)
        ids_label = self.tokenizer.convert_tokens_to_ids(labels)
        label_tensor = torch.tensor(ids_label)

        return text_tensor,label_tensor

    def __len__(self):
        if self.train:
          return len(self.train_)
        else:
          return len(self.val_)
    def pad_batch(self,batch):
        # collate_fn for Dataloader, pad sequence to same length and get mask tensor
        if batch[0][1] is not None:
            (text_tokens,labels) = zip(*batch)
            # labels = torch.stack(labels)
            labels = pad_sequence(labels,batch_first=True)
            
        else:
            (text_tokens,_) = zip(*batch)
        texts_tokens_pad = pad_sequence(text_tokens, batch_first=True)
        # segments_pad = pad_sequence(segments, batch_first=True)
        text_masks = torch.zeros(texts_tokens_pad.shape)
        text_len = [len(x) for x in text_tokens]
        for i in range(len(text_masks)):
            text_masks[i][:text_len[i]] = 1
        return texts_tokens_pad,text_masks,labels

class T5_model(nn.Module):
    def __init__(self):
        super(T5_model, self).__init__()

        self.embedding = T5Model.from_pretrained('t5-base')
    def forward(self,ids,masks,decoder_input_ids=None,lm_labels=None):
        output = self.embedding(input_ids=ids,attention_mask=masks,decoder_input_ids=decoder_input_ids,lm_labels=lm_labels)
        # cls_atten = embedded[0][:,0]  #[CLS]
        # # print(embedded[0].shape)
        # # print(embedded[1].shape)
        # # print(cls_atten.shape)
        # # sys.exit()
        # head = embedded[0][0]
        # logits = self.fc(cls_atten)

        return output

import os
os.chdir('/content/')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    learning_rate = 0.0001
    model_name = "T5_model"

    train_set = Dataset("./data.csv",train=True)
    val_set = Dataset("./data.csv",train=False)

    train_loader = DataLoader(train_set,batch_size=batch_size,collate_fn=train_set.pad_batch)
    val_loader = DataLoader(val_set,batch_size=batch_size,collate_fn=val_set.pad_batch)

    # model = T5_model()    
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    chart_data={"train_loss":[],"epoch":[]}
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        train_loss = 0
        count = 0
        correct = 0
        model.train()
        optimizer.zero_grad()
        for step, (batch) in enumerate(train_loader):
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            abstracts,masks,titles= [t.to(device) for t in batch]
            y_ids = titles[:, :-1].contiguous()
            lm_labels = titles[:, 1:].clone().detach()
            lm_labels[titles[:, 1:] == tokenizer.pad_token_id] = -100

            output = model(abstracts,masks,decoder_input_ids=y_ids, labels=lm_labels)
            generated_ids = model.generate(
            input_ids = abstracts,
            attention_mask = masks, 
            max_length=150, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in titles]
            model.zero_grad()
            loss = output[0]
            train_loss += loss
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # generated = sample_sequence(
            #     model,
            #     length=20,
            #     context=abstracts,
            #     segments_tokens=segments,
            #     num_samples=20,
            #     temperature=1,
            #     top_k=0,
            #     top_p=0.0
            #   )
            # torch.save(model.state_dict(), './'+model_name+'.pkl')
            if step % 20 == 0:
              print(preds)
              print('----------------')
              print(target)
              print('----------------')
        print('Epoch: ' , str(epoch) , \
                '\ttrain_loss: '+str(round(train_loss/(step+1),5)),\
            )
        chart_data['epoch'].append(epoch)
        chart_data['train_loss'].append(train_loss/(step+1))

draw_chart(chart_data,model_name)

