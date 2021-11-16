import json
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import save_metric_plot

import matplotlib.pyplot as plt
import seaborn as sns

class RoBert(nn.Module):

    
    def __init__(self, num_labels, bert_pretrained="avichr/heBERT", device="cuda", lstm_hidden_dim=32, dense_hidden_dim=16, 
        num_lstm_layers=1, num_dense_layers=1):
        super(RoBert, self).__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(bert_pretrained)
        config = AutoConfig.from_pretrained(bert_pretrained)
        self._petrained_language_model = AutoModel.from_config(config)
        self.device = device
        self.num_labels = num_labels
        self.size_petrained = self._petrained_language_model.config.hidden_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(self.size_petrained, lstm_hidden_dim, num_lstm_layers, bias=True, batch_first=True)
        self.dropout_plm = nn.Dropout(self._petrained_language_model.config.hidden_dropout_prob)
        self.dropout_lstm = nn.Dropout(self._petrained_language_model.config.hidden_dropout_prob)
        fc_layers = []
        prev_dim = lstm_hidden_dim
        for _ in range(num_dense_layers):
            # init dense layers
            linear_ = nn.Linear(prev_dim, dense_hidden_dim)
            linear_.weight.data.normal_(mean=0.0, std=self._petrained_language_model.config.initializer_range)
            linear_.bias.data.zero_()
            fc_layers.append(linear_)
            prev_dim = dense_hidden_dim
        fc_layers.append(nn.Linear(prev_dim, self.num_labels))
        self.fc = nn.Sequential(*fc_layers)
    
    def _generate_init_hidden_state(self, batch_size):
        h0 = Variable(torch.zeros((self.num_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device))
        c0 = Variable(torch.zeros((self.num_lstm_layers, batch_size, self.lstm_hidden_dim), requires_grad=False).to(self.device))
        return (h0, c0)

    def forward(self, x):
        
        # possible keys: input_ids, attention_mask, token_type_ids, num_segments, labels
        input_id, attention_mask, token_type_ids, num_segments = x['input_ids'], x['attention_mask'], x['token_type_ids'], x['num_segments']
        # [num_sentences, num_segments, segment_size] => [total_segments, segment_size]
        num_sentences, max_segments, segment_length = input_id.size()

        total_segments = num_sentences * max_segments
        input_id_ = input_id.view(total_segments, segment_length)
        attention_mask_ = attention_mask.view(total_segments, segment_length)
        token_type_ids_ = token_type_ids.view(total_segments, segment_length)
        pooler_output = self._petrained_language_model(
                input_ids=input_id_,
                attention_mask=attention_mask_,
                token_type_ids=token_type_ids_,
        )[1] 
        
        pooler_output = self.dropout_plm(pooler_output)
        document_embedding = pooler_output.view(num_sentences, max_segments, self.size_petrained) # [total_segments, size_petrained_model] -> [num_sentences, max_segments, size_petrained]
        
        lstm_outputs, _ = self.lstm(document_embedding, self._generate_init_hidden_state(num_sentences))
        lstm_outputs = self.dropout_lstm(lstm_outputs)        

        idxs_sentences = torch.arange(num_sentences)
        idx_last_output = num_segments - 1
        ouput_last_time_step = lstm_outputs[idxs_sentences, idx_last_output]

        logits = self.fc(ouput_last_time_step)

        if "labels" in x:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), x['labels'].view(-1))
            return loss, logits
        return logits
    
    def get_parameters_to_optimize(self, weight_decay=0.01):
        
        params_plmc = get_optimizer_grouped_parameters(
            list(self._petrained_language_model.named_parameters()), 
            weight_decay=weight_decay
        )
        
        params_fc = get_optimizer_grouped_parameters(
            list(self.fc.named_parameters()), 
            weight_decay=weight_decay
        )
        
        return params_plmc + params_fc

    def tokenizer(self):
        return self._tokenizer
    
    def petrained_language_model(self):
        return self._petrained_language_model

def get_optimizer_grouped_parameters(param_optimizer, weight_decay=0.01, no_decay=["bias", "LayerNorm.weight"]):
    if len(no_decay) > 0:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
    else:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            }
        ]
    return optimizer_grouped_parameters

def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)





def train(username,model, train_inputs, validation_inputs, device, path_model, id2label, batch_size=128, weight_decay=0.01, num_train_epochs=10, lr=5e-5, eps=1e-8, 
    num_warmup_steps=0, seed=42, decimals=3, max_grad_norm=1, use_token_type_ids=False, lr_lstm=0.001, epochs_decrease_lr_lstm=3, reduced_factor_lstm=0.95):
    
    train_sampler = RandomSampler(train_inputs)
    train_dataloader = DataLoader(train_inputs, sampler=train_sampler, batch_size=batch_size)
    validation_sampler = SequentialSampler(validation_inputs)
    validation_dataloader = DataLoader(validation_inputs, sampler=validation_sampler, batch_size=batch_size)
    robert = False
    
    model.to(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    

    num_training_batches = len(train_dataloader)
    total_training_steps = num_training_batches * num_train_epochs
    best_val_loss = 100.0
    epoch_wo_improve = 0
    
    if isinstance(model, RoBert):
        optimizer_grouped_parameters = model.get_parameters_to_optimize(weight_decay=weight_decay)
        optimizer_lstm = torch.optim.Adam(model.lstm.parameters(), lr=lr_lstm)
        robert = True
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)
    
    # Step 2: Training loop
    
    training_loss_steps = np.empty((num_train_epochs, num_training_batches))
    training_acc_steps = np.empty_like(training_loss_steps)

    validation_loss_steps = np.empty((num_train_epochs, len(validation_dataloader)))
    validation_acc_steps = np.empty_like(validation_loss_steps)

    best_validation_acc = 0.0
    best_model = None
    num_class_labels = len(id2label)
    labels = [i for i in range(num_class_labels)]
    class_labels = list(id2label.values())

    set_seed(random_seed=seed)
    
    for idx_epoch in range(0, num_train_epochs):
        
        """ Training """
        
        model.train()
        stime_train_epoch = time.time()

        for idx_train_batch, train_batch in enumerate(train_dataloader):
            # 0: input_ids, 1: attention_mask, 2:token_type_ids, 3:num_segments, 4: labels
            batch_train = tuple(data_.to(device) for data_ in train_batch)
            inputs = {
                'input_ids': batch_train[0],
                'attention_mask': batch_train[1],
                'labels': batch_train[-1],
            }
            if use_token_type_ids:
                inputs['token_type_ids'] = batch_train[2]

            optimizer.zero_grad()
            
            if robert:
                inputs['num_segments'] = batch_train[3]
                loss, logits = model(inputs)
            else:
                loss, logits = model(**inputs)
            
            if n_gpu > 1:
                loss = loss.mean()
                
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm) # clipping gradient for avoiding exploding gradients
            optimizer.step()
            scheduler.step()
            if robert:
                optimizer_lstm.step()
            
            logits = logits.detach().cpu().numpy()
            hypothesis = np.argmax(logits, axis=1)
            expected_predictions = inputs['labels'].to('cpu').numpy()

            training_loss_steps[idx_epoch, idx_train_batch] = loss.item()
            training_acc_steps[idx_epoch, idx_train_batch] = accuracy_score(expected_predictions, hypothesis)
        
        ftime_train_epoch = time.time()

        
        """ Validation"""
        
        model.eval()
        current_confussion_matrix = np.zeros((num_class_labels, num_class_labels), dtype=int)
        stime_validation_epoch = time.time()
        
        for idx_validation_batch, validation_batch in enumerate(validation_dataloader):
            batch_validation = tuple(data_.to(device) for data_ in validation_batch)
            inputs = {
                'input_ids': batch_validation[0],
                'attention_mask': batch_validation[1],
                'labels' : batch_validation[-1],
            }
            if use_token_type_ids:
                inputs['token_type_ids'] = batch_validation[2]
            
            with torch.no_grad():
                if robert:
                    inputs['num_segments'] = batch_validation[3]
                    loss, logits = model(inputs)
                else:
                    loss, logits = model(**inputs)
            
            if n_gpu > 1:
                loss = loss.mean()
            
            logits = logits.detach().cpu().numpy()
            hypothesis = np.argmax(logits, axis=1)
            expected_predictions = inputs['labels'].to('cpu').numpy()

            validation_loss_steps[idx_epoch, idx_validation_batch] = loss.item()
            validation_acc_steps[idx_epoch, idx_validation_batch] = accuracy_score(expected_predictions, hypothesis)
        
        current_validation_acc = np.mean(validation_acc_steps[idx_epoch, :])
        current_val_loss = np.mean(validation_loss_steps[idx_epoch, :])
        ftime_validation_epoch = time.time()


    
        if current_validation_acc > best_validation_acc:
            best_validation_acc = current_validation_acc
            best_model = model
        if robert:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
            else:
                epoch_wo_improve += 1
                if epoch_wo_improve > epochs_decrease_lr_lstm:
                    for g in optimizer_lstm.param_groups:
                        g['lr'] = g['lr'] * reduced_factor_lstm
                    epoch_wo_improve = 0
                    print("Updated lr lstm")

    
    if os.path.exists(path_model):
        shutil.rmtree(path_model)  
    os.makedirs(path_model)
    

    torch.save(best_model.state_dict(), os.path.join(path_model, "pytorch_model.bin"))
    metrics_values = np.round(
        np.array([
            np.mean(training_loss_steps, axis=1),
            np.mean(training_acc_steps, axis=1),
            np.mean(validation_loss_steps, axis=1),
            np.mean(validation_acc_steps, axis=1)
        ]), 
        decimals=decimals,
    )
    
    metrics_labels = [
        "training_loss",
        "training_acc",
        "validation_loss",
        "validation_acc"
    ]
    
    df = pd.DataFrame(
        metrics_values.T,
        columns=metrics_labels,
    )

    save_metric_plot(os.path.join(path_model, "accuracy"), df["training_acc"], df["validation_acc"], "Epoch", "Accuracy", loc="lower right")
    save_metric_plot(os.path.join(path_model, "loss"), df["training_loss"], df["validation_loss"], "Epoch", "Loss", loc="upper right")
    path23 = os.path.join("saved_models",username,"data.txt")
    file1 = open(path23,"w")
    tot = [str(df["training_acc"].iloc[-1])+"\n",str(df["validation_acc"].iloc[-1])+"\n",str(df["training_loss"].iloc[-1])+"\n",str(df["validation_loss"].iloc[-1])+"\n"]
    file1.writelines(tot)

    file1.close() 


    id_to_label_str = json.dumps(id2label)
    with open(os.path.join(path_model, "labels.json"), "w") as fjson:
        fjson.write(id_to_label_str)
        
        

def test(model,validation_inputs, device, batch_size=1, use_token_type_ids=False):
    
    validation_sampler = SequentialSampler(validation_inputs)
    validation_dataloader = DataLoader(validation_inputs, sampler=validation_sampler, batch_size=batch_size)
    
    model.to(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    
    model.eval()
    
    for idx_validation_batch, validation_batch in enumerate(validation_dataloader):
        batch_validation = tuple(data_.to(device) for data_ in validation_batch)
        inputs = {
            'input_ids': batch_validation[0],
            'attention_mask': batch_validation[1],
            'labels' : batch_validation[-1],
        }
        if use_token_type_ids:
            inputs['token_type_ids'] = batch_validation[2]
        
        with torch.no_grad():

            inputs['num_segments'] = batch_validation[3]
            loss, logits = model(inputs)

        
        if n_gpu > 1:
            loss = loss.mean()
        
        logits = logits.detach().cpu().numpy()
        print(logits)
        hypothesis = np.argmax(logits, axis=1)
        print(hypothesis)
    return hypothesis
        
        
        
        
