import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


def get_dict(le):
    classes = le.classes_
    final_dict = {}
    for abc,ra in zip(classes,range(len(classes))):
        final_dict[abc] = ra
    return final_dict
        
def _load_20newsgroup(directory, max_length=512):
    text = []
    tags = []
    dir = directory
    for a in os.listdir(dir):
      new = os.path.join(dir,a)
      
      for x in os.listdir(new):
        if x.endswith(".txt"):
          read_dir = os.path.join(new,x)
          b = open(read_dir,"r",encoding="utf8", errors='ignore')
          text_file = b.read()
          text_file = text_file[:1]
          text.append(text_file)
          tags.append(a)
    dict = {'Text':text, 'number':tags}
    data = pd.DataFrame(dict) 
  
    data["number"] = le.fit_transform(data["number"])
    label2id = get_dict(le)
    data_train,data_test = train_test_split(data,test_size=0.7)
    X_train = data_train["Text"].to_list()
    y_train = data_train["number"].to_list()

    X_test = data_test["Text"].to_list()
    y_test = data_test["number"].to_list()
    return X_train, y_train, X_test, y_test, label2id

def load_20newsgroup_segments1(text, max_length=512, size_segment=200, size_shift=50):
    X_test_full = [text]
    y_test = [0]
    label2id = {"aa":0}
    
    def get_segments(sentence):
        list_segments = []
        lenght_ = size_segment - size_shift
        tokens = sentence.split()
        num_tokens = len(tokens)
        num_segments = math.ceil(len(tokens) / lenght_)
        if num_tokens > lenght_:
            for i in range(0, num_tokens, lenght_):
                j = min(i+size_segment, num_tokens)
                list_segments.append(" ".join(tokens[i:j]))
        else:
                list_segments.append(sentence)    
        return list_segments, len(list_segments)
    def get_segments_from_section(sentences):
        list_segments = []
        list_num_segments = []
        for sentence in sentences:
            ls, ns = get_segments(sentence)
            list_segments += ls
            list_num_segments.append(ns)
        return list_segments, list_num_segments
    
    X_test, num_segments_test = get_segments_from_section(X_test_full)
    
    return X_test, y_test, num_segments_test, label2id

def load_20newsgroup_segments(directory, max_length=512, size_segment=200, size_shift=50):
    X_train_full, y_train, X_test_full, y_test, label2id = _load_20newsgroup(directory, max_length=max_length)
    def get_segments(sentence):
        list_segments = []
        lenght_ = size_segment - size_shift
        tokens = sentence.split()
        num_tokens = len(tokens)
        num_segments = math.ceil(len(tokens) / lenght_)
        if num_tokens > lenght_:
            for i in range(0, num_tokens, lenght_):
                j = min(i+size_segment, num_tokens)
                list_segments.append(" ".join(tokens[i:j]))
        else:
                list_segments.append(sentence)    
        return list_segments, len(list_segments)
    def get_segments_from_section(sentences):
        list_segments = []
        list_num_segments = []
        for sentence in sentences:
            ls, ns = get_segments(sentence)
            list_segments += ls
            list_num_segments.append(ns)
        return list_segments, list_num_segments
    X_train, num_segments_train = get_segments_from_section(X_train_full)
    X_test, num_segments_test = get_segments_from_section(X_test_full)
    
    return X_train, y_train, num_segments_train, X_test, y_test, num_segments_test, label2id

def generate_dataset_20newsgroup_segments(X, Y, num_segments, tokenizer, pad_to_max_length=True, add_special_tokens=True, max_length=200, return_attention_mask=True, 
    return_tensors='pt'):
    tokens = tokenizer.batch_encode_plus(
        X, 
        pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        return_attention_mask=return_attention_mask, # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
        return_tensors="pt",
    )
    num_sentences = len(Y)
    max_segments = max(num_segments)
    input_ids = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["input_ids"].dtype)
    attention_mask = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["attention_mask"].dtype)
    token_type_ids = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["token_type_ids"].dtype)
    # pad_token = 0
    pos_segment = 0
    for idx_segment, n_segments in enumerate(num_segments):
        for n in range(n_segments):
            input_ids[idx_segment, n] = tokens["input_ids"][pos_segment]
            attention_mask[idx_segment, n] = tokens["attention_mask"][pos_segment]
            token_type_ids[idx_segment, n] = tokens["token_type_ids"][pos_segment]
            pos_segment += 1 
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, torch.tensor(num_segments), torch.tensor(Y))
    return dataset


