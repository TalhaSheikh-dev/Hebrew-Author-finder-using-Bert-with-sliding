

from dataset import load_20newsgroup_segments, generate_dataset_20newsgroup_segments
from models import RoBert, train
import os
import shutil
import torch
import json

def main(name,directory):
    
    pretrained_model = "avichr/heBERT"
    output_dir ="saved_models"
    batch_size = 2
    max_length = 512
    learning_rate =5e-5
    adam_epsilon =1e-8
    weight_decay =0.01
    num_train_epochs =1
    num_warmup_steps =0
    max_grad_norm =1.0
    seed =42
    lstm_hidden_dim =32
    dense_hidden_dim =16
    num_lstm_layers =1
    num_dense_layers =1
    size_segment =200
    size_shift =50
    lr_lstm =0.001
    epochs_decrease_lr_lstm =3
    reduced_factor_lstm =0.95

    

    
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    use_token_type_ids = True
    
    X_train, y_train, num_segments_train, X_test, y_test, num_segments_test, label2id = load_20newsgroup_segments(
        directory, max_length=max_length, size_segment=size_segment, size_shift=size_shift)
       
    num_classes =len(label2id)

    model_classifier = RoBert(
        num_classes, 
        bert_pretrained=pretrained_model, 
        device=device, 
        lstm_hidden_dim=lstm_hidden_dim, 
        dense_hidden_dim=dense_hidden_dim, 
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers, 
    )

    model_tokenizer = model_classifier.tokenizer()

 
    train_set = generate_dataset_20newsgroup_segments(X_train, y_train, num_segments_train, model_tokenizer, max_length=size_segment)
    validation_set = generate_dataset_20newsgroup_segments(X_test, y_test, num_segments_test, model_tokenizer, max_length=size_segment)
    
    model_path = os.path.join(output_dir, name)
    label_path = os.path.join(output_dir, name,"labels.json")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
            
    os.makedirs(model_path)

    id2label = {value:key for (key, value) in label2id.items()}
    with open(label_path, 'w') as f:
        json.dump(id2label, f)

    try:
        train(name,model_classifier, train_set, validation_set, device, model_path, id2label, batch_size=batch_size, weight_decay=weight_decay, 
            num_train_epochs=num_train_epochs, lr=learning_rate, eps=adam_epsilon, num_warmup_steps=num_warmup_steps, 
            max_grad_norm=max_grad_norm, seed=seed, use_token_type_ids=use_token_type_ids, lr_lstm=lr_lstm, epochs_decrease_lr_lstm=epochs_decrease_lr_lstm, 
            reduced_factor_lstm=reduced_factor_lstm)
        return 1
    except:
        return 0


#if __name__ == "__main__": 
#    main("abc","/home/talha/Documents/BERT-long-sentence/data_our")
