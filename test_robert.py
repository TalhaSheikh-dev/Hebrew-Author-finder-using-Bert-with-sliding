
from dataset import load_20newsgroup_segments1, generate_dataset_20newsgroup_segments
from models import RoBert, test
import os
import torch
import json



def main(text,model_name):
    
    dataset = '20_simplified'
    pretrained_model = "avichr/heBERT"
    batch_size = 2
    max_length = 512
    lstm_hidden_dim =32
    dense_hidden_dim =16
    num_lstm_layers =1
    num_dense_layers =1
    size_segment =200
    size_shift =50



    
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    use_token_type_ids = True
    model_load_path = os.path.join(os.getcwd(),"saved_models",model_name,"pytorch_model.bin")
    label_path = os.path.join("saved_models",model_name,"labels.json")

    with open(label_path, 'r') as f:
        id2label = json.load(f)   
        
    num_classes = len(id2label)

    model_classifier = RoBert(
        num_classes, 
        bert_pretrained=pretrained_model, 
        device=device, 
        lstm_hidden_dim=lstm_hidden_dim, 
        dense_hidden_dim=dense_hidden_dim, 
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers, 
    )


    model_classifier.load_state_dict(torch.load(model_load_path))
    model_tokenizer = model_classifier.tokenizer()
    X_test, y_test, num_segments_test, label2id = load_20newsgroup_segments1(
        text, max_length=max_length, size_segment=size_segment, size_shift=size_shift)
        
    validation_set = generate_dataset_20newsgroup_segments(X_test, y_test, num_segments_test, model_tokenizer, max_length=size_segment)
    

    
    output = test(model_classifier,validation_set, device, batch_size=batch_size,use_token_type_ids=use_token_type_ids)[0]
    
    author_name = id2label[str(output)]
    return author_name
    
if __name__ == "__main__": 
    main("asdjsadjsakd","Default")  