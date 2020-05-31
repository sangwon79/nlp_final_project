import numpy as np
import pandas as pd
import os
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
from model import TweetModel
from trainer import loss_fn, train_model
from evaluation import get_selected_text, jaccard, compute_jaccard_score
from dataloader import get_train_val_loaders, get_test_loader
warnings.filterwarnings('ignore')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
def main():
    seed = 42
    seed_everything(seed)
    
    num_epochs = 3
    batch_size = 32
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    

    train_df = pd.read_csv('data/train.csv')
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1): 
        print(f'Fold: {fold}')

        model = TweetModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
        criterion = loss_fn    
        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

        train_model(
            model, 
            dataloaders_dict,
            criterion, 
            optimizer, 
            num_epochs,
            f'roberta_fold{fold}.pth')
        
    # inference

    test_df = pd.read_csv('data/test.csv')
    test_df['text'] = test_df['text'].astype(str)
    test_loader = get_test_loader(test_df)
    predictions = []
    models = []
    
    for fold in range(skf.n_splits):
        model = TweetModel()
        model.cuda()
        model.load_state_dict(torch.load(f'roberta_fold{fold+1}.pth'))
        model.eval()
        models.append(model)

    for data in test_loader:
        ids = data['ids'].cuda()
        masks = data['masks'].cuda()
        tweet = data['tweet']
        offsets = data['offsets'].numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)
        for i in range(len(ids)):    
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)
            
    #submission
    
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df['selected_text'] = predictions
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()

if __name__ == '__main__':
    main()