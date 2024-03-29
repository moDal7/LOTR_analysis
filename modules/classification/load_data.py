import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import sklearn

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def race2label(race):
    conversion = {
                  "Dwarves": 0,
                  "Elves": 1,
                  "Ents": 2,
                  "Hobbits": 3,
                  "Men": 4,
                  "Maiar": 5,
                  "Uruk-hai/Orcs": 6,
                  "Wraith/Undead": 6
                  }
    return conversion[race]

def label2race(label):
    conversion = {
                  0: "Dwarves",
                  1: "Elves",
                  2: "Ents",
                  3: "Hobbits",
                  4: "Men",
                  5: "Maiar",
                  6: "Uruk-hai/Orcs/Wraith/Undead"
                  }
    return conversion[label]

def load_data(data_path, indexes):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

    data = pd.read_csv(data_path)
    new_df = pd.DataFrame()
    new_df = new_df['text'] = data['Dialog']
    new_df['labels'] = data['Race'].apply(race2label)

    # generate train and test sets based on indexes
    train_df = new_df.iloc[indexes[0]]
    valid_df = new_df.iloc[indexes[1]]

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(valid_df.shape))

    training_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
    validation_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
    
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **train_params)
   
    return training_loader, validation_loader