from transformers import TapasTokenizer

from src.utils.constants import model_name
import pandas as pd
import torch

tokenizer = TapasTokenizer.from_pretrained(model_name)

table_csv_path = "individual_csv/"


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # TapasTokenizer expects the table data to be text only
        table = pd.read_csv(table_csv_path + item.table_file).astype(str)
        # print("Answer Coordinates:", item.answer_coordinates)
        # print(table)
        encoding = self.tokenizer(table=table,
                                  queries=item.questions,
                                  answer_coordinates=item.answer_coordinates,
                                  answer_text=item.answer_text,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt"
                                  )
        # remove the batch dimension which the tokenizer adds
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding

    def __len__(self):
        return len(self.df)


