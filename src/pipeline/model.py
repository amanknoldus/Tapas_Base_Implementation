from transformers import TapasTokenizer, TapasConfig
from src.utils.constants import model_name, campaign_query_table, saved_model, saved_tokenizer
import pandas as pd
import torch
from transformers import TapasForQuestionAnswering, AdamW
import pickle
from src.utils.helpers.encoding_data import TableDataset
from src.utils.helpers.group_queries import get_group
from src.utils.helpers.parse_values import parse_answer_text, parse_coordinates

read_file = pd.read_csv(campaign_query_table)
data = pd.DataFrame(read_file)

data['answer_text'] = data['answer_text'].apply(lambda txt: parse_answer_text(txt))
data['answer_coordinates'] = data['answer_coordinates'].apply(lambda coords_str: parse_coordinates(coords_str))

grouped_data = get_group(data)

tokenizer = TapasTokenizer.from_pretrained(model_name)

# Saving tokenizer to pickle file
pickle.dump(tokenizer, open(saved_tokenizer, 'wb'))

train_dataset = TableDataset(df=data, tokenizer=tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2)

config = TapasConfig(
    num_aggregation_labels=4,
    use_answer_as_supervision=True,
    answer_loss_cutoff=0.664694,
    cell_selection_preference=0.207951,
    huber_loss_delta=0.121194,
    init_cell_selection_weights_to_zero=True,
    select_one_column=True,
    allow_empty_column_selection=False,
    temperature=0.0352513,
)

model = TapasForQuestionAnswering.from_pretrained(model_name, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = AdamW(model.parameters(), lr=5e-5)

# loop over the dataset multiple times
for epoch in range(10):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()

# Saving fine tuned model to pickle file
pickle.dump(model, open(saved_model, 'wb'))
