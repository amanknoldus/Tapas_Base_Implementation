import pickle
import pandas as pd
from transformers import TapasTokenizer
from src.utils.constants import saved_model, saved_tokenizer
from transformers import TapasForQuestionAnswering

tapas_model = pickle.load(open(saved_model, 'rb'))
# tokenizer = pickle.load(open(saved_tokenizer, 'rb'))

# model_name = 'google/tapas-base-finetuned-wtq'
# model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
# tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

data = {'platforms': ["youtube", "twitch", "amazon"],
        'dates': ["2023-01-07", "2023-03-07", "2023-04-07"],
        'views': ["1907", "1495", "2537"],
        'clicks': ["1452", "1998", "1200"],
        'location': ["US", "India", "Austria"],
        'age_group': ["0", "1", "2"]
        }

queries = ["What is the highest clicks count?",
           "Which platforms has most views?",
           "What are total number of views?",
           "Can you tell me the Age group which has most clicks?",
           "Please suggest me platform with most clicks and views",
           "Which social site has more engagement of views?",
           "Which campaign has more engagement of clicks?",
           "On what date views were less?"]

table = pd.DataFrame.from_dict(data)

inputs = tokenizer(table=table, queries=queries, padding='max_length', return_tensors="pt")
outputs = model(**inputs)

predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

answers = []

for coordinates in predicted_answer_coordinates:
    if len(coordinates) == 1:
        # only a single cell:
        answers.append(table.iat[coordinates[0]])

    else:
        # multiple cells
        cell_values = []
        for coordinate in coordinates:
            cell_values.append(table.iat[coordinate])
        answers.append(", ".join(cell_values))

print("")
for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
    print(query)
    print("Predicted answer: " + answer)
