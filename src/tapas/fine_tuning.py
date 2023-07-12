from src.queries.queries import q1, q2, q3, q5, q4, q6, q7, ans_1, ans_2, ans_3, ans_4, ans_5, ans_7, ans_6
from transformers import TapasTokenizer
from src.utils.constants import model_name
import pandas as pd

tokenizer = TapasTokenizer.from_pretrained(model_name)

data = {"platforms": ["youtube", "twitch", "amazon", "snapchat", "linkedin"],
        "dates": ["2023-07-07", "2023-01-07", "2023-03-07", "2023-04-07", "2023-05-07"],
        "views": ["809", "1136", "1023", "2452", "1311", ],
        "clicks": ["1574", "1998", "1271", "1746", "1726"],
        "location": ["India", "Austria", "US", 'India', "Austria"],
        "age_group": ["0", "0", "1", "2", "1"]
        }

queries = [q1, q2, q3, q4, q5, q6, q7]

# Passing index of answer present in above data
answer_coordinates = [[(1, 2)],
                      [(0, 0)],
                      [(3, 0)],
                      [(4, 0)],
                      [(1, 0)],
                      [(2, 0)],
                      [(2, 1)]]

answer_text = [[ans_1], [ans_2], [ans_3], [ans_4], [ans_5], [ans_6], [ans_7]]

table = pd.DataFrame.from_dict(data)
inputs = tokenizer(table=table, queries=queries, answer_coordinates=answer_coordinates, answer_text=answer_text,
                   padding='max_length', return_tensors='pt')
