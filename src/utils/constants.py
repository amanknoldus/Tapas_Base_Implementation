from pathlib import Path

path = Path(__file__).resolve().parent.parent
llm_model_dir = path / "llm_model"

model_name = "google/tapas-base"

data_dir = path / "dataset"
campaign_dataset = data_dir / "campaign_data.csv"
campaign_query_table = data_dir / "campaign_query_table.csv"

tuned_model_dir = path / "tuned_model"
saved_model = tuned_model_dir / "tapas_base_tuned_model.pkl"
saved_tokenizer = tuned_model_dir / "tokenizer.pkl"

# google/tapas-base-finetuned-wtq


