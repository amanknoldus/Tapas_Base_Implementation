from pathlib import Path

path = Path(__file__).resolve().parent.parent
llm_model_dir = path / "llm_model"

model_name = "google/tapas-base"

data_dir = path / "dataset"
dataset = data_dir / "campaign_data.csv"
