import pandas as pd

from src.utils.constants import dataset

read_file = pd.read_csv(dataset)
df = pd.DataFrame(read_file)

platforms = df.platforms.values
dates = df.dates.values
views = df.views.values
clicks = df.clicks.values
location = df.location.values
age_group = df.age_group.values
