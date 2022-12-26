import pandas as pd

# load database
df = pd.read_csv('database/ame2020.csv').drop('Unnamed: 0', axis=1)
