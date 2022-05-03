import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_json('result.json',lines=True)
print(df)