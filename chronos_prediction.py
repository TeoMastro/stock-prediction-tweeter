import os
import pandas as pd
import torch
from chronos import ChronosPipeline
import numpy as np
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np
from sklearn.model_selection import train_test_split




pipeline_tiny = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_mini = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_small = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

data = pd.read_csv('data/AAPL.csv')
filtered_data = data[['Date', 'Adj Close', 'ts_polarity', 'twitter_volume']]

split_index = int(len(filtered_data) * 0.97)
# Split the DataFrame
historical_data = filtered_data.iloc[:split_index]
testing_data = filtered_data.iloc[split_index:]



list_context = [torch.tensor(filtered_data['Adj Close']), torch.tensor(filtered_data['ts_polarity']), torch.tensor(filtered_data['twitter_volume'])]


forecast = pipeline_small.predict(
    context=list_context,
    prediction_length=30,
    num_samples=20,
)




forecast_index = range(len(filtered_data), len(filtered_data) + 30)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(filtered_data['Adj Close'], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()