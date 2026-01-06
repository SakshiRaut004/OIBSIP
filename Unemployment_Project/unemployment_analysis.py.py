import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
print(data.head())

print(data.info())
print(data.describe())

data.columns = data.columns.str.strip()
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

print(data.isnull().sum())

plt.figure(figsize=(10,5))
sns.lineplot(
    x="Date",
    y="Estimated Unemployment Rate (%)",
    data=data
)
plt.title("Unemployment Rate During Covid-19")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(
    x="Region",
    y="Estimated Unemployment Rate (%)",
    data=data
)
plt.xticks(rotation=90)
plt.title("Region-wise Unemployment Rate")
plt.show()

sns.boxplot(
    x="Region",
    y="Estimated Unemployment Rate (%)",
    data=data
)
plt.title("Urban vs Rural Unemployment Rate")
plt.show()
