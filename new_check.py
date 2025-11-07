import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "churn_dataset.csv"
df = pd.read_csv(file_path)

# 🔹 Step 1: Create tenure groups for easier analysis
bins = [0, 12, 24, 48, 72]
labels = ['<1 Year', '1–2 Years', '2–4 Years', '4–6 Years']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

churn_rate_trend = (
    df.groupby(['tenure', 'sim'])['Churn']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .reset_index(name='ChurnRate(%)')
)

plt.figure(figsize=(8,5))
sns.lineplot(data=churn_rate_trend, x='tenure', y='ChurnRate(%)', hue='sim')
plt.title('Churn Rate Trend Over Tenure (Months)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Churn Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
