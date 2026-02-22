import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("global_ads_performance_dataset.csv")

# Stratified split based on platform
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["platform"]
)

# Save to CSV
train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

print("Train and test sets saved successfully!")