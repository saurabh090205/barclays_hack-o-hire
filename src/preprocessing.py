import pandas as pd
from sklearn.model_selection import train_test_split

# Load final dataset
df = pd.read_csv("data/processed/final_dataset.csv")

# Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

print("\nTrain Label Distribution:")
print(train_df['label'].value_counts())

print("\nTest Label Distribution:")
print(test_df['label'].value_counts())

# Save
train_df.to_csv("data/processed/train_split.csv", index=False)
test_df.to_csv("data/processed/test_split.csv", index=False)

print("\nTrain/Test splits saved.")
