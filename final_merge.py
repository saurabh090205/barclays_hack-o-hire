import pandas as pd

legit = pd.read_csv("emails.csv")
phish_real = pd.read_csv("phishing_real.csv")
phish_ai = pd.read_csv("phishing_ai.csv")

final_df = pd.concat(
    [legit, phish_real, phish_ai],
    ignore_index=True
)

# Shuffle dataset
final_df = final_df.sample(frac=1, random_state=42)

final_df.to_csv("FINAL_BARCLAYS_DATASET.csv", index=False)

print("\n🏆 FINAL DATASET CREATED")
print(final_df['label'].value_counts())
