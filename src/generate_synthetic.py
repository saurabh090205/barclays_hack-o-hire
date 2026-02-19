import pandas as pd
import random

# Configuration
NUM_SAMPLES = 400

urgency_phrases = [
    "within 24 hours",
    "immediately",
    "to avoid suspension",
    "as soon as possible",
    "before your account is restricted"
]

actions = [
    "verify your account information",
    "confirm your identity",
    "reset your password",
    "update your payment details",
    "validate your recent transaction"
]

fake_domains = [
    "barclays-secure-login.com",
    "barclays-verification.net",
    "barclays-account-alert.com",
    "secure-barclays-authentication.com",
    "barclays-support-team.com"
]

subjects = [
    "Urgent: Unusual Activity Detected on Your Account",
    "Important Security Notification",
    "Action Required: Account Verification Needed",
    "Notice: Temporary Account Restriction",
    "Security Alert from Barclays Online Banking"
]

def generate_email():
    subject = random.choice(subjects)
    urgency = random.choice(urgency_phrases)
    action = random.choice(actions)
    domain = random.choice(fake_domains)

    body = f"""
Dear Customer,

We have detected unusual activity on your Barclays account. 
For security reasons, you must {action} {urgency}.

Failure to comply may result in temporary suspension of your online banking access.

Please click the secure link below to proceed:

https://{domain}/verify

If you believe this message was sent in error, please contact Barclays support immediately.

Kind Regards,
Barclays Security Team
"""

    return subject + "\n\n" + body.strip()

# Generate synthetic phishing emails
synthetic_data = []

for _ in range(NUM_SAMPLES):
    email_text = generate_email()
    synthetic_data.append({"text": email_text, "label": 1})

df_synthetic = pd.DataFrame(synthetic_data)

# Save
df_synthetic.to_csv("data/synthetic/barclays_synthetic_phishing.csv", index=False)

print("Synthetic dataset created.")
print("Shape:", df_synthetic.shape)
