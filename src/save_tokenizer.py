from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

tokenizer.save_pretrained("models/best_model")

print("Tokenizer saved successfully.")
