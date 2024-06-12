from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("tasksource/leandojo")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to extract and concatenate 'state_before', 'state_after', and 'tactic'
def extract_tactics_data(example):
    if example['traced_tactics']:  # Ensure the list is not empty
        tactics_data = " ".join([f"State Before: {tactic['state_before']} State After: {tactic['state_after']} Tactic: {tactic['tactic']}" for tactic in example['traced_tactics']])
        return tactics_data
    else:
        return ""

# Apply the extraction function to the dataset and filter out empty results
extracted_texts = dataset.map(lambda x: {"text": extract_tactics_data(x)}, batched=False, remove_columns=dataset['train'].column_names)
extracted_texts = extracted_texts.filter(lambda x: x['text'] != "")

# Tokenize the extracted text
tokenized_texts = extracted_texts.map(lambda x: tokenizer(x["text"], return_length=True), batched=True, remove_columns=["text"])

# Sum the lengths of all tokenized examples
total_tokens = sum(tokenized_texts["train"]["length"])

print(f"Total number of tokens in the extracted 'traced_tactics' from the tasksource/leandojo dataset: {total_tokens}")
