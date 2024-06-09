import os
from transformers import AutoTokenizer, LlamaForCausalLM, GPT2LMHeadModel
from datasets import load_dataset
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
hf_token = ''

# Define the paths to the saved model
#model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "gpt2"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#model = LlamaForCausalLM.from_pretrained(
#    model_name,
#    device_map="auto",  # Automatically distributes across available devices
#    torch_dtype=torch.float16,  # Use mixed precision if supported
#)
model = GPT2LMHeadModel.from_pretrained(
    model_name, 
    token=hf_token,
    device_map="auto"  # Automatically distribute model across available devices
)



# Define the input format
def load_and_process_dataset(dataset_name: str = 'hoskinson-center/proofnet', split: str = 'test', text_field: str = 'nl_statement', text_field2: str = 'formal_statement') -> list:
    dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True, token=hf_token)
    text_data = []
    for item in dataset:
        text = f'Informal statement {item[text_field]} Formal statement {item[text_field2]}'
        text_data.append(text)
    return text_data

# Prepare the inputs
inputs = load_and_process_dataset()

# Tokenize the inputs
tokenized_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Move inputs to GPU #5
input_ids = tokenized_inputs['input_ids'].to(device)
attention_mask = tokenized_inputs['attention_mask'].to(device)

# Generate outputs and calculate cross-entropy loss
model.eval()
total_loss = 0
num_batches = 0
with torch.no_grad():
    outputs = []
    batch_size = 4  # Adjust batch size as needed
    for i in tqdm(range(0, len(input_ids), batch_size)):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        
        # Get the model outputs
        output = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
        loss = output.loss
        total_loss += loss.item()
        num_batches += 1

        # Generate predictions
        generated_tokens = model.generate(batch_input_ids, attention_mask=batch_attention_mask, max_length=512)
        batch_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs.extend(batch_outputs)

# Print or save the results
for input_text, output_text in zip(inputs, outputs):
    print(f"Input: {input_text}\nOutput: {output_text}\n")

# Calculate average cross-entropy loss
avg_loss = total_loss / num_batches
print(f"Average Cross-Entropy Loss: {avg_loss}")

# Save the results to a file if needed
with open('evaluation_results.txt', 'w') as f:
    for input_text, output_text in zip(inputs, outputs):
        f.write(f"Input: {input_text}\nOutput: {output_text}\n")
