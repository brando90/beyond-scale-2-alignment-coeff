import os
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset
from itertools import chain
from accelerate import Accelerator

accelerator = Accelerator()
hf_token = ''
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
block_size = 1024  # Reduce sequence length to fit in memory

def load_and_process_datasets(dataset_name: str, config_name: str = None, split: str = 'validation', text_field: str = 'nl_statement', text_field2: str = 'formal_statement', isProofDataSet=False, max_tokens=4096, isLean=False) -> list:
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True, token=hf_token)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True, token=hf_token)
    text_data = []

    total_tokens = 0
    for item in dataset:
        if isProofDataSet:
            text = f'informal statement {item[text_field]} {tokenizer.eos_token} formal statement {item[text_field2]}'
        elif isLean:
            if item[text_field]:
                tactics_data = " ".join([f"State Before: {tactic['state_before']} State After: {tactic['state_after']} Tactic: {tactic['tactic']}" for tactic in item[text_field]])
                text = tactics_data
            else:
                continue
        else:
            text = item[text_field]
        
        tokens = tokenizer(text)['input_ids']
        token_count = len(tokens)
        
        if total_tokens + token_count > max_tokens:
            break
        text_data.append(text)
        total_tokens += token_count

    print("Number of sequences: ", len(text_data))
    print("Number of tokens: ", total_tokens)
    print(text_data[0])
    return text_data

def create_blocks(text_data, tokenizer, block_size):
    concatenated_tokens = list(chain(*[tokenizer(text)['input_ids'] for text in text_data]))
    total_length = len(concatenated_tokens)
    total_length = (total_length // block_size) * block_size
    all_tokens = [concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)]
    return all_tokens

class ProofNetDataset(TorchDataset):
    def __init__(self, tokenized_blocks):
        self.input_ids = tokenized_blocks
        self.labels = tokenized_blocks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def fine_tune_on_dataset(dataset_name, config_name: str = None, split='validation', text_field='informal', text_field2='formal', isProofDataSet=True, isLean=False, tcount=22000):
    train_text_data = load_and_process_datasets(dataset_name, config_name=config_name, split=split, text_field=text_field, text_field2=text_field2, isProofDataSet=isProofDataSet, max_tokens=tcount, isLean=isLean)
    train_blocks = create_blocks(train_text_data, tokenizer, block_size)
    train_dataset = ProofNetDataset(train_blocks)

    test_text_data = load_and_process_datasets('hoskinson-center/proofnet', split='test', isProofDataSet=True, max_tokens=22000)
    test_blocks = create_blocks(test_text_data, tokenizer, block_size)
    test_dataset = ProofNetDataset(test_blocks)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hf_token,
        #device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=5e-7,
        per_device_train_batch_size=2,  # Reduce batch size
        gradient_accumulation_steps=4,  # Simulate larger batch size
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    ))
    
    trainer.train()
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    print(f"Test Cross-Entropy Loss: {eval_loss}")



print("proofnet validation: ")
fine_tune_on_dataset('hoskinson-center/proofnet', split="validation", text_field="nl_statement", text_field2="formal_statement", isProofDataSet=True, tcount=22000)

#print("LeanDojo: ")
#fine_tune_on_dataset('tasksource/leandojo', split="train", text_field='traced_tactics', isLean=True, isProofDataSet=False, tcount=22000)

print("DocString: ")
fine_tune_on_dataset('calum/the-stack-smol-python-docstrings', split="train", text_field='body', text_field2="docstring",  isProofDataSet=True, tcount=22000)

print("AF: ")
fine_tune_on_dataset('UDACA/AF', split="train", text_field="text", isProofDataSet=False, tcount=22000)

print("AF split: ")
fine_tune_on_dataset('UDACA/AF-split', split="train", text_field="Statement:", isProofDataSet=False, tcount=22000)

print("C4: ")
fine_tune_on_dataset('allenai/c4', config_name="en", split="train", text_field="text", isProofDataSet=False, tcount=22000)

print("Wikitext: ")
fine_tune_on_dataset('wikitext', config_name="wikitext-103-v1", split="train", text_field="text", isProofDataSet=False, tcount=22000)
