from datasets import load_dataset
from transformers import AutoTokenizer

class TokenCounter:
    def __init__(self, path, name, split, tokenizer_name="mistralai/Mistral-7B-v0.1"):  # Use Mistral tokenizer
        """
        Initialize the TokenCounter with the dataset and tokenizer.

        Parameters:
        - dataset_name: str, the name of the dataset to load.
        - split: str, the dataset split to use (e.g., 'train', 'test').
        - tokenizer_name: str, the name of the tokenizer to use (default: 'mistralai/Mistral-7B-v0.1').
        """
        self.dataset = load_dataset(path, name, split=split, streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.total_tokens = 0
        self.index_stopped = 0

    def count_tokens(self, stop_index=None):
        """
        Count tokens in the dataset.

        Parameters:
        - stop_index: int, optional, the index at which to stop counting. If None, count the whole dataset.
        """
        for i, example in enumerate(self.dataset):
            if stop_index is not None and i >= stop_index:
                break
            inputs = self.tokenizer(example['text'], return_tensors="pt")  # Tokenize
            self.total_tokens += inputs['input_ids'].size(1)  # Count tokens
            self.index_stopped = i

        return self.total_tokens, self.index_stopped

    def find_index_by_token_count(self, target_tokens):
        """
        Find the index where the total token count reaches or exceeds the target.

        Parameters:
        - target_tokens: int, the target number of tokens.

        Returns:
        - int, the index where the target token count is reached or exceeded.
        """
        self.total_tokens = 0
        self.index_stopped = 0

        for i, example in enumerate(self.dataset):
            inputs = self.tokenizer(example['text'], return_tensors="pt")  
            self.total_tokens += inputs['input_ids'].size(1) 
            self.index_stopped = i

            if self.total_tokens >= target_tokens:
                return i

        # If the target is not reached within the dataset 
        return None 

# Example usage:
path = 'UDACA/AF' 
name ='default'
split = 'train' 

# Initialize the TokenCounter
token_counter = TokenCounter(path, name, split)

# # Count tokens, optionally up to a certain index
# total_tokens, index_stopped = token_counter.count_tokens(stop_index=None)

# print(f"Total tokens: {total_tokens}, Index stopped: {index_stopped}")

target_tokens = 492615  # Set your target number of tokens

index = token_counter.find_index_by_token_count(target_tokens)

if index is not None:
    print(f"Index where the target token count is reached: {index}")
else:
    print("Target token count not reached within the dataset.")