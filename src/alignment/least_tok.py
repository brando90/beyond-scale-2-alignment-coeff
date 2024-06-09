from transformers import AutoTokenizer
from datasets import load_dataset

hf_token = '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'

# Load Llama3 tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

def load_and_process_datasets(dataset_name: str, config_name: str = None, text_field: str = 'text', split: str = 'train', isCodeDataSet=False, isProofDataSet=False, text_field2: str = None) -> int:
    """Loads a subset of a dataset from Hugging Face and processes it to count tokens."""


    # Load dataset
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    token_count = 0

    for item in dataset:
        if isProofDataSet:
            text = f'informal statement {item[text_field]} formal statement {item[text_field2]}'
        elif isCodeDataSet:
            text = f'Code {item[text_field]} Docstring {item[text_field2]}'
        else:
            text = item[text_field]

        tokens = tokenizer(text)['input_ids']
        token_count += len(tokens)

    return token_count

def count_tokens_in_datasets(datasets_info):
    token_counts = {}
    for info in datasets_info:
        dataset_name = info.get('dataset_name')
        config_name = info.get('config_name')
        text_field = info.get('text_field', 'text')
        isCodeDataSet = info.get('isCodeDataSet', False)
        isProofDataSet = info.get('isProofDataSet', False)
        text_field2 = info.get('text_field2', None)
        split = info.get('split', 'train')

        token_count = load_and_process_datasets(dataset_name, config_name, text_field, split, isCodeDataSet, isProofDataSet, text_field2)
        token_counts[dataset_name] = token_count

        print(f"The dataset {dataset_name} has {token_count} tokens.")

    return token_counts


def count_tokens_in_coq_gym(dataset_name: str, split: str = 'test'):
    """Count the number of tokens in the Coq-Gym dataset."""
    dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True, use_auth_token=hf_token)
    token_count = 0

    print("here")
    for item in dataset:
        # Concatenate relevant text fields
        relevant_lemmas = ' '.join(item['relevant_lemmas']) if 'relevant_lemmas' in item else ''
        prev_tactics = ' '.join(item['prev_tactics']) if 'prev_tactics' in item else ''
        context = ' '.join([f"{key}: {value}" for key, value in item['context'].items()]) if 'context' in item else ''
        tactic = item['tactic'] if 'tactic' in item else ''

        text = f"{relevant_lemmas} {prev_tactics} {context} {tactic}"
        
        # Tokenize the text and count the tokens
        tokens = tokenizer(text)['input_ids']
        token_count += len(tokens)
        print(token_count)
    return token_count

# List of datasets and their configurations
datasets_info = [
    {'dataset_name': 'UDACA/AF', 'config_name': None, 'text_field': 'text'}
    #{'dataset_name': 'UDACA/AF', 'config_name': None, 'text_field': 'text'},
    #{'dataset_name': 'UDACA/AF-split', 'config_name': None, 'text_field': 'Statement:'},
    #{'dataset_name': 'hoskinson-center/minif2f-lean4', 'config_name': None, 'text_field': 'informal_stmt', 'text_field2': 'formal_statement', 'split': 'validation', 'isProofDataSet': True},
    #{'dataset_name': 'hoskinson-center/proofnet', 'split': 'validation', 'config_name': None, 'text_field': 'nl_statement', 'text_field2': 'formal_statement', 'isProofDataSet': True}
]

# Count tokens in each dataset
#token_counts = count_tokens_in_datasets(datasets_info)
# Count tokens in the Coq-Gym dataset
dataset_name = 'brando/Coq-Gym-Data-Set'
split = 'test'  # or 'train' if you want to count tokens in the training split
total_token_count = count_tokens_in_coq_gym(dataset_name, split)
print(f"The total number of tokens in the {dataset_name} ({split} split) is {total_token_count}.")