from datasets import load_dataset
import numpy as np
import gzip
import io
import time

hf_token = ''

def compress(data: str) -> int:
    """Compresses the input data using gzip and returns the size of the compressed data."""
    with io.BytesIO() as byte_stream:
        with gzip.GzipFile(fileobj=byte_stream, mode='wb') as gzip_file:
            gzip_file.write(data.encode('utf-8'))
        compressed_data = byte_stream.getvalue()
    return len(compressed_data)

def normalized_compression_distance(s1: str, s2: str) -> float:
    """Calculates the Normalized Compression Distance between two strings."""
    c1 = compress(s1)
    c2 = compress(s2)
    c12 = compress(s1 + s2)
    return (c12 - min(c1, c2)) / max(c1, c2)

def similarity(s1: str, s2: str) -> float:
    """Calculates similarity based on the normalized compression distance."""
    return 1 - normalized_compression_distance(s1, s2)

def load_and_process_datasets(dataset_name: str, config_name: str = None, subset_size: int = 128, text_field: str = 'text', split: str = 'train', isCodeDataSet = False, isProofDataSet = False, text_field1: str = None, text_field2: str = None, isLean = False, isCoqGym = False) -> list:
    """Loads a subset of a dataset from Hugging Face and processes it for similarity computation."""
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True, use_auth_token=hf_token)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True, use_auth_token=hf_token)
    text_data = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        if isProofDataSet:
            text = f'informal statement {item[text_field1]} formal statement {item[text_field2]}'
            text_data.append(text)
        elif isCodeDataSet:
            text = f'Code {item[text_field1]} Docstring {item[text_field2]}'
            text_data.append(text)  
        elif isLean:
            if item[text_field]:  # Ensure the list is not empty
                # Concatenate the 'state_before', 'state_after', and 'tactic' from traced_tactics
                tactics_data = " ".join([f"State Before: {tactic['state_before']} State After: {tactic['state_after']} Tactic: {tactic['tactic']}" for tactic in item[text_field]])
                text = tactics_data
                text_data.append(text)
            else:
                continue  # Skip if the list is empty
        elif isCoqGym:
            prev_tactics_text = " ".join(item['prev_tactics']) if item['prev_tactics'] else ""
            context_text = " ".join([f"{key}: {value}" for key, value in item['context'].items()])
            text = f"{prev_tactics_text} {context_text}"
            text_data.append(text)
        else:
            text_data.append(item[text_field])
    return text_data

def compute_average_pairwise_similarity(data1: list, data2: list) -> float:
    """Computes the average pairwise similarity between two lists of text data."""
    similarities = []
    for text1 in data1:
        for text2 in data2:
            sim = similarity(text1, text2)
            similarities.append(sim)
    return np.mean(similarities)

# Load datasets
#data_AF = load_and_process_datasets('UDACA/AF', None, 229, text_field='text') 
#data_docstring = load_and_process_datasets('calum/the-stack-smol-python-docstrings', None, 98, text_field1='body', text_field2='docstring', isCodeDataSet = True) 

#data_c4 = load_and_process_datasets('allenai/c4', 'en', 60)
#data_lean = load_and_process_datasets('hoskinson-center/minif2f-lean4', None, 161, text_field1='informal_stmt', text_field2='formal_statement', split = 'validation', isProofDataSet = True) 

#data_mini2f = load_and_process_datasets('AI4M/minif2f_dataset', None, 186, text_field1='informal', text_field2='formal', split = 'test', isProofDataSet = True) 
data_dojo = load_and_process_datasets('tasksource/leandojo', None, 365, split="train", text_field='traced_tactics', isLean=True, isProofDataSet=False)
#data_coqgym = load_and_process_datasets('brando/Coq-Gym-Data-Set', None, 345, split="test", isCoqGym=True, isProofDataSet=False)

data_PROOFNET = load_and_process_datasets('hoskinson-center/proofnet', None, 186, text_field1='nl_statement', text_field2='formal_statement', split = 'test', isProofDataSet = True) 
#data_AF_SPLIT = load_and_process_datasets('UDACA/AF-split', None, 284, text_field='Statement:')
#data_wikitext = load_and_process_datasets('wikitext', 'wikitext-103-v1', 428)

# Compute average pairwise similarity
start_time = time.time()
average_similarity = compute_average_pairwise_similarity(data_PROOFNET, data_dojo)
end_time = time.time()

print(f"Average Pairwise Similarity: {average_similarity:.4f}")
print(f"Time taken to compute average similarity: {end_time - start_time:.2f} seconds")
