import json
from tqdm import tqdm
from typing import List
from collections import defaultdict


def get_first_key(d):
    return list(d.keys())[0]


def dict_to_jsonl(d):
    if len(d) == 0:
        return []
    else:
        return [{k: d[k][i] for k in d.keys()} for i in range(len(d[get_first_key(d)]))]


def jsonl_to_dict(l):
    if len(l) == 0:
        return {}
    else:
        return {k: [s[k] for s in l] for k in l[0].keys()}


def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                response = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print("Line num: ", i)
                raise(e)
            data.append(response)
    return data


def dump_jsonl(dataset, filename):
    with open(filename, "w") as f:
        for sample in dataset:
            json.dump(sample, f)
            f.write("\n")


def group_by_prompt(dataset, key="prompt", K=None):
    grouped_samples = defaultdict(lambda: defaultdict(list))
    for sample in tqdm(dataset):
        val_key = sample[key]
        for k, v in sample.items():
            grouped_samples[val_key][k].append(v)

    # Trim extra samples if K is not None
    trimmed_groups = dict()
    for k, samples in grouped_samples.items():
        K = len(samples[key]) if K is None else K
        trimmed_samples = {ki: vi[:K] for ki, vi in samples.items()}
        trimmed_samples["K"] = K * [K]
        trimmed_groups[k] = trimmed_samples

    return trimmed_groups



def grouped_prompts_to_ordered_dict(grouped_dataset):
    """
    Converts a dataset grouped by prompt into a single depth dict
    + grouped_dataset: Of the form grouped_dataset[prompt] = {"prompt": [...], "response": [...], "K": int}
    Note K is not a list but an int
    """
    # Initialize fields of un-grouped dataset
    dataset = {k: [] for k in list(grouped_dataset.values())[0]}
    for sample in grouped_dataset.values():
        for k in sample:
            dataset[k] += sample[k]
    return dataset


def get_base_dtype(l):
    """
    Returns the base dtype of l
    + l: variable to find base dtype of
    """
    allowed_types = ["list", "str", "int", "float", "bool"]
    if type(l).__name__ not in allowed_types:
        raise ValueError(f"Disallowed type for encoding: {type(l).__name__}!!!\n\
                           Value: {l}\n\
                           Only {allowed_types} allowed.")
    if type(l) is list and len(l) > 0:
        return get_base_dtype(l[0])
    else:
        return type(l)


def encode_nested_lists(l, depth):
    allowed_types = ["list", "str", "int", "float", "bool"]
    if type(l).__name__ not in allowed_types:
        raise ValueError(f"Disallowed type for encoding: {type(l).__name__}!!!\n\
                           Value: {l}\n\
                           Only {allowed_types} allowed.")
    
    # If l is not a list, encode directly as str
    if type(l) is not list:
        enc_l = [str(l)]
    # Otherwise first encode any nested lists as str at lower depth
    else:
        enc_l = [encode_nested_lists(sub_l, depth+1) for sub_l in l]
    # Then encode at cur depth
    encoding = f"<{depth}>".join(enc_l)
    # Finally, mark top level encoding as encoded + save dtype
    if depth == 0:
        # Get base dtype of l
        dtype = get_base_dtype(l)
        if dtype.__name__ not in allowed_types:
            raise ValueError(f"Disallowed type for encoding: {type(l).__name__}!!!\
                           Only {allowed_types} allowed.")
        encoding = f"<-1><{dtype.__name__}>" + encoding
    return encoding


def decode_nested_lists(l: str, depth: int, dtype: type = str):
    # Check to make sure l is an encoded nested list
    if depth == 0:
        assert "<-1>" == l[:4]
        l = l[4:]
        # Recover base dtype
        l = l.split(">")
        dtype = eval(l[0][1:])
        l = ">".join(l[1:])
    l = l.split(f"<{depth}>")
    # If there are no delimiters then we can directly convert
    if len(l) == 1:
        return dtype(l[0])
    # Otherwise simply must decode recursively in each component
    return [decode_nested_lists(sub_l, depth+1, dtype) for sub_l in l]


def split(l, n):
    """Split list l into chunks of size (at most) n"""
    num_chunks = (len(l) + n - 1) // n
    return [l[i*n : (i+1)*n] for i in range(num_chunks)]


def classify_error_type(model_label, gt_label):
    if gt_label != 0 and gt_label != 1:
        raise ValueError(f"gt_label is wrong: {gt_label}, {type(gt_label)}. Must be -1 or 1!!!")
    # Classify malformed model_labels as either false positives or false negatives
    if model_label != 0 and model_label != 1:
        if gt_label:
            return "fn"
        else:
            return "fp"
    # Otherwise handle labels as usual
    if gt_label and model_label:
        return "tp"
    elif gt_label and not model_label:
        return "fn"
    elif not gt_label and model_label:
        return "fp"
    else:
        return "tn"
    

def fill_missing_values(jsonl: List[dict], value=None, infer_dtype=False, debug=False):
    """
    Fills in missing keys with None value
    + jsonl: jsonl to update
    + value: value used to update jsonl
    + infer_dtype: If true infers dtype of field and uses default type value
    """
    # First take union of all keys over all samples in jsonl
    key_set = dict()
    for s in jsonl:
        for k in s.keys():
            key_set[k] = type(s[k])() if infer_dtype else value
    print("Keys found: ", key_set) if debug else None
    # Then set s[k] for all s in jsonl
    for s in jsonl:
        for k in key_set:
            s[k] = s.get(k, key_set[k])

    return jsonl