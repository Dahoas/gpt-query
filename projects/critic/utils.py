import json
from tqdm import tqdm


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


def group_by_prompt(dataset, key="prompt"):
    prompt_scores = {}
    for sample in tqdm(dataset):
        sample_prompt = sample[key]
        pre_score = prompt_scores.get(sample_prompt)
        if prompt_scores.get(sample_prompt):
            for k, v in sample.items():
                prompt_scores[sample_prompt][k].append(v)
        else:
            prompt_scores[sample_prompt] = {k: [v] for k, v in sample.items()}

    for sample in prompt_scores.values():
        K = len(sample["prompt"])
        sample["K"] = K * [K]
    return prompt_scores



def grouped_prompts_to_dict(grouped_dataset):
    """Converts a dataset grouped by prompt into a single depth dict
    + grouped_dataset: Of the form grouped_dataset[prompt] = {"prompt": [...], "response": [...], "K": int}
    Note K is not a list but an int
    """
    # Initialize fields of un-grouped dataset
    dataset = {k: [] for k in list(grouped_dataset.values())[0]}
    for sample in grouped_dataset.values():
        for k in sample:
            dataset[k] += sample[k]
    return dataset


def encode_nested_lists(l, depth):
    # If l is not a list, encode directly as str
    if type(l) is not list:
        return str(l)
    # First encode any nested lists as str at lower depth
    l = [encode_nested_lists(sub_l, depth+1) for sub_l in l]
    # Then encode at cur depth
    encoding = f"<{depth}>".join(l)
    # Finally, mark top level encoding as encoded
    if depth == 0:
        encoding = "<-1>" + encoding
    return encoding


def decode_nested_lists(l: str, depth: int, dtype: type):
    # Check to make sure l is an encoded nested list
    if depth == 0:
        assert "<-1>" == l[:4]
        l = l[4:]
    l = l.split(f"<{depth}>")
    # If there are no delimiters then we can directly convert
    if len(l) == 1:
        return dtype(l[0])
    # Otherwise simply must decode recursively in each component
    return [decode_nested_lists(sub_l, depth+1, dtype) for sub_l in l] 