from pathlib import Path
import json
import os


def dump_jsonl(dataset, filename):
    with open(filename, "w") as f:
        for sample in dataset:
            json.dump(sample, f)
            f.write("\n")


dataset = []
folder = Path(".")
paths = folder.glob("Math/test/*/*")
for i, path in enumerate(paths):
    sample = json.load(open(path, "r"))
    sample["id"] = i
    sample["type"] = os.path.basename(os.path.dirname(str(path)))
    sample["question"] = sample.pop("problem")
    sample["prompt"] = "Q: " + sample["question"] + "A: "
    sample["answer"] = sample.pop("solution")
    dataset.append(sample)
dump_jsonl(dataset, "cot_math.jsonl")
import random
random.shuffle(dataset)
dump_jsonl(dataset[:1000], "cot_math_small.jsonl")