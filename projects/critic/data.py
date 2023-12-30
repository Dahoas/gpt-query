from pathlib import Path
import json
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import sympy
from eval import eval
from utils import group_by_prompt


"""Helper functions"""

def dump_jsonl(dataset, filename):
    with open(filename, "w") as f:
        for sample in dataset:
            json.dump(sample, f)
            f.write("\n")


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


def get_first_key(d):
    return list(d.keys())[0]


def dict_to_jsonl(d):
    if len(d) == 0:
        return []
    else:
        return [{k: d[k][i] for k in d.keys()} for i in range(len(d[get_first_key(d)]))]


"""Tasks"""

def t1():
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


def t2():
    prm_dataset = load_jsonl("benchmarks/math_prm_test_2.jsonl")
    print(len(prm_dataset))
    stats = defaultdict(int)
    for sample in prm_dataset:
        sample["prompt"] = sample["question"]["problem"]
        sample["finish_reason"] = sample["label"]["finish_reason"]
        sample["question"]["ground_truth_solution"] = sample["question"]["ground_truth_solution"] + "Final Answer: " + sample["question"]["ground_truth_answer"]
        sample["pre_generated_steps"] = sample["question"]["pre_generated_steps"]
        sample["pre_generated_solution"] = "\n".join(sample["pre_generated_steps"])
        sample["pre_generated_solution"] = sample["pre_generated_solution"].split("Answer\n\n")[0] + "Final Answer: " + sample["question"]["pre_generated_answer"]
        stats[sample["label"]["finish_reason"]] += 1
        sample["is_correct"] = eval(model_answers=sample["pre_generated_solution"],
                                    gt_answer=sample["question"]["ground_truth_solution"],
                                    benchmark="math",)["maj@1"]#check_equals(sample["question"]["ground_truth_answer"], sample["question"]["pre_generated_answer"])
    grouped_dataset = group_by_prompt(prm_dataset, key="prompt")
    print(len(grouped_dataset))
    print(prm_dataset[0].keys())
    print("Estimated num correct:", np.mean([sample["is_correct"] for sample in prm_dataset]))
    print("Num correct:", np.mean([sample["finish_reason"] == "solution" for sample in prm_dataset]))
    print(stats)
    pass_n = np.mean(["solution" in samples["finish_reason"] for samples in grouped_dataset.values()])
    print("pass@n: ", pass_n)
    # Filter out questions where labeler got stuck
    # Far fewer positive answers so prioritize those
    # Take a positive when I can other wise a negative
    new_dataset = []
    for j, samples in enumerate(grouped_dataset.values()):
        #print("Sample", j)
        samples = dict_to_jsonl(samples)
        s = None
        for sample in samples:
          if "solution" in sample["finish_reason"]:
              s = sample
              break
        if s is None:
            s = samples[0]
        answer = ""
        for i, line in enumerate(s["pre_generated_steps"]):
            answer += f"\nStep {i}:\n{line}"
        step_labels = []
        for i, step in enumerate(s["label"]["steps"]):
            step_label = None
            for completion in step["completions"]:
                if completion["text"] == s["pre_generated_steps"][i] and completion["rating"] is not None:
                    step_label = completion["rating"]
                    break
            #if step_label is None:
            #    print(s["pre_generated_steps"][i])
            #    print(step["completions"])
            step_labels.append(step_label)
        #step_labels = [step["completions"][0]["rating"] for step in s["label"]["steps"]]
        step_labels += (len(s["pre_generated_steps"]) - len(step_labels)) * [-1]
        assert len(step_labels) == len(s["pre_generated_steps"])
        new_sample = dict(
                            question=s["question"]["problem"],
                            answer=s["question"]["ground_truth_solution"],
                            prompt="Q: "+s["question"]["problem"]+"A: ",
                            model_answer=answer,
                            model_steps=s["pre_generated_steps"],
                            step_labels=step_labels,
                            is_correct=s["is_correct"],
                            finish_reason=s["finish_reason"],
                            answer_match=s["is_correct"] == (s["finish_reason"] == "solution"),
                         )
        # Filter out samples with malformed step_labels
        if None not in step_labels:
            new_dataset.append(new_sample)

    print("New dataset len: ", len(new_dataset))
    print("Num correct:", np.mean([sample["finish_reason"] == "solution" for sample in new_dataset]))
    print("Estimated num correct:", np.mean([sample["is_correct"] for sample in new_dataset]))
    print("Grading error: ", 1 - np.mean([sample["answer_match"] for sample in new_dataset]))

    # Filter out samples with evaluation mismatch
    print("Filtering out grading mismatches...")
    new_dataset = [sample for sample in new_dataset if sample["answer_match"]]

    print("New dataset len: ", len(new_dataset))
    print("Num correct:", np.mean([sample["finish_reason"] == "solution" for sample in new_dataset]))
    dump_jsonl(new_dataset, "benchmarks/filtered_math_prm_test_2.jsonl")


def t3():
    prm_dataset = load_jsonl("benchmarks/math_prm_test_2.jsonl")
    print(len(prm_dataset))
    stats = defaultdict(int)
    for sample in prm_dataset:
        sample["prompt"] = sample["question"]["problem"]
        sample["finish_reason"] = sample["label"]["finish_reason"]
        sample["question"]["ground_truth_solution"] = sample["question"]["ground_truth_solution"] + "Final Answer: " + sample["question"]["ground_truth_answer"]
        sample["pre_generated_steps"] = sample["question"]["pre_generated_steps"]
        sample["pre_generated_solution"] = "\n".join(sample["pre_generated_steps"])
        sample["pre_generated_solution"] = sample["pre_generated_solution"].split("Answer\n\n")[0] + "Final Answer: " + sample["question"]["pre_generated_answer"]
        stats[sample["label"]["finish_reason"]] += 1
        sample["is_correct"] = eval(model_answers=sample["pre_generated_solution"],
                                    gt_answer=sample["question"]["ground_truth_solution"],
                                    benchmark="math",)["maj@1"]#check_equals(sample["question"]["ground_truth_answer"], sample["question"]["pre_generated_answer"])
    grouped_dataset = group_by_prompt(prm_dataset, key="prompt")
    print(len(grouped_dataset))
    print(prm_dataset[0].keys())
    print("Estimated num correct:", np.mean([sample["is_correct"] for sample in prm_dataset]))
    print("Num correct:", np.mean([sample["finish_reason"] == "solution" for sample in prm_dataset]))
    print(stats)
    pass_n = np.mean(["solution" in samples["finish_reason"] for samples in grouped_dataset.values()])
    print("pass@n: ", pass_n)
    lens = [samples["K"][0] for samples in grouped_dataset.values()]
    print("Mean len: ", np.mean(lens))
    print("Min len: ", np.min(lens))
    print("Max len: ", np.max(lens))
    print("Len variance: ", np.var(lens))
    # Keep all responses to a given question except those with 
    # 1) malformed labels
    # 2) not graded correctly by our evaluation code
    new_dataset = []
    for j, samples in enumerate(grouped_dataset.values()):
        samples = dict_to_jsonl(samples)
        s1 = samples[0]
        new_sample = dict(
                            question=s1["question"]["problem"],
                            answer=s1["question"]["ground_truth_solution"],
                            prompt="Q: "+s1["question"]["problem"]+"A: ",
                         )
        cnt = 0
        for sample in samples:
            # Construct step_labels list
            step_labels = []
            for i, step in enumerate(sample["label"]["steps"]):
                step_label = None
                for completion in step["completions"]:
                    if completion["text"] == sample["pre_generated_steps"][i] and completion["rating"] is not None:
                        step_label = completion["rating"]
                        break
                #if step_label is None:
                #    print(s["pre_generated_steps"][i])
                #    print(step["completions"])
                step_labels.append(step_label)
            step_labels += (len(sample["pre_generated_steps"]) - len(step_labels)) * [-1]
            assert len(step_labels) == len(sample["pre_generated_steps"])
            # Filter out samples which have malformed step labels or we cannot evaluate correctly
            if None not in step_labels \
            and sample["is_correct"] == (sample["finish_reason"] == "solution"):
                new_sample[f"model_steps_{cnt+1}"] = sample["pre_generated_steps"]
                new_sample[f"step_labels_{cnt+1}"] = step_labels
                model_answer = ""
                for i, line in enumerate(sample["pre_generated_steps"]):
                    model_answer += f"\nStep {i}:\n{line}"
                new_sample[f"model_answer_{cnt+1}"] = model_answer
                cnt += 1
        if cnt > 0:
            new_sample["K"] = cnt
            new_dataset.append(new_sample)

    print("New dataset len: ", len(new_dataset))
    lens = [sample["K"] for sample in new_dataset]
    print("Total samples: ", sum(lens))
    print("Mean len: ", np.mean(lens))
    print("Min len: ", np.min(lens))
    print("Max len: ", np.max(lens))
    print("Len variance: ", np.var(lens))

    # Filter out samples with less than three responses
    new_dataset = list(filter(lambda s: s["K"] >= 3, new_dataset))
    print("Filtering out samples with less than three")
    print("New dataset len: ", len(new_dataset))
    lens = [sample["K"] for sample in new_dataset]
    print("Total samples: ", sum(lens))
    print("Mean len: ", np.mean(lens))
    print("Min len: ", np.min(lens))
    print("Max len: ", np.max(lens))
    print("Len variance: ", np.var(lens))

    dump_jsonl(new_dataset, "benchmarks/filtered_math_prm_test_2_multi_sample.jsonl")


if __name__ == "__main__":
    t3()