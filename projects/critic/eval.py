import sympy
from typing import Union, List, Optional
import re
import json
import os
import argparse

from utils import (load_jsonl, 
                   group_by_prompt, 
                   grouped_prompts_to_dict, 
                   encode_nested_lists, 
                   decode_nested_lists,
                   get_first_key,
                   dump_jsonl,
                   dict_to_jsonl,)

from datasets import Dataset
import numpy as np


verdict_parse_error = 0
final_answer_parse_error = 0

#####Some helper functions#####

def check_equals(n1, n2):
    """Check if two string numbers are equal"""
    try:
        return int(float(sympy.simplify(n1)) == float(sympy.simplify(n2)))
    except:
        return int(n1 == n2)


def get_final_answer(response):
    return response.split("Final Answer: ")[-1].split("#### ")[-1]


def clean_final_answer(final_answer : str):
    global final_answer_parse_error
    final_answer = final_answer.split("=")[-1]
    # Extract number from final answer
    integer_final_answer = re.findall(r"\d[\d,\.]*", final_answer)
    if len(integer_final_answer) > 1:
        print("MULTIPLE INTEGER FINAL ANSWERS FOUND!!!")
        print(final_answer)
        print(integer_final_answer)
        print("----------------")
        final_answer_parse_error += 1
    elif len(integer_final_answer) == 0:
        print("NO INTEGER FINAL ANSWERS FOUND!!!")
        print(final_answer)
        print(integer_final_answer)
        print("----------------")
        final_answer_parse_error += 1
        return ""
    integer_final_answer = integer_final_answer[0]
    integer_final_answer = integer_final_answer.replace("$", "").replace(",", "")
    return integer_final_answer


def split(l, n):
    """Split list l into chunks of size (at most) n"""
    num_chunks = (len(l) + n - 1) // n
    return [l[i*n : (i+1)*n] for i in range(num_chunks)]


#####Eval functions#####
# Evaluate single question single/multiple answer formats

def sparse_heuristic(model_final_answer, gt_final_answer):
    return float(check_equals(model_final_answer, gt_final_answer))

def pass_at_n(model_final_answers, gt_final_answer):
    return max([float(check_equals(mfa, gt_final_answer)) for mfa in model_final_answers])

def majority_vote(model_final_answers, gt_final_answer):
    counts = {}
    maj_ans = model_final_answers[0]
    for mfa in model_final_answers:
        counts[mfa] = 0 if counts.get(mfa) is None else 1 + counts.get(mfa)
        maj_ans = maj_ans if counts[mfa] < counts[maj_ans] else mfa
    return float(check_equals(maj_ans, gt_final_answer))

# NOTE: This could (should) be subsumed by a reranking evaluator
def eval_draft_refinement(gt_fa, draft_fa, refinement_fa, draft_score, refinement_score):
    chosen_fa = draft_fa if draft_score >= refinement_score else refinement_fa
    return sparse_heuristic(chosen_fa, gt_fa)

def eval(model_answers : Union[str, List[str]], tok_outputs, gt_answer : str, model_tok = None, mode : str = "sparse_heuristic", rubric : Optional[str] = None, rm = None, *args, **kwargs) -> int:
    global verdict_parse_error
    gt_final_answer = clean_final_answer(get_final_answer(gt_answer))
    if type(model_answers) is list:
        model_final_answers = [clean_final_answer(get_final_answer(model_answer)) for model_answer in model_answers]
    else:
        model_final_answer = clean_final_answer(get_final_answer(model_answers))
    if mode == "sparse_heuristic":
        score = sparse_heuristic(model_final_answer, gt_final_answer)
    elif mode == "pass_at_n":
        score = pass_at_n(model_final_answers, gt_final_answer)
    elif mode == "majority_vote":
        score = majority_vote(model_final_answers, gt_final_answer)
    elif mode == "eval_gpt4_single_step_eval":
        score = sparse_heuristic(model_final_answer, gt_final_answer)
        eval_score = kwargs.get("verdict")
        eval_score = re.split(r"Final Verdict( A2)?: ", eval_score)[-1].strip()  # NOTE: A2 is the model generated answer
        #eval_score = re.split(r"Final Verdict( A[12])?: ", eval_score)[-1].strip()
        if eval_score.lower() == "yes" or eval_score.lower() == "correct":
            eval_score = 1
        elif eval_score.lower() == "no" or eval_score.lower() == "incorrect":
            eval_score = 0
        else:
            print(f"Eval verdict is unknown: {eval_score}!!!")
            print("---------------------")
            verdict_parse_error += 1
        return int(score == eval_score), score, eval_score
    elif mode == "refinement":
        draft_fa = model_final_answer
        refinement_fa = clean_final_answer(get_final_answer(kwargs.get("refinement")))
        score = eval_draft_refinement(gt_fa=gt_final_answer, 
                                      draft_fa=draft_fa,
                                      refinement_fa=refinement_fa,
                                      draft_score=kwargs.get("draft_score"),
                                      refinement_score=kwargs.get("refinement_score"),)
    elif mode == "model rubric":
        assert rubric is not None
        raise NotImplementedError
    elif mode == "rm":
        assert rm is not None
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported evaluation model {mode}!")
    return score


#####Reward functions for RL#####
# These are essentially batched versions of the evaluation functions

def sparse_reward_fn(outputs, tok_outputs, gt_answers, *args, **kwargs):
    return [eval(output, tok_output, gt_ans, mode="sparse_heuristic", *args, **kwargs) for output, tok_output, gt_ans in zip(outputs, tok_outputs, gt_answers)]


def pass_at_n_reward_fn(outputs, tok_outputs, gt_answers, num_return_sequences, *args, **kwargs):
    # Batch outputs, gt_answers into chunks of size num_return_sequences
    output_chunks = split(outputs, num_return_sequences)
    tok_output_chunks = split(tok_outputs, num_return_sequences)
    gt_answer_chunks = split(gt_answers, num_return_sequences)
    scores = []
    for output_chunk, tok_output_chunk, gt_answer_chunk in zip(output_chunks, tok_output_chunks, gt_answer_chunks):
        score = eval(output_chunk, tok_output_chunk, gt_answer_chunk[0], mode="pass_at_n", *args, **kwargs)
        scores += [score] * len(output_chunk)
    # Return a score for each output, gt_answer pair (note this will be redundant num_return_sequences times)
    return scores

def majority_vote_reward_fn(outputs, tok_outputs, gt_answers, num_return_sequences, *args, **kwargs):
    # Batch outputs, gt_answers into chunks of size num_return_sequences
    output_chunks = split(outputs, num_return_sequences)
    tok_output_chunks = split(tok_outputs, num_return_sequences)
    gt_answer_chunks = split(gt_answers, num_return_sequences)
    scores = []
    for output_chunk, tok_output_chunk, gt_answer_chunk in zip(output_chunks, tok_output_chunks, gt_answer_chunks):
        score = eval(output_chunk, tok_output_chunk, gt_answer_chunk[0], mode="majority_vote", *args, **kwargs)
        scores += [score] * len(output_chunk)
    # Return a score for each output, gt_answer pair (note this will be redundant num_return_sequences times)
    return scores

def eval_draft_refinement_reward_fn(outputs, refinements, gt_answers, draft_scores, refinement_scores, *args, **kwargs):
    return [eval(model_answers=draft,
                 tok_outputs=None,
                 refinement=refinement, 
                 draft_score=draft_score,
                 refinement_score=refinement_score, 
                 gt_answer=gt_answer,
                 mode="refinement",) for draft, refinement, draft_score, refinement_score, gt_answer in zip(outputs, refinements, draft_scores, refinement_scores, gt_answers)]

#####Metric Functions#####

def extract_response(output, i):
    # Get ith repsonse from pipeline output
    try:
        # First split on intermediate critiques, then split on the answer
        output = output.split("<C>")[i]
        if i > 0:
            # If extracting a refinement, take text after answer begins
            output = re.split(r"A:|R:", output)[1]
        return output
    except:
        return "I do not have an answer."


def default_metric_fn(outputs, tok_outputs, gt_answers, *args, **kwargs):
    metrics = {}  
    metrics["maj@1"] = sparse_reward_fn(outputs, tok_outputs, gt_answers, *args, **kwargs)
    metrics["parse_fail_rate"] = final_answer_parse_error / len(outputs)
    #loose_dense_rewards = loose_dense_reward_fn(outputs, tok_outputs, gt_answers, model_tok, require_end=False, *args, **kwargs)
    #metrics["loose_dense_rewards"] = [sum(xs) for xs in loose_dense_rewards]
    #self_consistency = self_consistent_reward_fn(outputs, tok_outputs, gt_answers, model_tok, *args, **kwargs)
    #metrics["self_consistency"] = [abs(sum(xs)) for xs in self_consistency]
    if kwargs.get("num_return_sequences") is not None and kwargs["num_return_sequences"][0] > 1:
        num_return_sequences = kwargs["num_return_sequences"][0]
        metrics["pass@n"] = pass_at_n_reward_fn(outputs, tok_outputs, gt_answers,  num_return_sequences)
        metrics["maj@n"] = majority_vote_reward_fn(outputs, tok_outputs, gt_answers, num_return_sequences)
    if kwargs.get("save_file", None) is not None:
        return metrics, f"default_{kwargs.get('num_return_sequences')[0]}"
    else:
        return metrics


def eval_metric_fn(outputs, tok_outputs, gt_answers, verdicts, *args, **kwargs):
    metrics = {}
    results = [eval(output, tok_output, gt_ans, verdict=verdict, mode="eval_gpt4_single_step_eval", *args, **kwargs) for output, tok_output, gt_ans, verdict in zip(outputs, tok_outputs, gt_answers, verdicts)]
    gts = [res[1] for res in results]
    predictions = [res[2] for res in results]
    tp, fp, tn, fn = 0, 0, 0, 0
    for gt, prediction in zip(gts, predictions):
        if gt == 1 and prediction == 1:
            tp += 1
        elif gt == 1 and prediction == 0:
            fn += 1
        elif gt == 0 and prediction == 1:
            fp += 1
        else:
            tn += 1
    metrics["accuracy"] = [res[0] for res in results]
    metrics["verdict_parse_fail_rate"] = verdict_parse_error / len(outputs)
    metrics["final_answer_parse_fail_rate"] = final_answer_parse_error / len(outputs)
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["% correct verdict"] = (tp + fp) / len(outputs)
    metrics["f1"] = metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    return metrics, ""


def refinement_metric_fn(outputs, tok_outputs, gt_answers, *args, **kwargs):
    metrics = {}  
    metrics["maj@1"] = eval_draft_refinement_reward_fn(outputs=outputs, gt_answers=gt_answers, *args, **kwargs)
    metrics["parse_fail_rate"] = final_answer_parse_error / len(outputs)
    return metrics, ""


def get_metric_fn(metric_name):
    if metric_name == "default":
        return default_metric_fn
    elif metric_name == "refinement":
        return refinement_metric_fn
    elif metric_name == "eval":
        return eval_metric_fn
    else:
        raise ValueError(f"Unknown metric name {metric_name}!!!")

#####Aggregation functions#####

def mean_acc_aggregate_fn(stats):
    """Used to aggregate stats from metric_fn. Just averages everything"""
    var_means = {k: np.mean(v) for k, v in stats.items()}
    return var_means

#####Runner#####

def run_eval(dataset_path, 
             end=None,
             solution_key="model_answer",
             K=1,
             mode="default",
             dump_labeled_dataset=False,
            ):
    print(f"Computing metrics for {dataset_path}...")

    dataset = load_jsonl(dataset_path)
    if end is not None:
        dataset = dataset[:end]
    # Group by prompt so the metric_fn receives the assumed ordering (e.g. for majority vote)
    grouped_dataset = group_by_prompt(dataset)
    # Trim dataset according to K_rerank
    dataset = grouped_prompts_to_dict(grouped_dataset)
    # Need to convert any nested list entries to giant strings to play nice with HF datasets
    for k, v in dataset.items():
        if type(v[0]) is list:
            dataset[k] = [encode_nested_lists(l, depth=0) for l in v]
    # Turn into HF dataset for easy processing
    dataset = Dataset.from_dict(dataset)
    metric_fn = get_metric_fn(mode)

    def make_metric_fields(sample):
        sample["outputs"] = sample[solution_key]
        sample["tok_outputs"] = None
        sample["gt_answers"] = sample["answer"]
        sample["num_return_sequences"] = K
        if mode == "eval":
            if "gpt4_single_step_eval" in sample:
                sample["verdicts"] = sample["gpt4_single_step_eval"]
            elif "gpt4_step_by_step_eval" in sample:
                sample["verdicts"] = sample["gpt4_step_by_step_eval"]
            elif "gpt4_default_eval" in sample:
                sample["verdicts"] = sample["gpt4_default_eval"]
            elif "gpt3.5_step_by_step_eval" in sample:
                sample["verdicts"] = sample["gpt3.5_step_by_step_eval"]
            elif "verdict" in sample:
                sample["verdicts"] = sample.pop("verdict")
            else:
                raise ValueError("No available eval keys!!!")
        return sample

    dataset = dataset.map(make_metric_fields)
    # Convert to dict and decode any encoded nested lists
    dataset = dataset.to_dict()
    for k, v in dataset.items():
        if type(v[0]) is str and "<-1>" == v[0][:4]:
            # NOTE: For now just assuming dtype is always float
            dataset[k] = [decode_nested_lists(l, depth=0, dtype=float) for l in v]

    stats, metric_save_file = metric_fn(**dataset, save_file="")
    results = mean_acc_aggregate_fn(stats)
    print(f"Evaluating {dataset_path}...")
    print(json.dumps(results, indent=2))

    save_file = ".".join(os.path.basename(dataset_path).split(".")[:-1])
    save_file = f"{save_file}_{metric_save_file}"

    with open(f"results/{save_file}.json", "w") as f:
        json.dump(results, f)

    if dump_labeled_dataset:
        for key, stat in stats.items():
            if type(stat) is list and len(stat) == len(dataset[get_first_key(dataset)]):
                dataset[key] = stat
            dump_jsonl(dict_to_jsonl(dataset), f"eval_data/{save_file}.jsonl")
    return results

if __name__ == "__main__":
    #### CLI utilities to evaluate offline datasets ####
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--solution_key", default="model_answer", help="Key in dataset to model answers")
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--K", default=1, type=int, help="Number of candidate answers to rerank")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--dump_labeled_dataset", action="store_true")
    args = parser.parse_args()

    run_eval(**vars(args))


#llama2_7B_step_balanced_positive_params_noprop_positive_final_label_fully_consistent_negative_params_noprop_positive_final_label_fully_consistent_all_step_verified_K_60_sorm