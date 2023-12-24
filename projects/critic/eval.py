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
                   dict_to_jsonl,
                   jsonl_to_dict,)

from eval_gsm8k import get_gsm8k_final_answer, clean_gsm8k_final_answer
from eval_math import get_math_final_answer, clean_math_final_answer

from datasets import Dataset
import numpy as np


#####Some helper functions#####

def check_equals(n1, n2, eps=1e-3):
    """Check if two string numbers are equal"""
    try:
        return int(np.abs(float(sympy.simplify(n1)) - float(sympy.simplify(n2))) < eps)
    except:
        return int(n1 == n2)


def split(l, n):
    """Split list l into chunks of size (at most) n"""
    num_chunks = (len(l) + n - 1) // n
    return [l[i*n : (i+1)*n] for i in range(num_chunks)]


def get_clean_final_answer(answer, benchmark):
    results = {}
    if benchmark.lower() == "gsm8k":
        final_answer = get_gsm8k_final_answer(answer)
        results["final_answer"], results["final_answer_parse_error"] = clean_gsm8k_final_answer(final_answer)
    elif benchmark.lower() == "math":
        final_answer = get_math_final_answer(answer)
        results["final_answer"], results["final_answer_parse_error"] = clean_math_final_answer(final_answer)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}!!!")
    return results


#####Eval functions#####
# Evaluate single question single/multiple answer formats

def maj_1(model_final_answer, gt_final_answer):
    return float(check_equals(model_final_answer, gt_final_answer))

def pass_n(model_final_answers, gt_final_answer):
    return max([float(check_equals(mfa, gt_final_answer)) for mfa in model_final_answers])

def maj_n(model_final_answers, gt_final_answer, return_chosen=False):
    counts = {}
    # TODO(alex): fix error where 70 and 70.0 likely get counted as two separate answers
    maj_ans = model_final_answers[0]
    for mfa in model_final_answers:
        counts[mfa] = 0 if counts.get(mfa) is None else 1 + counts.get(mfa)
        maj_ans = maj_ans if counts[mfa] < counts[maj_ans] else mfa
    score = float(check_equals(maj_ans, gt_final_answer))
    if return_chosen:
        return score, maj_ans
    else:
        return score

# NOTE: This could (should) be subsumed by a reranking evaluator 
def eval_draft_refinement(gt_fa, draft_fa, refinement_fa, draft_score, refinement_score):
    chosen_fa = draft_fa if draft_score >= refinement_score else refinement_fa
    return maj_1(chosen_fa, gt_fa)

def eval(model_answers : Union[str, List[str]], gt_answer : str, mode : str = "maj@1", benchmark="gsm8k", *args, **kwargs) -> int:
    results = {}
    gt_final_answer, results["gt_parse_error"] = get_clean_final_answer(answer=gt_answer, benchmark=benchmark).values()
    if type(model_answers) is list:
        model_final_answers, final_answer_parse_error = jsonl_to_dict([get_clean_final_answer(model_answer, benchmark=benchmark) for model_answer in model_answers]).values()
    else:
        model_final_answer, final_answer_parse_error = get_clean_final_answer(model_answers, benchmark=benchmark).values()
    results["final_answer_parse_error"] = np.sum(final_answer_parse_error)

    if mode == "maj@1":
        results["maj@1"] = maj_1(model_final_answer, gt_final_answer)
    elif mode == "pass@n":
        results["pass@n"] = pass_n(model_final_answers, gt_final_answer)
    elif mode == "maj@n":
        if kwargs.get("return_chosen"):
            results["maj@n"], results["chosen_maj"] = maj_n(model_final_answers, gt_final_answer, return_chosen=kwargs.get("return_chosen"))
        else:
            results["maj@n"] = maj_n(model_final_answers, gt_final_answer, return_chosen=kwargs.get("return_chosen"))
    elif mode == "eval_verifier":
        score = maj_1(model_final_answer, gt_final_answer)
        results["maj@1"] = score
        results["verdict_parse_error"] = 0
        eval_score = kwargs.get("verdict")
        loc = kwargs.get("loc")
        eval_score = re.split(f"Final Verdict( A{loc})?: ", eval_score)[-1].split("\n")[0].strip()
        if eval_score.lower() == "yes" or eval_score.lower() == "correct":
            eval_score = 1
        elif eval_score.lower() == "no" or eval_score.lower() == "incorrect":
            eval_score = 0
        else:
            print(f"Eval verdict is unknown: {eval_score}!!!")
            print("---------------------")
            results["verdict_parse_error"] += 1
            eval_score = 0
        results["verifier_score"] = eval_score
        results["eval_verifier"] = int(score == eval_score)
    elif mode == "refinement":
        assert False
        draft_fa = model_final_answer
        refinement_fa = get_clean_final_answer((kwargs.get("refinement")))
        score = eval_draft_refinement(gt_fa=gt_final_answer, 
                                      draft_fa=draft_fa,
                                      refinement_fa=refinement_fa,
                                      draft_score=kwargs.get("draft_score"),
                                      refinement_score=kwargs.get("refinement_score"),)
    elif mode == "rerank":
        chosen_answer = kwargs["verdict"]
        try:
            #chosen_answer = re.match(r"Final Verdict: A(\d+)", chosen_answer).group(1)
            chosen_answer = re.split(r"[ ,]", chosen_answer.split("Final Verdict: ")[-1])[0][1:]
            chosen_answer = int(chosen_answer)
            chosen_answer = kwargs[f"model_answer_{chosen_answer}"]
            results["verdict_parse_error"] = 0
        except Exception as e:
            #print(e)
            #print(chosen_answer)
            #exit()
            chosen_answer = ""
            results["verdict_parse_error"] = 1
        model_final_answer, results["final_answer_parse_error"] = get_clean_final_answer(chosen_answer, benchmark=benchmark).values()
        results["rerank@n"] = maj_1(model_final_answer, gt_final_answer)
        # Compute maj@n baseline
        model_answers = [kwargs[k] for k in kwargs if "model_answer_" in k]
        maj_results = eval(model_answers=model_answers, gt_answer=gt_answer, mode="maj@n", benchmark=benchmark, return_chosen=True)
        results["maj@n"] = maj_results["maj@n"]
        results["rerank_maj_agreement"] = check_equals(model_final_answer, maj_results["chosen_maj"])
    else:
        raise ValueError(f"Unsupported evaluation model {mode}!")
    return results


#####Reward functions for RL#####
# These are essentially batched versions of the evaluation functions

def maj_1_reward_fn(outputs, gt_answers, benchmark="gsm8k", *args, **kwargs):
    return [eval(output, gt_ans, mode="maj@1", benchmark=benchmark, *args, **kwargs) for output, gt_ans in zip(outputs, gt_answers)]


def pass_n_reward_fn(outputs, gt_answers, num_return_sequences, benchmark="gsm8k", *args, **kwargs):
    # Batch outputs, gt_answers into chunks of size num_return_sequences
    output_chunks = split(outputs, num_return_sequences)
    gt_answer_chunks = split(gt_answers, num_return_sequences)
    scores = []
    for output_chunk, gt_answer_chunk in zip(output_chunks, gt_answer_chunks):
        score = eval(output_chunk, gt_answer_chunk[0], mode="pass@n", benchmark=benchmark, *args, **kwargs)
        scores += [score] * len(output_chunk)
    # Return a score for each output, gt_answer pair (note this will be redundant num_return_sequences times)
    return scores

def maj_n_reward_fn(outputs, gt_answers, num_return_sequences, benchmark="gsm8k", *args, **kwargs):
    # Batch outputs, gt_answers into chunks of size num_return_sequences
    output_chunks = split(outputs, num_return_sequences)
    gt_answer_chunks = split(gt_answers, num_return_sequences)
    scores = []
    for output_chunk, gt_answer_chunk in zip(output_chunks, gt_answer_chunks):
        score = eval(output_chunk, gt_answer_chunk[0], mode="maj@n", benchmark=benchmark, *args, **kwargs)
        scores += [score] * len(output_chunk)
    # Return a score for each output, gt_answer pair (note this will be redundant num_return_sequences times)
    return scores

def rerank_n_reward_fn(outputs, gt_answers, benchmark="gsm8k", *args, **kwargs):
    scores = []
    # Construct input arguments to eval
    for i, gt_answer in enumerate(gt_answers):
        print(f"Sample {i} of ", len(gt_answers))
        args = {
                "model_answers": outputs[i], 
                "gt_answer": gt_answer, 
                "mode": "rerank",
                "benchmark": benchmark,
                "verdict": kwargs["verdicts"][i]
               }
        for k in kwargs:
            if "model_answer_" in k:
                args[k] = kwargs[k][i]
        score = eval(**args)
        scores.append(score)
    return scores

def eval_draft_refinement_reward_fn(outputs, refinements, gt_answers, draft_scores, refinement_scores, benchmark="gsm8k", *args, **kwargs):
    return [eval(model_answers=draft,
                 tok_outputs=None,
                 refinement=refinement, 
                 draft_score=draft_score,
                 refinement_score=refinement_score, 
                 gt_answer=gt_answer,
                 mode="refinement",
                 benchmark=benchmark,) for draft, refinement, draft_score, refinement_score, gt_answer in zip(outputs, refinements, draft_scores, refinement_scores, gt_answers)]

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


def default_metric_fn(outputs, gt_answers, benchmark, *args, **kwargs):
    metrics = {}
    metrics.update(jsonl_to_dict(maj_1_reward_fn(outputs, gt_answers, benchmark=benchmark, *args, **kwargs)))
    if kwargs.get("num_return_sequences") is not None and kwargs["num_return_sequences"][0] > 1:
        num_return_sequences = kwargs["num_return_sequences"][0]
        metrics.update(jsonl_to_dict(pass_n_reward_fn(outputs, gt_answers, benchmark=benchmark, num_return_sequences=num_return_sequences)))
        metrics.update(jsonl_to_dict(maj_n_reward_fn(outputs, gt_answers, benchmark=benchmark, num_return_sequences=num_return_sequences)))
    if kwargs.get("save_file", None) is not None:
        return metrics, f"default_{kwargs.get('num_return_sequences')[0]}"
    else:
        return metrics


def eval_metric_fn(outputs, gt_answers, verdicts, K, solution_key, *args, **kwargs):
    metrics = {}
    solution_key = solution_key[0]
    K = K[0]
    for i in range(1, K+1):
        results = jsonl_to_dict([eval(output, gt_ans, verdict=verdict, loc=i, mode="eval_verifier", *args, **kwargs) for output, gt_ans, verdict in zip(kwargs.get(f"{solution_key}_{i}"), gt_answers, verdicts)])
        gts = results["maj@1"]
        predictions = results["verifier_score"]
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

        metrics[f"accuracy_A{i}"] = results["eval_verifier"]
        metrics[f"precision_A{i}"] = tp / (tp + fp)
        metrics[f"recall_A{i}"] = tp / (tp + fn)
        metrics[f"f1_A{i}"] = metrics[f"precision_A{i}"] * metrics[f"recall_A{i}"] / (metrics[f"precision_A{i}"] + metrics[f"recall_A{i}"])
        metrics[f"maj@1_A{i}"] = gts
        metrics[f"verifier_score_A{i}"] = predictions
        metrics[f"final_answer_parse_error_A{i}"] = results["final_answer_parse_error"]
        metrics[f"verdict_parse_error_A{i}"] = results["verdict_parse_error"]
    return metrics, ""


def refinement_metric_fn(outputs, gt_answers, benchmark, *args, **kwargs):
    assert False
    # TODO(alex): Update after refactor
    metrics = {}  
    metrics["maj@1"] = eval_draft_refinement_reward_fn(outputs=outputs, gt_answers=gt_answers, *args, **kwargs)
    metrics["parse_fail_rate"] = final_answer_parse_error / len(outputs)
    return metrics, ""


def rerank_metric_fn(outputs, gt_answers, benchmark, *args, **kwargs):
    metrics = jsonl_to_dict(rerank_n_reward_fn(outputs, gt_answers, benchmark, *args, **kwargs))
    return metrics, ""


def get_metric_fn(metric_name):
    if metric_name == "default":
        return default_metric_fn
    elif metric_name == "refinement":
        return refinement_metric_fn
    elif metric_name == "eval":
        return eval_metric_fn
    elif metric_name == "rerank":
        return rerank_metric_fn
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
             benchmark="gsm8k",
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
        if "model_answer_1" not in sample and "model_answer" in sample:
            sample["model_answer_1"] = sample["model_answer"]
        sample["solution_key"] = solution_key
        sample["tok_outputs"] = None
        sample["gt_answers"] = sample["answer"]
        sample["num_return_sequences"] = K
        sample["K"] = K
        if mode == "eval" or mode == "rerank":
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
    print("Dataset keys: ", dataset.keys())
    for k, v in dataset.items():
        if type(v[0]) is str and "<-1>" == v[0][:4]:
            # NOTE: For now just assuming dtype is always float
            dataset[k] = [decode_nested_lists(l, depth=0, dtype=float) for l in v]

    stats, metric_save_file = metric_fn(benchmark=benchmark, **dataset, save_file="")
    results = mean_acc_aggregate_fn(stats)
    print(f"Evaluating {dataset_path}...")
    print(json.dumps(results, indent=2))

    save_file = ".".join(os.path.basename(dataset_path).split(".")[:-1])
    save_file = f"{save_file}_{metric_save_file}"

    with open(f"results/{save_file}.json", "w") as f:
        json.dump(results, f, indent=2)

    if dump_labeled_dataset:
        for key, stat in stats.items():
            if type(stat) is list and len(stat) == len(dataset[get_first_key(dataset)]):
                dataset[key] = stat
            dump_jsonl(dict_to_jsonl(dataset), f"eval_data/{save_file}.jsonl")
    return results

if __name__ == "__main__":
    """ CLI utilities to evaluate offline datasets 
    Usage: python eval.py --dataset_path rollouts/gpt_3.5_turbo_math_five_sample.jsonl --K 5 --benchmark math --mode default
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--solution_key", default="model_answer", help="Key in dataset to model answers")
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--K", default=1, type=int, help="Number of candidate answers to rerank")
    parser.add_argument("--mode", default="default", choices=["default", "eval", "rerank", "refinement"])
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--dump_labeled_dataset", action="store_true")
    args = parser.parse_args()

    run_eval(**vars(args))


#llama2_7B_step_balanced_positive_params_noprop_positive_final_label_fully_consistent_negative_params_noprop_positive_final_label_fully_consistent_all_step_verified_K_60_sorm