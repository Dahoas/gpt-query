from prompts import prompts
from utils import load_jsonl, dump_jsonl, group_by_prompt
from eval import get_clean_final_answer, eval
import random
from collections import defaultdict
import argparse
from tqdm import tqdm


def t1():
    example = {
    "question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?",
    "answer": "First find how many gigabytes are in 40% of the file: 200 GB * 40% = <<200*40*.01=80>>80 GB\nThen divide that number by the download rate to find the time until Windows restarts: 80 GB / 2 GB/minute = <<80/2=40>>40 minutes\nThen find the time to download the whole file after the restart: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, to download the whole file, and to wait for Windows to update: 40 minutes + 100 minutes + 20 minutes = <<40+100+20=160>>160 minutes\n#### 160",
    "prompt": "Q: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?\nA: ",
    "model_answer": "First, let's find out how much of the file Carla was able to download before the restart. 40% of 200 GB is 80 GB. \n\nAt a rate of 2 GB/minute, it would take Carla 80 GB / 2 GB/minute = 40 minutes to download the first 80 GB.\n\nAfter the restart, Carla has to start the download from the beginning, so she will need to download the full 200 GB.\n\nAt a rate of 2 GB/minute, it would take Carla 200 GB / 2 GB/minute = 100 minutes to download the remaining 200 GB.\n\nHowever, we need to add the 20 minutes for the Windows update restart.\n\nSo, the total time it takes to download the file is 40 minutes (first part) + 20 minutes (restart) + 100 minutes (second part) = 160 minutes.\n\nFinal Answer: It takes Carla 160 minutes to download the file.",
    "gpt4_default_eval": "The justification provided in the answer has a logical error. The time spent downloading the first 80 GB before the restart should not be included in the total time calculation because that progress was lost. Carla has to download the full 200 GB after the restart.\n\nThe correct calculation should only consider the time it takes to download the full 200 GB after the restart and the time taken for the restart itself.\n\nAt a rate of 2 GB/minute, it would take Carla 200 GB / 2 GB/minute = 100 minutes to download the full file.\n\nAdding the 20 minutes for the Windows update restart, the total time it takes to download the file is 100 minutes (for the full file after restart) + 20 minutes (restart) = 120 minutes.\n\nFinal Verdict: Incorrect",
    "K": 1,
    "outputs": "First, let's find out how much of the file Carla was able to download before the restart. 40% of 200 GB is 80 GB. \n\nAt a rate of 2 GB/minute, it would take Carla 80 GB / 2 GB/minute = 40 minutes to download the first 80 GB.\n\nAfter the restart, Carla has to start the download from the beginning, so she will need to download the full 200 GB.\n\nAt a rate of 2 GB/minute, it would take Carla 200 GB / 2 GB/minute = 100 minutes to download the remaining 200 GB.\n\nHowever, we need to add the 20 minutes for the Windows update restart.\n\nSo, the total time it takes to download the file is 40 minutes (first part) + 20 minutes (restart) + 100 minutes (second part) = 160 minutes.\n\nFinal Answer: It takes Carla 160 minutes to download the file.",
    "tok_outputs": None,
    "gt_answers": "First find how many gigabytes are in 40% of the file: 200 GB * 40% = <<200*40*.01=80>>80 GB\nThen divide that number by the download rate to find the time until Windows restarts: 80 GB / 2 GB/minute = <<80/2=40>>40 minutes\nThen find the time to download the whole file after the restart: 200 GB / 2 GB/minute = <<200/2=100>>100 minutes\nThen add the time to download 40% of the file, to download the whole file, and to wait for Windows to update: 40 minutes + 100 minutes + 20 minutes = <<40+100+20=160>>160 minutes\n#### 160",
    "num_return_sequences": 1,
    "verdicts": "The justification provided in the answer has a logical error. The time spent downloading the first 80 GB before the restart should not be included in the total time calculation because that progress was lost. Carla has to download the full 200 GB after the restart.\n\nThe correct calculation should only consider the time it takes to download the full 200 GB after the restart and the time taken for the restart itself.\n\nAt a rate of 2 GB/minute, it would take Carla 200 GB / 2 GB/minute = 100 minutes to download the full file.\n\nAdding the 20 minutes for the Windows update restart, the total time it takes to download the file is 100 minutes (for the full file after restart) + 20 minutes (restart) = 120 minutes.\n\nFinal Verdict: Incorrect",
    "accuracy": 0
    }


    system_prompt_text = "You are a top professor evaluating math problems. \
    Print your final verdict as either 'Correct' or 'Incorrect' after the words 'Final Verdict: '."
    examples = load_jsonl("eval_data/gpt_4_turbo_gsm8k_eval_default_.jsonl")
    examples = [example for example in examples if example["accuracy"] == 1]
    random.shuffle(examples)
    incorrect_solutions = [example for example in examples if example["verdicts"].split("Final Verdict: ")[-1] == "Incorrect"]
    def make_example(sample):
        text = sample["prompt"] + sample["answer"] + "\n\n" + sample["verdicts"]
        return text
    ex1 = make_example(examples[0])
    ex2 = make_example(examples[1])
    ex3 = make_example(incorrect_solutions[0])
    few_shot = f"Below are some examples of correct and incorrect Q, A pairs from the same class of problems. \
    {ex1}\n\n\
    {ex2}\n\n\
    {ex3}\n\n"
    prompt = system_prompt_text + "\n\n" + few_shot + "\n\n" + prompts["eval_gsm8k_default"].format(prompt=example["prompt"], model_answer=example["model_answer"])

    with open("scratch/scratch.txt", "w") as f:
        f.write(prompt)


    def make_example(sample):
        text = sample["prompt"] + sample["answer"] + "\n\n" + "Final Verdict: " + example["verdicts"]
        return text

    text_examples = ""
    for i in range(3):
        text_examples += make_example(examples[i]) + "\n\n"
        text_examples += make_example(incorrect_solutions[i]) + "\n\n"

    intro = "Below are examples of Q, A pairs with labels for correctness. Using the examples, \
    specify criteria to help identify whether a new candidate solution is correct."
    prompt = intro + "\n\n" + text_examples

    with open("scratch/criteria.txt", "w") as f:
        f.write(prompt)


#####Make dataset of references and model solutions#####


def t3():
    model_answers = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_greedy.jsonl")

    for sample in model_answers:
        sample["model_answer_1"] = sample["answer"]
        sample["model_answer_2"] = sample["model_answer"]

    dump_jsonl(model_answers, "rollouts/gpt_3.5_turbo_gsm8k_greedy_ref_answer.jsonl")

def t4():
    sample_data = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_sampled.jsonl")
    grouped_sample_data = group_by_prompt(sample_data, key="prompt")
    model_answers = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_greedy.jsonl")
    diff_fa_cnt = 0
    diff_a_cnt = 0
    failure = 0
    for sample in model_answers:
        chosen_sample = None
        try:
            samples = grouped_sample_data[sample["prompt"]]
            for answer in samples["model_answer"]:
                if not eval(model_answers=answer, tok_outputs=None, gt_answer=sample["model_answer"]):
                    chosen_sample = answer
                    diff_fa_cnt += 1
                    break
            if chosen_sample is None:
                for answer in samples["model_answer"]:
                    if answer != sample["model_answer"]:
                        chosen_sample = answer
                        diff_a_cnt += 1
                        break
            if chosen_sample is None:
                chosen_sample = samples["model_answer"][0]
        except Exception:
            chosen_sample = sample["model_answer"]
            failure += 1
        sample["model_answer_1"] = chosen_sample
        sample["model_answer_2"] = sample["model_answer"]

    dump_jsonl(model_answers, "rollouts/gpt_3.5_turbo_gsm8k_greedy_diverse_ref.jsonl")
    print("diff_fa_cnt: ", diff_fa_cnt)
    print("diff_a_cnt: ", diff_a_cnt)
    print("fail: ", failure)


def prepare_refinement_for_eval():
    dataset = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_eval_draft_refine.jsonl")
    for sample in dataset:
        sample["refinements"] = sample["refinement"]
        verdict = sample["verdict"]
        sample["draft_scores"] = int(verdict.split("Final Verdict A1: ")[-1].split("\n")[0].lower() == "correct")
        sample["refinement_scores"] = int(verdict.split("Final Verdict A2: ")[-1].lower() == "correct")
    dump_jsonl(dataset, "rollouts/gpt_3.5_turbo_gsm8k_eval_draft_refine.jsonl")



def choose_samples(samples, gt, num, mode, benchmark):
    if num == 0:
        return []
    if mode == "default":
        chosen_samples = samples
    if mode == "diverse":
        fas = [get_clean_final_answer(answer=answer, benchmark=benchmark)["final_answer"] for answer in samples]
        score_map = defaultdict(list)
        for answer, fa in zip(samples, fas):
            score_map[fa].append(answer)
        chosen_samples = []
        while len(score_map) > 0:
            to_remove = []
            for key in score_map:
                chosen_samples.append(score_map[key].pop(0))
                if len(score_map[key]) == 0:
                    to_remove.append(key)
            [score_map.pop(key) for key in to_remove]
    elif mode == "correctness_balance":
        scores = [eval(model_answers=answer, gt_answer=gt, benchmark=benchmark) for answer in samples]
        correct_samples = [sample for sample, score in zip(samples, scores) if score == 1]
        incorrect_samples = [sample for sample, score in zip(samples, scores) if score == 0]
        chosen_samples = []
        while len(chosen_samples) < num and len(correct_samples) > 0 and len(incorrect_samples) > 0:
            chosen_samples.append(correct_samples.pop(0))
            chosen_samples.append(incorrect_samples.pop(0))
        chosen_samples += choose_samples(correct_samples + incorrect_samples, gt, num - len(chosen_samples), mode="diverse", benchmark=benchmark)
    else:
        raise ValueError(f"Unknown mode: {mode}!!!")
    chosen_samples = chosen_samples[:num]
    random.shuffle(chosen_samples)
    return chosen_samples


def create_multi_sample_dataset(dataset_path, num_samples, mode, benchmark):
    sample_data = load_jsonl(dataset_path)
    print("Old dataset len: ", len(sample_data))
    grouped_sample_data = group_by_prompt(sample_data, key="prompt")
    print("Num groups: ", len(grouped_sample_data))
    new_dataset = []
    for samples in tqdm(grouped_sample_data.values()):
        sample = {k: samples[k][0] for k in samples}
        chosen_samples = choose_samples(samples=samples["model_answer"], gt=samples["answer"][0], num=num_samples, mode=mode, benchmark=benchmark)
        for i, answer in enumerate(chosen_samples):
            sample[f"model_answer_{i+1}"] = answer
        if len(chosen_samples) == num_samples:
            new_dataset.append(sample)
        else:
            print("Len chosen samples: ", len(chosen_samples), "num samples: ", num_samples)

    file_name = f"rollouts/gpt_3.5_turbo_{benchmark}_multi_sample_{num_samples}_{mode}.jsonl"
    print(f"Dumping new dataset at {file_name}")
    dump_jsonl(new_dataset, file_name)
    print("New dataset len: ", len(new_dataset))


def filter_no_answer(dataset_path, benchmark, **kwargs):
    dataset = load_jsonl(dataset_path)
    new_dataset = []
    for sample in dataset:
        pass_n_score = eval(model_answers=[sample[k] for k in sample if "model_answer_" in k],
                     gt_answer=sample["answer"],
                     benchmark=benchmark,
                     mode="pass@n",)["pass@n"]
        if pass_n_score:
            new_dataset.append(sample)
    dataset_path = dataset_path.split(".jsonl")[0]
    dump_jsonl(new_dataset, f"{dataset_path}_filtered_no_answer.jsonl")


def update():
    benchmark_path = "rollouts/filtered_math_prm_test_2_multi_sample.jsonl"
    verifier_path = "old_rollouts/filtered_math_prm_test_2_multi_sample_eval_gpt_4_old.jsonl"

    benchmark = load_jsonl(benchmark_path)
    verifier = load_jsonl(verifier_path)
    benchmark = {s["prompt"]: s for s in benchmark}
    verifier = {s["prompt"]: s for s in verifier}

    for prompt, sample in benchmark.items():
        verifier_sample = verifier[prompt]
        sample["verdict"] = verifier_sample["verdict"]

    dump_jsonl(list(benchmark.values()), "filtered_math_prm_test_2_multi_sample_eval_gpt_4.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_samples", default=2, type=int)
    parser.add_argument("--mode", default="default", choices=["default", "diverse", "correctness_balance"])
    parser.add_argument("--benchmark", default="math", choices=["gsm8k", "math"])

    args = parser.parse_args()

    update()
    #filter_no_answer(**vars(args))
    #create_multi_sample_dataset(**vars(args))


# "rollouts/gpt_3.5_turbo_math_five_sample.jsonl"