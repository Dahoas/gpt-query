from prompts import prompts
from utils import load_jsonl, dump_jsonl, group_by_prompt
from eval import get_final_answer, clean_final_answer, eval
import random

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


def create_multi_sample_dataset():
    sample_data = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_sampled.jsonl")
    grouped_sample_data = group_by_prompt(sample_data, key="prompt")
    new_dataset = []
    for samples in grouped_sample_data.values():
        sample = {k: samples[k][0] for k in samples}
        for i, answer in enumerate(samples["model_answer"]):
            sample[f"model_answer_{i+1}"] = answer
        if len(samples["model_answer"]) == 3:
            new_dataset.append(sample)

    dump_jsonl(new_dataset, "rollouts/gpt_3.5_turbo_gsm8k_multi_sample_3.jsonl")
    print("New dataset len: ", len(new_dataset))

if __name__ == "__main__":
    create_multi_sample_dataset()