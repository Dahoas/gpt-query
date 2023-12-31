from gptquery.gpt import GPT
from gptquery.logger import Logger
from prompts import prompts
from datasets import load_dataset

from utils import load_jsonl
from argparse import ArgumentParser


oai_key = None

def solve():
    system_prompt_text = "You are a top student solving mathematics questions. \
After any justification, print your final answer and only your final answer with no units or description after 'Final Answer: '"
    task_prompt_text = prompts["solve_gsm8k"]

    gpt = GPT(model_name="gpt-3.5-turbo-1106",
            temperature=1.0,
            system_prompt_text=system_prompt_text,
            task_prompt_text=task_prompt_text,
            oai_key=oai_key,
            mb_size=1,
            verbose=True,)

    Logger.init("rollouts/gpt_3.5_turbo_math_five_sample.jsonl")
    data = load_jsonl("benchmarks/cot_math_small.jsonl")
    # Repeat each sample 5 times
    data = [sample for sample in data for i in range(5)]
    outputs = gpt(data, output_key="model_answer")


def verify():
    #system_prompt_text = "You are a top professor evaluating math problems. \
#Print your final verdict as either 'Correct' or 'Incorrect' after the words 'Final Verdict: '"
    system_prompt_text = "You are a top professor evaluating math problems."
    task_prompt_text = prompts["prm_verifier_recheck"]

    gpt = GPT(model_name="gpt-4-1106-preview",#"gpt-3.5-turbo-1106",#"gpt-4-1106-preview",
            temperature=0,
            system_prompt_text=system_prompt_text,
            task_prompt_text=task_prompt_text,
            oai_key=oai_key,
            mb_size=1,
            verbose=True,)

    #Logger.init("rollouts/cot_gsm8k_golden_gpt_3.5_turbo_eval_single_step.jsonl")
    data = load_jsonl("rollouts/filtered_math_prm_test_2_gpt4_eval_filtered_correct_answer.jsonl")
    Logger.init("rollouts/filtered_math_prm_test_2_gpt4_eval_filtered_correct_answer_gpt_4_recheck.jsonl")
    #data = load_jsonl("rollouts/cot_gsm8k.jsonl")
    outputs = gpt(data, one_by_one=False, output_key="verdict")


def refine():
    system_prompt_text = "You are a top student solving mathematics questions. \
After any justification, print your final answer and only your final answer with no units or description after 'Final Answer: '"
    task_prompt_text = prompts["refine_gsm8k"]

    gpt = GPT(model_name="gpt-3.5-turbo-1106",
            temperature=0,
            system_prompt_text=system_prompt_text,
            task_prompt_text=task_prompt_text,
            oai_key=oai_key,
            mb_size=5,
            verbose=True,)

    Logger.init("rollouts/gpt_3.5_turbo_gsm8k_greedy_diverse_ref_verification.jsonl")
    #Logger.init("rollouts/cot_gsm8k_golden_gpt_3.5_turbo_eval_single_step.jsonl")
    data = load_jsonl("rollouts/gpt_3.5_turbo_gsm8k_greedy_diverse_ref.jsonl")
    #data = load_jsonl("rollouts/cot_gsm8k.jsonl")
    outputs = gpt(data, one_by_one=False, output_key="refinement")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--oai_key", default=None)
    args = parser.parse_args()
    oai_key = args.oai_key

    assert oai_key is not None
    #solve_gsm8k()
    #sample_gsm8k()
    verify()
    #refine_gsm8k()