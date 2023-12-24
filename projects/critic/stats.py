import argparse

from utils import load_jsonl
from eval import get_clean_final_answer, eval


def check_agreements(dataset, benchmark):
    agreement = 0
    for sample in dataset:
        agreement += eval(model_answers=sample["model_answer_1"], 
                          gt_answer=sample["model_answer_2"],
                          benchmark=benchmark,)["maj@1"]
    print("Agreement rate: ", agreement / len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math"])
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset_path)
    check_agreements(dataset=dataset, benchmark=args.benchmark)