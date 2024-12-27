"""
Offline (no environment) LLM inference script
"""

import argparse
import json
from typing import List
from time import sleep
from datasets import load_dataset

from gptquery import GPT
from gptquery.utils import setup_models, load_jsonl, parse_list_of_int_lists, dynamic_import


def filter_with_output(data, output_path, id_key):
    try:
        outputs = load_jsonl(output_path)
    except FileNotFoundError:
        print(f"Could not find {output_path}...inputs not filtered")
        return data
    input_ids = set([sample[id_key] for sample in data])
    output_ids = set([sample[id_key] for sample in outputs])
    remaining_input_ids = input_ids.difference(output_ids)
    # assumes data has contiguous slice of ids
    if isinstance(data[0][id_key], int):
        assert data[0][id_key] + len(data) - 1 == data[-1][id_key]
        offset = data[0][id_key]
        return [data[ind-offset] for ind in remaining_input_ids]
    else:
        return [sample for sample in data if sample[id_key] in remaining_input_ids]


def run(input_path: str, 
        output_path: str,
        split: str,
        lower: int,
        upper: int,
        id_key: str,
        keys: str,
        max_num_tokens: int,
        temperature: float,
        mb_size: int,
        K: int,
        server_params: List[dict],
        local: bool,
        offline: bool,
        gpus_per_model: int,
        chat: bool,
        max_model_len: int,
        prompt_file: str,
        prompt_key: str,
        **kwargs,):
    if ".jsonl" in input_path:
        data = load_jsonl(input_path)[lower:upper]  # assumes problems is given in 'question' field
    else:
        data = list(load_dataset(input_path)[split])[lower:upper]
    print(f"Loaded {len(data)} prompts...")
    data = filter_with_output(data, output_path, id_key)
    print(f"{len(data)} prompts remaining after filtration...")

    # set up keys
    try:
        with open(keys, "r") as f:
            keys = json.load(f)
    except FileNotFoundError:
        keys = {"OPENAI_API_KEY": "1"}

    # retrieve prompt
    prompts = dynamic_import(prompt_file, "prompts")
    task_prompt_text = prompts[prompt_key]
    
    # set up gpt
    assert len(server_params) == 1
    server_param = server_params[0]
    model_name = "openai/{model_name}".format(**server_param) if local and not offline else server_param["model_name"]
    model = GPT(model_name=model_name,
                model_endpoint="http://{hostname}:{port}/v1".format(**server_param),
                task_prompt_text=task_prompt_text,
                max_num_tokens=max_num_tokens,
                keys=keys,
                temperature=temperature,
                logging_path=output_path,
                mb_size=mb_size,
                offline=offline,
                tensor_parallel_size=gpus_per_model,
                dtype="float16",
                chat=chat,
                max_model_len=max_model_len)

    # query LLM
    if len(data) > 0:
        data = model(data, K=K)
    print("Finished inference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to jsonl containing input data.")
    parser.add_argument("--output_path", type=str, help="Path to output file.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lower", type=int, default=0)
    parser.add_argument("--upper", type=int, default=1_000_000_000)
    parser.add_argument("--id_key", type=str, default="inference_id")

    parser.add_argument("--model_name", default="casperhansen/llama-3-70b-instruct-awq")
    parser.add_argument("--keys", default="keys.json", help="Path to json containing API keys.")
    parser.add_argument("--max_num_tokens", default=2048, type=int, help="Num output tokens.")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--mb_size", type=int, default=1, help="Size of mini-batch querying LLM.")
    parser.add_argument("--K", type=int, default=1, help="Number of samples per prompt")
    
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--num_servers", default=1, type=int)
    parser.add_argument("--gpus_per_model", type=int)
    parser.add_argument("--cuda_logging_folder", type=str, default="vllm_logs/")
    parser.add_argument("--cuda_list", type=parse_list_of_int_lists)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sleep_time", default=None, type=int)
    parser.add_argument("--host", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max_model_len", default=None, type=int)
    parser.add_argument("--prompt_file", type=str, help="Path to python prompts file")
    parser.add_argument("--prompt_key", type=str, default="default")

    args = parser.parse_args()
    args.server_params = setup_models(model_name=args.model_name,
                                 num_servers=args.num_servers,
                                 gpus_per_model=args.gpus_per_model,
                                 logging_folder=args.cuda_logging_folder,
                                 default_port=args.port,
                                 cuda_list=args.cuda_list,
                                 host=args.host,
                                 max_model_len=args.max_model_len)
    if args.sleep_time:
        sleep(args.sleep_time)

    run(**vars(args))