from gptquery.gpt import GPT
from time import time
from transformers import AutoTokenizer
from gptquery.utils import setup_models
from time import sleep

import litellm

litellm.set_verbose=True

task_prompt_text = "{prompt}"

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer_path = model_path
host = False
server_param = setup_models(model_name=model_path,
                            num_servers=1,
                            gpus_per_model=2,
                            logging_folder="temp_logs/",
                            default_port=8000,
                            cuda_list=[[0,1]],
                            host=host)[0]
if host:
    sleep(60)
gpt = GPT(model_name=f"openai/{model_path}",
          model_endpoint="http://{hostname}:{port}/v1".format(**server_param),
          task_prompt_text=task_prompt_text,
          max_num_tokens=4096,)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def timed_generate(input):
    t = time()
    response = gpt(input, is_complete_keywords=["<DONE>"])[0]["response"]
    t = time() - t
    return response, t


prompt = "Hello world."
input = [{"prompt": prompt}]
response, t = timed_generate(input)
print(response)
print(f"Generated in {t:.4f} seconds")

input_len = len(tokenizer(prompt).input_ids)
output_len = len(tokenizer(response).input_ids)
print(f"Input len: {input_len}")
print(f"Output len: {output_len}")