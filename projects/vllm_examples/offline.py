from gptquery.gpt import GPT
from time import time
from transformers import AutoTokenizer


task_prompt_text = "{prompt}"
model_path = "Qwen/Qwen1.5-4B-Chat"
gpt = GPT(model_name=model_path,
          task_prompt_text=task_prompt_text,
          max_num_tokens=128,
          tensor_parallel_size=2,
          dtype="float16",
          offline=True,
          chat=False,)


def timed_generate(input):
    t = time()
    response = gpt(input, is_complete_keywords=["<DONE>"])[0]["response"]
    t = time() - t
    return response, t

input = [{"prompt": "Hello world!"}]
response, t = timed_generate(input)
print(response)
print(f"Generated in {t:.4f} seconds")