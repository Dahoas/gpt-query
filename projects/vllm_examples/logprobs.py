from gptquery.gpt import GPT
from time import time
from transformers import AutoTokenizer


task_prompt_text = "{prompt}"

model_path = "Qwen/Qwen1.5-4B-Chat"
tokenizer_path = model_path
gpt = GPT(model_name=f"{model_path}",
          model_endpoint="http://math-ord2-0:9000/v1",
          task_prompt_text=task_prompt_text,
          max_num_tokens=128,
          offline=True,
          tensor_parallel_size=2,
          dtype="fp16",
          )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def timed_generate(input):
    t = time()
    response = gpt(input, is_complete_keywords=["<DONE>"])[0]["response"]
    t = time() - t
    return response, t

prompt = "Hello world"
input = [{"prompt": prompt}]
response, t = timed_generate(input)
print(response)
print(f"Generated in {t:.4f} seconds")

input_len = len(tokenizer(prompt).input_ids)
output_len = len(tokenizer(response).input_ids)
print(f"Input len: {input_len}")
print(f"Output len: {output_len}")