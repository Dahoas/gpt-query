from gptquery.gpt import GPT
from time import time
from transformers import AutoTokenizer


task_prompt_text = "{prompt}"

model_path = "meta-llama/Meta-Llama-3-8B"
tokenizer_path = model_path
gpt = GPT(model_name=f"openai/{model_path}",
          model_endpoint="http://atl1-1-03-006-5-0:8000/v1",
          task_prompt_text=task_prompt_text,
          max_num_tokens=4096,)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def timed_generate(input):
    t = time()
    response = gpt(input, is_complete_keyword="<DONE>")[0]["response"]
    t = time() - t
    return response, t

prompt_path = "prompts/conditional_elm_prompt.txt"
with open(prompt_path, "r") as f:
    prompt = f.read()
input = [{"prompt": prompt}]
response, t = timed_generate(input)
print(response)
print(f"Generated in {t:.4f} seconds")

input_len = len(tokenizer(prompt).input_ids)
output_len = len(tokenizer(response).input_ids)
print(f"Input len: {input_len}")
print(f"Output len: {output_len}")