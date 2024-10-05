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
          chat=True,)


def timed_generate(input, K=1):
    t = time()
    responses = gpt(input, K=K)
    responses = [sample["response"] for sample in responses]
    t = time() - t
    return responses, t

input = [{"prompt": "Hello world!"}, {"prompt": "Goodbye world!"}]
responses, t = timed_generate(input)
print(responses)
print(f"Generated in {t:.4f} seconds")

# sample K = 5 responses per prompt
responses, t = timed_generate(input, K=5)
print(responses)
print(f"Generated in {t:.4f} seconds")