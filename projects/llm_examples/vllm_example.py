from gptquery.gpt import GPT
from gptquery.logger import Logger


task_prompt_text = "Answer the following question for me in the style of a David Hume. {question}"

gpt = GPT(model_name="openai/meta-llama/Meta-Llama-3-8B-Instruct",
          model_endpoint="http://atl1-1-03-002-29-0:8000/v1",
          task_prompt_text=task_prompt_text,
          max_num_tokens=20,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)
