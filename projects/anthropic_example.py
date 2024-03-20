from gptquery.gpt import GPT
from gptquery.logger import Logger


anthropic_key = "API_KEY_HERE"

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="claude-3-haiku-20240307",
          task_prompt_text=task_prompt_text,
          keys={"ANTHROPIC_API_KEY": anthropic_key},
          log=False,
          max_num_tokens=20,
          asynchronous=False,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)
