from gptquery.gpt import GPT
from gptquery.logger import Logger


oai_key = "YOUR_OAI_KEY_HERE"

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="gpt-3.5-turbo",
          task_prompt_text=task_prompt_text,
          oai_key=oai_key,
          log=False,
          max_num_tokens=20,
          asynchronous=True,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)