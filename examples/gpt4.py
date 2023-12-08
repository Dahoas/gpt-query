from gptquery.gpt import GPT
from gptquery.logger import Logger


oai_key = "INSERT_YOUR_KEY_HERE"

system_prompt_text = "You are a helpful AI assistant."
task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="gpt-4-1106-preview",
          system_prompt_text=system_prompt_text,
          task_prompt_text=task_prompt_text,
          oai_key=oai_key,)

Logger.init("temp.jsonl")

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)