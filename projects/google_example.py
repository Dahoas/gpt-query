from gptquery.gpt import GPT
from gptquery.logger import Logger


google_key = "YOUR_GOOGLE_KEY_HERE"

task_prompt_text = "Answer the following question for me in the style of a David Hume. {question}"

gpt = GPT(model_name="gemini/gemini-pro",
          task_prompt_text=task_prompt_text,
          keys=dict(PALM_API_KEY=google_key, GEMINI_API_KEY=google_key),
          log=False,
          max_num_tokens=20,  # Some issue with passing 'max_num_tokens' to gemini-pro
          asynchronous=False,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)
