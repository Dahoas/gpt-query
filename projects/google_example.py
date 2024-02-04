from gptquery.gpt import GPT
from gptquery.logger import Logger


google_key = "AIzaSyARcyOX4FwQXrYBvwp1M9zNg5Ntw9uORaE"

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

'''gpt = GPT(model_name="palm/chat-bison",
          task_prompt_text=task_prompt_text,
          keys=dict(PALM_API_KEY=google_key, GEMINI_API_KEY=google_key),
          log=False,
          max_num_tokens=20,
          asynchronous=False,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)'''


gpt = GPT(model_name="gemini/gemini-pro",
          task_prompt_text=task_prompt_text,
          keys=dict(PALM_API_KEY=google_key, GEMINI_API_KEY=google_key),
          log=False,
          max_num_tokens=20,  # Some issue with passing 'max_num_tokens' to gemini-pro
          asynchronous=False,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)