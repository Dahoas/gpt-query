from gptquery.gpt import GPT
from gptquery.logger import Logger


google_key = "AIzaSyDqwPpd4qH6OjcgHPAozYNUt2zZNdzsiKs"

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="gemini/gemini-1.5-pro-latest",
          task_prompt_text=task_prompt_text,
          keys=dict(GEMINI_API_KEY=google_key),
          max_num_tokens=60,  # Some issue with passing 'max_num_tokens' to gemini-pro
)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)
