from gptquery.gpt import GPT
from gptquery.logger import Logger


oai_key = "YOUR_OAI_KEY_HERE"
task_prompt_text = "{request}"

gpt = GPT(model_name="gpt-3.5-turbo",
          task_prompt_text=task_prompt_text,
          oai_key=oai_key,
          log=False,
          max_num_tokens=20,
          max_interactions=3,)

is_complete_keyword = "<DONE>"
input = [{"request": f"Count to 20. When you are done at the very end of your response print {is_complete_keyword}."},
         {"request": f"Count to 10. When you are done at the very end of your response print {is_complete_keyword}."}]
output = gpt(input, is_complete_keyword=is_complete_keyword)
print(output)
