from gptquery.gpt import GPT
from gptquery.logger import Logger
import litellm

litellm.set_verbose=True

keys = dict(
    AZURE_API_KEY="",
    AZURE_API_BASE="",
    AZURE_API_VERSION="2024-02-01",
)

task_prompt_text = "Answer the following question for me in the style of a David Hume. {question}"

gpt = GPT(model_name="azure/gpt-4-turbo",
          task_prompt_text=task_prompt_text,
          keys=keys,
          max_num_tokens=20,)

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)
