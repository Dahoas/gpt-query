from gptquery.gpt import GPT

openrouter_api_key = "YOUR_API_KEY"

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="qwen/qwen-2-7b-instruct:free", 
          max_num_tokens=512,
          keys={"OPENROUTER_API_KEY": openrouter_api_key},
          task_prompt_text=task_prompt_text,
          backend="openrouter")

input = [{"question": "What is the meaning of life?"}, {"question": "What's in my pocket?"}]
output = gpt(input)
print(output)