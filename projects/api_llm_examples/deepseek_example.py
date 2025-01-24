from gptquery.gpt import GPT

deepseek_api_key = ""

task_prompt_text = "Answer the following question for me in the style of a Thomas Hume. {question}"

gpt = GPT(model_name="deepseek-reasoner", 
          max_num_tokens=512,
          keys={"DEEPSEEK_API_KEY": deepseek_api_key},
          task_prompt_text=task_prompt_text,
          backend="openai",
          url="https://api.deepseek.com",)

input = [{"question": "Find the roots of the polynomial x^2+2x=1"}]
output = gpt(input)
print(output)