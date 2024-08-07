from gptquery import GPT, GPTRouter
from gptquery.utils import setup_models
from time import time, sleep


model_name = "gpt2"
num_servers = 2
cuda_list = [[0], [0]]
gpus_per_model = 1
logging_folder = "test"
server_parms = setup_models(model_name=model_name,
                            num_servers=num_servers,
                            gpus_per_model=gpus_per_model,
                            logging_folder=logging_folder,
                            cuda_list=cuda_list,
                            limit_memory=0.25,
                            host=False,)
gpts = []
#sleep(30)
for i, params in enumerate(server_parms):
    print(params)
    gpts.append(GPT(model_name="openai/{model_name}".format(**params),
                    task_prompt_text="{prompt}",
                    model_endpoint="http://{hostname}:{port}/v1".format(**params),
                    max_num_tokens=128))
gpt_router = GPTRouter(gpts)
num_requests = 5000
requests = [{"prompt": "Hello how are you?"} for _ in range(num_requests)]
start = time()
results = gpt_router(requests)
elapsed = time() - start
print("Elapsed time: ", elapsed)