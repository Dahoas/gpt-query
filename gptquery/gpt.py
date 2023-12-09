import os
import asyncio
from typing import List
from time import time

from gptquery.utils import chunk
from gptquery.logger import Logger

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain


class GPT:
    def __init__(self, 
                 model_name,
                 temperature=0.7, 
                 max_num_tokens=4096, 
                 mb_size=10,
                 system_prompt_text="You are a helpful AI assistant.",
                 task_prompt_text=None,
                 log=True,
                 oai_key=None,
                 verbose=False,):
        assert oai_key is not None or os.environ.get("OPENAI_API_KEY") is not None
        os.environ["OPENAI_API_KEY"] = oai_key

        self.model_name = model_name
        self.temperature = temperature
        self.max_num_tokens = max_num_tokens
        self.mb_size = mb_size
        self.request_timeout = max(15, int(max_num_tokens / 15)) if "gpt-3.5" in model_name else max(20, int(max_num_tokens / 10)) # noqa : E501
        self.endpoint = ChatOpenAI(request_timeout=self.request_timeout, model_name=model_name, temperature=temperature, max_tokens=max_num_tokens)

        # Set system prompt and task format
        self.system_prompt_text = system_prompt_text
        self.system_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt_text)
        assert task_prompt_text is not None
        # NOTE: This assumes using chat model gpt-3.5 or gpt-4
        self.task_prompt_text = task_prompt_text
        self.task_prompt = HumanMessagePromptTemplate.from_template(task_prompt_text)
        self.prompt_template = ChatPromptTemplate.from_messages([self.system_prompt, self.task_prompt])
        self.agent = LLMChain(llm=self.endpoint, prompt=self.prompt_template)

        self.log = log
        self.verbose = verbose

    def __call__(self, batch: List[dict], one_by_one=False, output_key="response"):
        t = time()
        if one_by_one:
            for i, sample in enumerate(batch):
                response = self.agent.apply([sample])
                sample[output_key] = response[0]["text"]
                if self.log:
                    Logger.log([sample])
                if self.verbose:
                    print(f"Finished batch {i} of {len(batch)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60)} min. per sample)")
        else:
            mbs = chunk(batch, self.mb_size)
            for i, mb in enumerate(mbs):
                batch_responses = asyncio.run(self.agent.aapply(mb))
                for sample, response in zip(mb, batch_responses):
                    sample[output_key] = response["text"]
                if self.log:
                    Logger.log(mb)
                if self.verbose:
                    print(f"Finished batch {i} of {len(mbs)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60*self.mb_size)} min. per sample)")
        return batch