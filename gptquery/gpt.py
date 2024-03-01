import os
import asyncio
from typing import List
from time import time
from dataclasses import dataclass, asdict

from gptquery.utils import chunk
from gptquery.logger import Logger

from litellm import batch_completion, acompletion


@dataclass
class Message:
    content: str
    role: str


@dataclass
class LLMRequest:
    messages: List[Message]

    def to_list(self) -> List[dict]:
        return [asdict(message) for message in self.messages]


def configure_keys(keys: dict):
    for name, key in keys.items():
        if name == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = key
        elif name == "GOOGLE_API_KEY":
            import google.generativeai as genai
            genai.configure(api_key=key)
        elif name == "PALM_API_KEY":
            os.environ["PALM_API_KEY"] = key
        elif name == "GEMINI_API_KEY":
            os.environ["GEMINI_API_KEY"] = key
        else:
            raise ValueError(f"Unknown key: {name}!!!")


class GPT:
    def __init__(self, 
                 model_name,
                 temperature=0.7, 
                 max_num_tokens=4096, 
                 mb_size=10,
                 task_prompt_text=None,
                 log=True,
                 logging_path=None,
                 oai_key=None,
                 keys=dict(),
                 verbose=False,
                 asynchronous=False,
                 model_endpoint=None,):
        if oai_key is not None:
            keys["OPENAI_API_KEY"] = oai_key
        configure_keys(keys)

        self.model_name = model_name
        self.temperature = temperature
        self.max_num_tokens = max_num_tokens
        self.mb_size = mb_size
        self.request_timeout = max(15, int(max_num_tokens / 15)) if "gpt-3.5" in model_name else max(20, int(max_num_tokens / 10)) # noqa : E501

        assert task_prompt_text is not None
        self.task_prompt_text = task_prompt_text

        self.log = log
        self.verbose = verbose
        self.asynchronous = asynchronous
        self.model_endpoint = model_endpoint

        if logging_path is not None:
            Logger.init(logging_path)

    def synchronous_completion(self, samples: List[LLMRequest]):
        responses = batch_completion(
                    model=self.model_name,
                    messages=[sample.to_list() for sample in samples],
                    temperature=self.temperature,
                    max_tokens=self.max_num_tokens,
                    api_base=self.model_endpoint,
                )
        return responses
    
    async def asynchronous_completion(self, sample: LLMRequest):
        responses = await acompletion(
                    model=self.model_name,
                    messages=sample.to_list(),
                    temperature=self.temperature,
                    max_tokens=self.max_num_tokens,
                    api_base=self.model_endpoint,
                )
        return responses

    def __call__(self, batch: List[dict], one_by_one=False, output_key="response"):
        t = time()
        if one_by_one:
            for i, sample in enumerate(batch):
                requests = [LLMRequest(messages=[Message(content=self.task_prompt_text.format(**sample), role="user")])]
                responses = self.synchronous_completion(requests)
                sample[output_key] = responses[0].choices[0].message.content
                if self.log:
                    Logger.log([sample])
                if self.verbose:
                    print(f"Finished batch {i} of {len(batch)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60)} min. per sample)")
        else:
            mbs = chunk(batch, self.mb_size)
            for i, mb in enumerate(mbs):
                requests = [LLMRequest(messages=[Message(content=self.task_prompt_text.format(**sample), role="user")]) for sample in mb]
                if self.asynchronous:
                    batch_responses = [asyncio.run(self.asynchronous_completion(request)) for request in requests]
                else:
                    batch_responses = self.synchronous_completion(requests)
                for sample, response in zip(mb, batch_responses):
                    sample[output_key] = response.choices[0].message.content
                if self.log:
                    Logger.log(mb)
                if self.verbose:
                    print(f"Finished batch {i} of {len(mbs)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60*self.mb_size)} min. per sample)")
        return batch