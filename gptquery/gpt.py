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
        elif name == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = key
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
                 model_endpoint=None,
                 max_interactions=2,):
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
        self.max_interactions = max_interactions

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
    
    def is_complete_response(self, response, is_complete_keyword):
        return is_complete_keyword is None or is_complete_keyword in response

    def __call__(self, batch: List[dict], 
                 output_key="response",
                 is_complete_keyword=None,):
        t = time()
        # Label with unique ids and add output_key field with empty string value
        batch = [{**sample, **{"gptquery_id": i, output_key: ""}} for i, sample in enumerate(batch)]
        mbs = chunk(batch, self.mb_size)
        for i, mb in enumerate(mbs):    
            cur_mb = mb
            for _ in range(self.max_interactions):
                if len(cur_mb) == 0: break
                # Construct LLM requests
                requests = []
                for sample in cur_mb:
                    messages = [Message(content=self.task_prompt_text.format(**sample), role="user")]
                    if len(sample[output_key]) > 0:
                        messages += [Message(content=sample[output_key], role="assistant"), Message(content="", role="user")]
                    request = LLMRequest(messages=messages)
                    requests.append(request)
                if self.asynchronous:
                    responses = [asyncio.run(self.asynchronous_completion(request)) for request in requests]
                else:
                    responses = self.synchronous_completion(requests)
                # Extract responses from response format
                responses = [response.choices[0].message.content for response in responses]
                # Update output_key field
                for sample, response in zip(cur_mb, responses):
                    # Combine intermediate response with update by simple concatenation
                    sample[output_key] = sample[output_key] + response
                # Filter out completed samples
                cur_mb = [sample for sample in cur_mb if not self.is_complete_response(sample[output_key], is_complete_keyword)]
            if self.log:
                Logger.log(mb)
            if self.verbose:
                print(f"Finished batch {i} of {len(mbs)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60*self.mb_size)} min. per sample)")
        # Remove gptquery_ids
        for sample in batch:
            sample.pop("gptquery_id")
        return batch