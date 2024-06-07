import os
import asyncio
from typing import List
from time import time
from dataclasses import dataclass, asdict
import uuid

from gptquery.utils import chunk
from gptquery.logger import Logger
from gptquery.datatypes import Message, LLMRequest

from litellm import batch_completion, acompletion


def configure_keys(keys: dict):
    for name, key in keys.items():
        print(f"Setting {name} to {key}")
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
            os.environ[name] = key

class GPT:
    def __init__(self, 
                 model_name,
                 temperature=0.7, 
                 max_num_tokens=4096, 
                 mb_size=10,
                 task_prompt_text=None,
                 logging_path=None,
                 oai_key=None,
                 keys=dict(),
                 verbose=False,
                 asynchronous=False,
                 model_endpoint=None,
                 max_interactions=1,):
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

        self.logging_path = logging_path
        self.do_log = logging_path is not None
        self.log_identity = str(uuid.uuid4())
        self.verbose = verbose
        self.asynchronous = asynchronous
        self.model_endpoint = model_endpoint
        self.max_interactions = max_interactions
        self.azure = False

        if self.logging_path is not None:
            Logger.init(logging_path, identity=self.log_identity)

        #if "azure" in self.model_name:
        #    self.setup_azure()

    def setup_azure(self):
        # Currently only supports synchronous single sample requests for azure
        assert not self.asynchronous
        self.model_name = self.model_name.split("/")[1]
        self.mb_size = 1
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
                        api_key=os.environ["azure_openai_api_key"],  
                        api_version=os.environ["azure_api_version"],
                        azure_endpoint=os.environ["azure_endpoint"],
                        )
        self.azure = True

    def azure_completion(self, sample: LLMRequest, is_complete_keywords: List[str]):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=sample.to_list(),
            temperature=self.temperature,
            max_tokens=self.max_num_tokens,
            stop=is_complete_keywords,
        )
        return response

    def synchronous_completion(self, samples: List[LLMRequest], is_complete_keywords: List[str]):
        if self.azure:
            responses = [self.azure_completion(samples[0], is_complete_keywords)]
        else:
            responses = batch_completion(
                        model=self.model_name,
                        messages=[sample.to_list() for sample in samples],
                        temperature=self.temperature,
                        max_tokens=self.max_num_tokens,
                        api_base=self.model_endpoint,
                        stop=is_complete_keywords,
                    )
        return responses
    
    async def asynchronous_completion(self, sample: LLMRequest, is_complete_keywords: List[str]):
        responses = await acompletion(
                    model=self.model_name,
                    messages=sample.to_list(),
                    temperature=self.temperature,
                    max_tokens=self.max_num_tokens,
                    api_base=self.model_endpoint,
                    stop=is_complete_keywords,
                )
        return responses
    
    def is_complete_response(self, response, is_complete_keywords):
        if len(is_complete_keywords) == 0:
            return True
        else:
            return bool(sum([keyword in response for keyword in is_complete_keywords]))
    
    def log(self, mb):
        Logger.log(mb, identity=self.log_identity)

    def __call__(self, batch: List[dict], 
                 output_key="response",
                 is_complete_keywords=[],
                 keep_keywords=False,):
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
                # Send requests
                if self.asynchronous:
                    responses = [asyncio.run(self.asynchronous_completion(request, is_complete_keywords)) for request in requests]
                else:
                    responses = self.synchronous_completion(requests, is_complete_keywords)
                # Extract responses from response format
                responses = [response.choices[0].message.content for response in responses]
                # Update output_key field
                for sample, response in zip(cur_mb, responses):
                    # Combine intermediate response with update by simple concatenation
                    sample[output_key] = sample[output_key] + response
                # Filter out completed samples
                cur_mb = [sample for sample in cur_mb if not self.is_complete_response(sample[output_key], is_complete_keywords)]
            # Remove gptquery_ids and truncate after 'is_complete_keywords'
            for sample in mb:
                sample.pop("gptquery_id")
                for keyword in is_complete_keywords:
                    if keyword in sample[output_key]:
                        sample[output_key] = sample[output_key].split(keyword)[0]
                        sample[output_key] += keyword if keep_keywords else ""
            if self.do_log:
                self.log(mb)
            if self.verbose:
                print(f"Finished batch {i} of {len(mbs)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60*self.mb_size)} min. per sample)")
        return batch