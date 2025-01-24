import os
from typing import List
from time import time
import aiohttp
import uuid
from time import sleep
import threading
import asyncio

from gptquery.utils import chunk
from gptquery.logger import Logger
from gptquery.datatypes import Message, LLMRequest

from litellm import batch_completion, acompletion

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    print("vllm not found!!!")


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
        elif name == "OPENROUTER_API_KEY":
            os.environ["OPENROUTER_API_KEY"] = key
        else:
            os.environ[name] = key

# TODO: handle completions API
class GPT:
    def __init__(self, 
                 model_name,
                 temperature=0.7, 
                 max_num_tokens=4096,
                 max_model_len=None,
                 mb_size=8,
                 task_prompt_text=None,
                 logging_path=None,
                 keys=dict(),
                 verbose=False,
                 model_endpoint=None,
                 max_interactions=1,
                 retry_wait_time=None,
                 offline=False,
                 tensor_parallel_size=None,
                 dtype="auto",
                 quantization=None,
                 chat=True,
                 K=1,
                 backend="litellm", # vllm, openrouter, litellm
                 ):
        self.keys = keys
        configure_keys(keys)

        self.model_name = model_name
        self.temperature = temperature
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.mb_size = mb_size
        self.request_timeout = max(15, int(max_num_tokens / 15)) if "gpt-3.5" in model_name else max(20, int(max_num_tokens / 10)) # noqa : E501

        assert task_prompt_text is not None
        self.task_prompt_text = task_prompt_text

        self.logging_path = logging_path
        self.do_log = logging_path is not None
        self.log_identity = str(uuid.uuid4())
        self.verbose = verbose
        self.model_endpoint = model_endpoint
        self.max_interactions = max_interactions
        self.retry_wait_time = retry_wait_time
        self.offline = offline
        self.chat = chat
        self.init_K = K
        self.backend = backend if not self.offline else "vllm"
        
        # TODO: support remote completion API
        assert self.offline or self.chat
        
        # Load model if offline
        if self.offline:
            # TODO: Figure out how to do multi-model, multi-gpu offline inference
            # in the same python process
            assert tensor_parallel_size is not None
            self.llm = LLM(model=self.model_name,
                           tokenizer=self.model_name,
                           tensor_parallel_size=tensor_parallel_size,
                           dtype=dtype,
                           quantization=quantization,
                           max_model_len=max_model_len,)

        if self.logging_path is not None:
            Logger.init(logging_path, identity=self.log_identity)
            
    def offline_apply_chat_template(self, message: List[dict]):
        return self.llm.llm_engine.tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    async def async_completions(self, samples: List[LLMRequest]):
        assert self.backend == "openrouter"
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        }
        async with aiohttp.ClientSession() as session:
            async def completion(sample: LLMRequest):
                try:
                    json = {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "max_tokens": self.max_num_tokens,
                        "messages": sample.to_list(),
                    }
                    async with session.post(url, json=json, headers=headers) as response:
                        response.raise_for_status()  # Raise an exception for bad status codes
                        return await response.json()
                except Exception as e:
                    print(f"Exception: {e}")
                    return None
            tasks = [completion(sample) for sample in samples]
            results = await asyncio.gather(*tasks)
        return results

    def completions(self, samples: List[LLMRequest], is_complete_keywords: List[str]):
        if self.offline:
            assert self.backend == "vllm"
            inputs = self.offline_apply_chat_template([sample.to_list() for sample in samples]) if self.chat else [sample.to_prompt() for sample in samples]
            sampling_params = SamplingParams(temperature=self.temperature,
                                             max_tokens=self.max_num_tokens,
                                             stop=is_complete_keywords,
                                             n=self.K,)
            responses = self.llm.generate(inputs, sampling_params=sampling_params)
            responses = [[response.outputs[i].text for i in range(self.K)] for response in responses]
        else:
            if self.backend == "openrouter":
                assert self.K == 1 and len(is_complete_keywords) == 0
                responses = asyncio.run(self.async_completions(samples))
                print(responses)
                responses = [[response["choices"][0]["message"]["content"]] for response in responses]
            elif self.backend == "vllm":
                raise NotImplementedError("Online direct vllm client not supported yet.")
            elif self.backend == "litellm":
                responses = batch_completion(
                            model=self.model_name,
                            messages=[sample.to_list() for sample in samples],
                            temperature=self.temperature,
                            max_tokens=self.max_num_tokens,
                            api_base=self.model_endpoint,
                            stop=is_complete_keywords,
                            n=self.K,
                        )
                responses = [[response.choices[i].message.content for i in range(self.K)] for response in responses]
            else:
                raise ValueError(f"Unknown backend: {self.backend}!!!")
        return responses
    
    def is_complete_response(self, response, is_complete_keywords):
        if len(is_complete_keywords) == 0:
            return True
        else:
            return bool(sum([keyword in response for keyword in is_complete_keywords]))
    
    def log(self, mb):
        Logger.log(mb, identity=self.log_identity)

    def __call__(self, batch: List[dict],
                 prompt_key=None,
                 output_key="response",
                 is_complete_keywords=[],
                 keep_keywords=False,
                 messages=False,
                 K=None,):
        self.K = K if K is not None else self.init_K
        t = time()
        if prompt_key is None:
            prompt_key = "gptquery_prompt"
            remove_prompt_key = True
        else:
            remove_prompt_key = False
        # Label with unique ids and add output_key field with empty string value
        batch = [{**sample, **{"gptquery_id": i, output_key: self.K*[""]}} for i, sample in enumerate(batch)]
        mbs = chunk(batch, self.mb_size)
        for i, mb in enumerate(mbs):    
            cur_mb = mb
            for _ in range(self.max_interactions):
                if len(cur_mb) == 0: break
                # Construct LLM requests
                requests = []
                for sample in cur_mb:
                    if not messages:
                        prompt = self.task_prompt_text.format(**sample)
                        messages = [Message(content=prompt, role="user")]
                    else:
                        # each input sample should be in a list of messages
                        assert "messages" in sample
                        messages = [Message(**message) for message in sample["messages"]]
                    if len(sample[output_key][0]) > 0:
                        messages += [Message(content=sample[output_key], role="assistant"), Message(content="", role="user")]
                        # TODO(dahoas): how to update prompts if given multiple messages?
                        raise NotImplementedError
                    request = LLMRequest(messages=messages)
                    requests.append(request)
                    sample[prompt_key] = prompt
                # Send requests
                try:
                    responses = self.completions(requests, is_complete_keywords)
                except AttributeError as e:
                    if self.retry_wait_time:
                        print("Error encountered, sleeping...")
                        sleep(self.retry_wait_time)
                        responses = self.completions(requests, is_complete_keywords)
                    else:
                        raise e
                # Update output_key field
                for sample, response in zip(cur_mb, responses):
                    # Combine intermediate response with update by simple concatenation
                    sample[output_key] = [output_i + response_i for output_i, response_i in zip(sample[output_key], response)]
                # Filter out completed samples
                # TODO: do not yet support keyword stopping for K > 1
                assert not (self.K > 1 and len(is_complete_keywords) > 0)
                if self.K == 1:
                    cur_mb = [sample for sample in cur_mb if not self.is_complete_response(sample[output_key], is_complete_keywords)]
            # Remove gptquery_ids and truncate after 'is_complete_keywords'
            for sample in mb:
                sample.pop("gptquery_id")
                if remove_prompt_key:
                    sample.pop(prompt_key)
                # If K = 1 remove responses list
                if self.K == 1:
                    sample[output_key] = sample[output_key][0]
                    for keyword in is_complete_keywords:
                        if keyword in sample[output_key]:
                            sample[output_key] = sample[output_key].split(keyword)[0]
                            sample[output_key] += keyword if keep_keywords else ""
            if self.do_log:
                self.log(mb)
            if self.verbose:
                print(f"Finished batch {i} of {len(mbs)} in {(time() - t) / 60} min. ({(time() - t) / ((i+1)*60*self.mb_size)} min. per sample)")
        return batch


class GPTRouter:
    """
    Routes request to multiple GPT endpoints.
    """
    def __init__(self, 
                 gpts: List[GPT],):
        self.gpts = gpts
        self.inference_conditions = dict()

    def __call(self, 
                batch: List[dict], 
                gpt: GPT,
                results: dict,
                rank: int,):
        result = gpt(batch, **self.inference_conditions)
        results[rank] = result

    def __call__(self, 
                 batch: List[dict],
                 **kwargs):
        """
        Naively splits batch among all gpts.
        """
        self.inference_conditions = kwargs
        mb_size = (len(batch) + len(self.gpts) - 1) // len(self.gpts)
        mbs = chunk(batch, mb_size)
        results = {i: None for i in range(len(mbs))}
        threads = []
        for i, (mb, gpt) in enumerate(zip(mbs, self.gpts)):
            t = threading.Thread(target=self.__call, args=(mb, gpt, results, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return [e for l in results.values() for e in l]