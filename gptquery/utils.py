import os
import numpy as np
from collections import defaultdict
from typing import Union, List
import torch
import subprocess
import pathlib
import socket


######## gpt.py utils ########

def chunk(batch, mb_size):
    """
    Chunk 'batch' into mini-batches of size at most 'mb_size'.
    """
    num_batches = (len(batch) + mb_size - 1) // mb_size
    return [batch[i * mb_size : (i+1) * mb_size] for i in range(num_batches)]


######## composer.py utils ########


def same_state(s1, s2):
    """
    Checks if the passed nested pipeline states are the same.
    """
    if len(s1) != len(s2):
        return False
    for ss1, ss2 in zip(s1, s2):
        if ss1.name != ss2.name or ss1.step != ss2.step:
            return False
    return True


def dict_to_jsonl(d):
    """
    Converts dict format d = {"col1": ..., "col2": ..., ..., "coln": ...} to jsonl.
    """
    if len(d) == 0:
        return []
    else:
        key = list(d)[0]
        return [{k: d[k][i] for k in d.keys()} for i in range(len(d[key]))]


def jsonl_to_dict(l):
    """
    Converts jsonl format l = [{"k1": v1, "k2": v2}, ...., {"k1": v1, "k2": v2}] to dict.
    """
    if len(l) == 0:
        return {}
    else:
        return {k: [s[k] for s in l] for k in l[0].keys()}
    

######## Logger utils ########

def recursively_serialize(d: Union[list, dict]):
    primitives = [bool, int, float, str]
    containers = [list, dict]
    converter = {
        np.ndarray: list,
        set: list,
    }
    if isinstance(d, list):
        d = [recursively_serialize(e) for e in d]
    elif isinstance(d, dict):
        for k, v in d.items():
            if type(v) in primitives:
                d[k] = v
            elif type(v) in containers:
                d[k] = recursively_serialize(v)
            else:
                d[k] = converter.get(type(v), str)(v)
    return d


######## VLLM utils ########

def setup_models(model_name: str,
                 num_servers: int,
                 gpus_per_model: int,
                 logging_folder: str,
                 default_port: int=8000,
                 cuda_list: List[List[int]]=[],
                 limit_memory=None,
                 host=True,):
    server_params = []
    pathlib.Path(logging_folder).mkdir(exist_ok=True, parents=True)
    if len(cuda_list) > 0:
        assert len(cuda_list) == num_servers
        cuda_list = [[str(e) for e in l] for l in cuda_list]
    else:
        assert gpus_per_model * num_servers <= torch.cuda.device_count()
    for ind in range(num_servers):
        port = default_port + ind
        if len(cuda_list) > 0:
            gpus = ",".join(cuda_list[ind])
        else:
            gpus = ",".join([str(i + gpus_per_model * ind) for i in range(gpus_per_model)])
        command = (
                f"CUDA_VISIBLE_DEVICES={gpus} python -m vllm.entrypoints.openai.api_server "
                f"--model {model_name} "
                "--dtype=half "
                f"--port {port} "
            )
        if gpus_per_model > 1:
            command += f"--tensor-parallel-size={gpus_per_model} "
        if limit_memory:
            command += f"--gpu-memory-utilization={limit_memory} "
        logging_path = os.path.join(logging_folder, f"vllm_{ind}.log")
        if host:
            print(f"Spinning up {model_name} on {gpus}...")
            with open(logging_path, "w") as f:
                subprocess.Popen(command, shell=True, stdout=f)
        server_params.append({
            "model_name": model_name,
            "hostname": socket.gethostname(),
            "port": port,
        })
    return server_params