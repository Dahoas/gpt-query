from gptquery.pipelines import InferenceComponent
from gptquery.utils import dict_to_jsonl, jsonl_to_dict
from typing import List, Dict, Any


class JsonlInferenceComponent(InferenceComponent):
    def __init__(self, name, max_steps):
        super().__init__(name, max_steps)

    def run(self, jsonl: List[dict]):
        pass

    def __call__(self, data: Dict[str, List[Any]], start, end):
        jsonl = dict_to_jsonl(data)
        jsonl = self.run(jsonl)
        return jsonl_to_dict(jsonl), start