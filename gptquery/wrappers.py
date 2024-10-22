from gptquery.pipelines import InferenceComponent, InferencePipeline
from gptquery.composer import InferenceComposer
from gptquery.utils import dict_to_jsonl, jsonl_to_dict
from typing import List, Dict, Any


class JsonlInferenceComponent(InferenceComponent):
    def __init__(self, name, max_steps=1):
        super().__init__(name, max_steps)

    def run(self, jsonl: List[dict]):
        pass

    def __call__(self, data: Dict[str, List[Any]], start, end):
        jsonl = dict_to_jsonl(data)
        jsonl = self.run(jsonl)
        return jsonl_to_dict(jsonl), start
    

class ComposerWrapper(InferenceComposer):
    """
    Wraps pipeline in dummy parent pipeline executing once.
    Useful when top level pipeline has 'max_steps' > 1
    """
    def __init__(self, inference_pipeline):
        wrapper_pipeline = InferencePipeline(name="composer_wrapper_dummy",
                                             max_steps=1,
                                             inference_components=[inference_pipeline])
        super().__init__(wrapper_pipeline)