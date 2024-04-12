from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


######## gpt.py datatypes ########

@dataclass
class Message:
    content: str
    role: str


@dataclass
class LLMRequest:
    messages: List[Message]

    def to_list(self) -> List[dict]:
        return [asdict(message) for message in self.messages]
    

######## composer.py datatypes ########

@dataclass
class PipelineState:
    name: str
    step: int
    batch_mask: Optional[List[bool]] = None


NestedPipelineState = List[PipelineState]


@dataclass
class ComposerStoreElement:
    id: int
    state: NestedPipelineState
    data: Dict[str, Any]