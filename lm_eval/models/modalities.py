
from typing import Union, List, Optional
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BatchEncoding
from modalities.models.huggingface_adapters.hf_adapter import HFAdapterConfig, HFAdapter
from .huggingface import AutoCausalLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class Modalities(AutoCausalLM):
    def __init__(self, *args, **kwargs):
        AutoConfig.register("modalities", HFAdapterConfig)
        AutoModelForCausalLM.register(HFAdapterConfig, HFAdapter)
        # TODO load our own tokenizer
        super().__init__(tokenizer="gpt2", *args, **kwargs)


    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)
