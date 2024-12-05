from typing import Union, List, Optional
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BatchEncoding
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapterConfig, HFModelAdapter
from .huggingface import HFLM
from lm_eval.api.registry import register_model

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

@register_model("modalities")
class Modalities(HFLM):
    def __init__(self, *args, **kwargs):
        AutoConfig.register("modalities", HFModelAdapterConfig)
        AutoModelForCausalLM.register(HFModelAdapterConfig, HFModelAdapter)
        # TODO load our own tokenizer
        super().__init__(tokenizer="gpt2", *args, **kwargs)

    def _model_call(
            self, inputs: TokenSequence, attn_mask = None, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)
