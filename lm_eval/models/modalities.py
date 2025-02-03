from typing import Union, List, Optional

import torch
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapterConfig, HFModelAdapter, HFTokenizerAdapter
from transformers import AutoConfig, AutoModelForCausalLM, BatchEncoding, AutoTokenizer

from lm_eval.api.registry import register_model
from .huggingface import HFLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


@register_model("modalities")
class Modalities(HFLM):
    def __init__(self, *args, **kwargs):
        AutoConfig.register("modalities", HFModelAdapterConfig)
        AutoModelForCausalLM.register(HFModelAdapterConfig, HFModelAdapter)

        # TODO load our own tokenizer
        AutoTokenizer.register(config_class=HFModelAdapterConfig, slow_tokenizer_class=HFTokenizerAdapter)

        super().__init__(*args, **kwargs)

    def _model_call(
            self, inputs: TokenSequence, attn_mask=None, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)
