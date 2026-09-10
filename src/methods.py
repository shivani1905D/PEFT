"""
Defines the 6 PEFT / quantization-integrated methods with first-class `peft` library support:
LoRA, QLoRA, Prefix-Tuning, Prompt-Tuning, P-Tuning, LoftQ.

(QA-LoRA and QDLoRA are deliberately excluded here — they have no official `peft` integration
and would need to be implemented from the original authors' research code separately. See README.)
"""

import torch
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    LoftQConfig,
    TaskType,
)


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bf16, NOT fp16 — avoids the GradScaler crash
    )


def get_method_configs() -> dict:
    loftq_config = LoftQConfig(loftq_bits=4)

    return {
        "lora": dict(
            quantize=False,
            peft_config=LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1),
        ),
        "qlora": dict(
            quantize=True,
            peft_config=LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1),
        ),
        "prefix_tuning": dict(
            quantize=False,
            peft_config=PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20),
        ),
        "prompt_tuning": dict(
            quantize=False,
            peft_config=PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20),
        ),
        "p_tuning": dict(
            quantize=False,
            peft_config=PromptEncoderConfig(
                task_type=TaskType.SEQ_CLS, num_virtual_tokens=20, encoder_hidden_size=128
            ),
        ),
        "loftq": dict(
            quantize=True,
            peft_config=LoraConfig(
                task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
                init_lora_weights="loftq", loftq_config=loftq_config,
            ),
        ),
    }
