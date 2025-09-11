import types
from transformers import GenerationConfig


def apply_masked_generate(trainer, logits_processor, horizon: int, eos_token_id: int, pad_token_id: int):
    """Patch the underlying model's generate to enforce masking and default gen config.

    TRL's GRPO uses the unwrapped model for generation; patch that object's generate.
    """
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    _orig_generate = unwrapped.generate

    def _generate_with_mask(self, *g_args, **g_kwargs):
        g_kwargs = dict(g_kwargs)
        gen_cfg = g_kwargs.get("generation_config")
        if gen_cfg is None:
            gen_cfg = GenerationConfig()
        # Enforce our defaults
        gen_cfg.max_new_tokens = horizon + 1
        gen_cfg.eos_token_id = eos_token_id
        gen_cfg.pad_token_id = pad_token_id
        gen_cfg.do_sample = True
        gen_cfg.top_k = 0
        g_kwargs["generation_config"] = gen_cfg
        # Ensure our logits processor is included
        lps = g_kwargs.get("logits_processor")
        if lps is None:
            g_kwargs["logits_processor"] = [logits_processor]
        else:
            l = list(lps)
            if logits_processor not in l:
                l = [logits_processor] + l
            g_kwargs["logits_processor"] = l
        # Provide attention mask if not passed (pad==eos can confuse inference)
        if "inputs" in g_kwargs and "attention_mask" not in g_kwargs:
            import torch
            inp = g_kwargs["inputs"]
            g_kwargs["attention_mask"] = (inp != pad_token_id).long()
        # Guard against invalid probabilities by re-normalizing within allowed logits
        # (Transformers handles softmax, but masked rows can be all -inf if mis-specified.)
        if "logits_processor" not in g_kwargs or not g_kwargs["logits_processor"]:
            g_kwargs["logits_processor"] = [logits_processor]
        return _orig_generate(*g_args, **g_kwargs)

    unwrapped.generate = types.MethodType(_generate_with_mask, unwrapped)


