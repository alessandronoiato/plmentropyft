"""
ProGen2 Model Wrapper

This module provides a wrapper for loading and using ProGen2 models
following SGPO's generation pipeline.
"""

import os
import sys
from typing import List, Optional

import torch


class ProGen2Wrapper:
    """
    Wrapper for ProGen2 model matching SGPO's generation pipeline.
    
    SGPO uses a fine-tuned ProGen2-base model (from jsunn-y/ProCALM).
    The tokenizer uses:
        - Token 3 = '1' (start/BOS context)
        - Token 4 = '2' (end/EOS)
        - Tokens 5-29 = amino acids
    
    Generation:
        - Context: '1' (start token)
        - Temperature: 1.0
        - Top-p (nucleus): 0.95
        - Max length: ~1.25 * seq_len
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sgpo_repo: Optional[str] = None,
    ):
        """
        Load ProGen2 model and tokenizer.
        
        Args:
            model_path: Path to fine-tuned model (e.g., checkpoints/causalLM_finetune/TrpB/best/)
            device: Device to run on
            sgpo_repo: Path to SGPO repo (for loading tokenizer from models/pretraining/model/progen2/)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pad_token_id = 0
        self.bos_token_id = 1  # '<|bos|>'
        self.eos_token_id = 2  # '<|eos|>'
        self.start_token_id = 3  # '1' - used as context for generation
        self.end_token_id = 4    # '2' - generation terminal
        
        # Try to load from SGPO-style path
        if model_path and os.path.isdir(model_path):
            self._load_model(model_path, sgpo_repo)
        else:
            print(f"[ProGen2Wrapper] No model found at {model_path}")
    
    def _load_model(self, model_path: str, sgpo_repo: Optional[str] = None):
        """Load model and tokenizer using SGPO's model class (has generate() via PreTrainedModel).
        
        Args:
            model_path: Path to model checkpoint
            sgpo_repo: Path to SGPO repo (required for model class)
        """
        import transformers
        from transformers import PreTrainedTokenizerFast
        
        # SGPO repo is required for their custom model class
        if not sgpo_repo or not os.path.isdir(sgpo_repo):
            raise ValueError(f"SGPO repo required but not found: {sgpo_repo}")
        
        # Add SGPO's model path to sys.path
        sgpo_model_path = os.path.join(sgpo_repo, "models", "pretraining", "model")
        if sgpo_model_path not in sys.path:
            sys.path.insert(0, sgpo_model_path)
        
        try:
            # Import SGPO's model class directly (like they do in models/causalLM.py)
            from progen2.model import ProGenForCausalLM as _OriginalProGenForCausalLM
            from progen2.tokenizer import get_tokenizer, PAD_TOKEN_ID
            from transformers import GenerationMixin
            
            # SGPO's ProGenForCausalLM doesn't include GenerationMixin (older transformers API)
            # We need to create a subclass that adds it
            if GenerationMixin not in _OriginalProGenForCausalLM.__mro__:
                print(f"[ProGen2Wrapper] Adding GenerationMixin to ProGenForCausalLM")
                
                class ProGenForCausalLM(_OriginalProGenForCausalLM, GenerationMixin):
                    """ProGenForCausalLM with GenerationMixin for generate() support."""
                    pass
            else:
                ProGenForCausalLM = _OriginalProGenForCausalLM
            
            print(f"[ProGen2Wrapper] Loading model from {model_path}")
            
            # Load model with the enhanced class
            self.model = ProGenForCausalLM.from_pretrained(model_path).to(self.device)
            
            # Set architectures for TRL compatibility (it reads config.architectures[0])
            if not hasattr(self.model.config, 'architectures') or not self.model.config.architectures:
                self.model.config.architectures = ["ProGenForCausalLM"]
            
            # Add missing config attributes that TRL expects
            import transformers as _tf
            if not hasattr(self.model.config, 'transformers_version'):
                self.model.config.transformers_version = _tf.__version__
            if not hasattr(self.model.config, '_name_or_path'):
                self.model.config._name_or_path = model_path
            
            # Register the class on transformers module for TRL compatibility
            # TRL does: architecture = getattr(transformers, config.architectures[0])
            transformers.ProGenForCausalLM = ProGenForCausalLM
            setattr(transformers, "ProGenForCausalLM", ProGenForCausalLM)
            
            # Also register the config class
            from progen2.configuration_progen import ProGenConfig
            transformers.ProGenConfig = ProGenConfig
            setattr(transformers, "ProGenConfig", ProGenConfig)
            
            # Load tokenizer
            raw_tokenizer = get_tokenizer()
            self._raw_tokenizer = raw_tokenizer
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                bos_token="1",
                eos_token="2",
                pad_token="<|pad|>",
            )
            self.pad_token_id = PAD_TOKEN_ID
            
            print(f"[ProGen2Wrapper] Model loaded successfully")
            print(f"[ProGen2Wrapper] Model has generate(): {hasattr(self.model, 'generate')}")
            
        except ImportError as e:
            print(f"[ProGen2Wrapper] Could not import from SGPO repo: {e}")
            raise
    
    def sample(
        self,
        num_return_sequences: int = 40,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_length: int = 487,  # 1.25 * 389
    ) -> List[str]:
        """
        Generate protein sequences matching SGPO's sampling method.
        
        Args:
            num_return_sequences: Batch size for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_length: Maximum sequence length
        
        Returns:
            List of generated sequences (cleaned, with terminal tokens removed)
        """
        if self.model is None:
            print("[ProGen2Wrapper] No model loaded, returning empty list")
            return []
        
        self.model.eval()
        
        with torch.no_grad():
            # Start with context token '1' (token_id=3)
            # Use raw tokenizer for encoding if available
            raw_tok = getattr(self, '_raw_tokenizer', None) or self.tokenizer
            if raw_tok is not None:
                if hasattr(raw_tok, 'encode') and hasattr(raw_tok.encode("1"), 'ids'):
                    # Raw tokenizer
                    input_ids = torch.tensor(raw_tok.encode("1").ids).view(1, -1).to(self.device)
                else:
                    # PreTrainedTokenizerFast
                    input_ids = torch.tensor(raw_tok.encode("1")).view(1, -1).to(self.device)
            else:
                input_ids = torch.tensor([[self.start_token_id]]).to(self.device)
            
            # Generate
            tokens_batch = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                max_length=max_length,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.end_token_id,  # '2' = 4
            )
            
            # Decode using raw tokenizer if available
            if raw_tok is not None and hasattr(raw_tok, 'decode_batch'):
                as_lists = lambda batch: [batch[i].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
                sequences = raw_tok.decode_batch(as_lists(tokens_batch))
            elif self.tokenizer is not None:
                sequences = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in tokens_batch]
            else:
                # Fallback decoding
                sequences = []
                vocab_inv = {0: "", 1: "", 2: "", 3: "1", 4: "2"}
                for aa, idx in zip("ABCDEFGHIKLMNOPQRSTUVWXYZ", range(5, 30)):
                    vocab_inv[idx] = aa
                for seq_ids in tokens_batch:
                    seq = "".join(vocab_inv.get(int(t), "") for t in seq_ids.cpu().numpy())
                    sequences.append(seq)
        
        # Clean sequences: remove '1' and '2' control tokens, truncate at terminals
        cleaned = []
        for seq in sequences:
            seq = self._clean_sequence(seq)
            cleaned.append(seq)
        
        return cleaned
    
    def _clean_sequence(self, seq: str) -> str:
        """Remove control tokens and truncate at terminal."""
        # Truncate at '1' or '2' (keeping only the protein part)
        for terminal in ['1', '2']:
            pos = seq.find(terminal, 1)  # Skip first char
            if pos != -1:
                seq = seq[:pos]
        # Remove all '1' and '2' characters
        seq = seq.replace('1', '').replace('2', '')
        return seq

