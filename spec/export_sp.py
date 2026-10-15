# Re-export a trained soft prompt as virtual-token embeddings on Qwen3-8B, mirroring
# soft_prompt_distillation.virtual_tokens.install_virtual_prompt (Zhang Zhen's repo):
# append <|soft_prompt_NNNN|> tokens, copy the learned vectors into their input-embedding rows,
# zero their (untied) lm_head rows; generation is additionally guarded by logit_bias -100.
# Usage: python export_sp.py <base_model_dir> <soft_prompt.safetensors> <out_dir>
import json, sys, hashlib
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

base, sp_path, out = sys.argv[1:4]
# Optional: skip the first SKIP appended ids (e.g. DFlash's mask_token_id 151669) by inserting unused pad tokens.
SKIP = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tensors = load_file(sp_path)
assert len(tensors) == 1, list(tensors)
prompt = next(iter(tensors.values()))
print("soft prompt", tuple(prompt.shape), prompt.dtype)

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
pads = [f"<|sp_unused_{i:02d}|>" for i in range(SKIP)]
names = [f"<|soft_prompt_{i:04d}|>" for i in range(prompt.shape[0])]
old = len(tok) + SKIP
assert tok.add_special_tokens({"additional_special_tokens": pads + names}) == SKIP + len(names)
ids = [tok.convert_tokens_to_ids(n) for n in names]
assert ids == list(range(old, old + len(names))), "virtual tokens not contiguous"
emb, head = model.get_input_embeddings(), model.get_output_embeddings()
assert ids[-1] < emb.weight.shape[0], "would need resize"
with torch.no_grad():
    rows = torch.tensor(ids)
    emb.weight.index_copy_(0, rows, prompt.to(emb.weight.dtype))
    assert emb.weight.data_ptr() != head.weight.data_ptr()
    head.weight.index_fill_(0, rows, 0.0)
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
meta = {"schema_version": 1, "base_model": "Qwen/Qwen3-8B", "prompt_length": len(ids), "tokens": names,
        "token_ids": ids, "logit_bias": -100.0,
        "soft_prompt_sha256": hashlib.sha256(open(sp_path, "rb").read()).hexdigest()}
json.dump(meta, open(f"{out}/soft_prompt_tokens.json", "w"), indent=2)
print("exported", out, "ids", ids[0], "..", ids[-1])
