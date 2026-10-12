# Generate N distinct synthetic rank-R all-linear LoRA adapters for Qwen3-8B (non-zero A and B).
import json, os, sys
import torch
from safetensors.torch import save_file

N = int(sys.argv[1]); R = int(sys.argv[2]); OUT = sys.argv[3]
H, I, NH, NKV, HD, L = 4096, 12288, 32, 8, 128, 36
shapes = {  # module -> (in, out)
    "self_attn.q_proj": (H, NH * HD), "self_attn.k_proj": (H, NKV * HD),
    "self_attn.v_proj": (H, NKV * HD), "self_attn.o_proj": (NH * HD, H),
    "mlp.gate_proj": (H, I), "mlp.up_proj": (H, I), "mlp.down_proj": (I, H),
}
cfg = {"base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM",
       "r": R, "lora_alpha": 2 * R, "lora_dropout": 0.0, "bias": "none",
       "target_modules": [m.split(".")[1] for m in shapes]}
for a in range(N):
    d = os.path.join(OUT, f"lora{a:03d}")
    if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
        continue
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(1000 + a)
    t = {}
    for l in range(L):
        for m, (fi, fo) in shapes.items():
            p = f"base_model.model.model.layers.{l}.{m}"
            t[f"{p}.lora_A.weight"] = (torch.randn(R, fi, generator=g) / fi ** 0.5).to(torch.bfloat16)
            t[f"{p}.lora_B.weight"] = (torch.randn(fo, R, generator=g) * 0.01).to(torch.bfloat16)
    save_file(t, os.path.join(d, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w"))
print("done", N, R)
