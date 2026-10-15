"""Download every asset of the speculative-decoding study at the exact revisions used,
verify checksums, and write spec/assets.env with local paths for run_spec.sh."""
import hashlib, os
from huggingface_hub import hf_hub_download, snapshot_download

MODELS = {
    "BASE": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "EAGLE_T": ("Tengyunw/qwen3_8b_eagle3", "2a1059d51f622b8cad7d7d72840153ffea5488a0"),
    "EAGLE_A": ("AngelSlim/Qwen3-8B_eagle3", "9629dfce7a4a10564dd48d3e5485c3976095653c"),
    "DFLASH": ("z-lab/Qwen3-8B-DFlash-b16", "9b41424b7109f9c5413454f481b09a82b85333f4"),
}
PAIRS = "namezz/soft-prompt-interpretability-six-pairs"
SP = (PAIRS, "7cdc757f7414080153daf65b8bf0175bf6e30d84",
      "code-areal8/train/checkpoints/update_00000100.safetensors",
      "0e985b1f5e537f8468a7b047ddaa93705c7417ab8ec4b6e08562e1c242ff592f")
LORA = (PAIRS, "ed60034a179ab1356d612bffdd9fb4cec258f5a8",
        "code-areal8/lora/checkpoints/update-000164",
        "a74f13156f0405e1f429b0cfbf070d870f0e1ce48876fc142a2a491196442a9a")
LCB = ("inclusionAI/AReaL-boba-2-RL-Code", "1abdc2c4884eb6c4411c0d5c5e42da5e2be817de",
       "code_benchmark/lcb_v5_2410_2502/test.jsonl")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


env = {}
for key, (repo, rev) in MODELS.items():
    env[key] = snapshot_download(repo, revision=rev)
sp = hf_hub_download(SP[0], SP[2], revision=SP[1])
assert sha256(sp) == SP[3], "soft prompt checksum mismatch"
env["SP_CKPT"] = sp
lora_root = snapshot_download(LORA[0], revision=LORA[1], allow_patterns=f"{LORA[2]}/*")
env["LORA_DIR"] = os.path.join(lora_root, LORA[2])
assert sha256(os.path.join(env["LORA_DIR"], "adapter_model.safetensors")) == LORA[3], "LoRA checksum mismatch"
env["LCB_PATH"] = hf_hub_download(LCB[0], LCB[2], revision=LCB[1], repo_type="dataset")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets.env")
with open(out, "w") as f:
    for k, v in env.items():
        f.write(f"{k}={v}\n")
print(open(out).read())
