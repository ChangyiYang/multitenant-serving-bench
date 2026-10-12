# Count LoRA CPU->GPU slot loads; enabled only when LORA_LOAD_COUNTER=1.
import os
if os.environ.get("LORA_LOAD_COUNTER") == "1":
    import builtins, sys
    _orig_import = builtins.__import__
    def _hook(name, *a, **k):
        m = _orig_import(name, *a, **k)
        mp = sys.modules.get("sglang.srt.lora.mem_pool")
        if mp is not None and hasattr(mp, "LoRAMemoryPool") and not getattr(mp, "_counted", False):
            mp._counted = True
            cls = mp.LoRAMemoryPool
            orig = cls.load_lora_weight_to_buffer
            n = [0]
            def wrapped(self, *a, **k):
                n[0] += 1
                with open(os.environ.get("LORA_LOAD_LOG", "/tmp/lora_loads.txt"), "w") as f:
                    f.write(str(n[0]))
                return orig(self, *a, **k)
            cls.load_lora_weight_to_buffer = wrapped
            builtins.__import__ = _orig_import
        return m
    builtins.__import__ = _hook
