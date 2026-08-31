"""Run SmolLM2-135M and dump stage outputs for a stage-by-stage check.

Weights are NOT dumped here: unlike BDH, SmolLM's tensors are ordinary
torch Linear weights stored (out, in), so scripts/extract-checkpoint's
transpose is the correct one and the normal import path applies.

Everything runs in f32. The checkpoint is bf16; widening at load makes
the reference comparable to the engine, which is f32 throughout.
"""

import json
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def dump(path, tensor):
    values = tensor.detach().to(torch.float32).contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    cfg = model.config
    vocab = json.loads((src / "config.json").read_text())["vocab_size"]
    ids = torch.arange(1, seq + 1, dtype=torch.long).view(1, seq) * 137 % vocab

    stages = {}
    with torch.no_grad():
        result = model(ids, output_hidden_states=True)
        for index, hidden in enumerate(result.hidden_states):
            stages[f"hidden_{index}"] = hidden
        stages["logits"] = result.logits

    for name, tensor in stages.items():
        dump(out / f"stage_{name}.f32", tensor)
    (out / "tokens.u32").write_bytes(
        struct.pack(f"<{seq}I", *ids.view(-1).tolist())
    )
    # Read the shape from config.json rather than the live config
    # object: transformers 5 relocated rope_theta, and the published
    # file is the authority the engine is written against anyway.
    published = json.loads((src / "config.json").read_text())
    published.update({"seq": seq, "hidden_states": len(result.hidden_states)})
    (out / "meta.json").write_text(json.dumps(published, indent=2))
    print("layers", published["num_hidden_layers"],
          "hidden states", len(result.hidden_states))
    print("logits", tuple(result.logits.shape))
    print("first logits", result.logits.view(-1)[:4].tolist())


if __name__ == "__main__":
    main()
