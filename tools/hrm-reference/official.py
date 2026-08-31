"""Run the OFFICIAL sapientinc/HRM ReasoningModule on the ported
checkpoint's real weights, and dump stages for the Rust side.

The checkpoint is a transformers-style port with different attribute
names; the shapes are identical, so the weights are remapped onto the
official module rather than reimplemented.
"""
import json, pathlib, struct, sys
sys.path.insert(0, "shim")           # flash_attn stand-in, CPU/MPS
sys.path.insert(0, ".")
import torch
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Config as Cfg,
    HierarchicalReasoningModel_ACTV1Block as Block,
    HierarchicalReasoningModel_ACTV1ReasoningModule as Module,
)
from models.layers import RotaryEmbedding

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cfgj = json.loads((src / "config.json").read_text())

cfg = Cfg(
    batch_size=1, seq_len=81, puzzle_emb_ndim=cfgj["puzzle_emb_ndim"],
    num_puzzle_identifiers=cfgj["num_puzzle_identifiers"], vocab_size=cfgj["vocab_size"],
    H_cycles=cfgj["h_cycles"], L_cycles=cfgj["l_cycles"],
    H_layers=cfgj["h_layers"], L_layers=cfgj["l_layers"],
    hidden_size=cfgj["hidden_size"], expansion=cfgj["expansion"],
    num_heads=cfgj["num_attention_heads"], pos_encodings=cfgj["pos_encodings"],
    halt_max_steps=cfgj["halt_max_steps"],
    halt_exploration_prob=cfgj["halt_exploration_prob"],
    forward_dtype="float32",
)
print("official config built; expansion", cfg.expansion, "inter?", )

# --- load the ported weights ---
with open(src / "model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); body = f.read()
hdr.pop("__metadata__", None)
def tensor(name):
    m = hdr[name]; lo, hi = m["data_offsets"]
    return torch.frombuffer(bytearray(body[lo:hi]), dtype=torch.float32).view(m["shape"])

low = Module(layers=[Block(cfg) for _ in range(cfg.L_layers)])
state = {}
for i in range(cfg.L_layers):
    p = f"model.inner.low_level_module.layers.{i}"
    state[f"layers.{i}.self_attn.qkv_proj.weight"]  = tensor(f"{p}.self_attn.qkv_projection.weight")
    state[f"layers.{i}.self_attn.o_proj.weight"]    = tensor(f"{p}.self_attn.output_projection.weight")
    state[f"layers.{i}.mlp.gate_up_proj.weight"]    = tensor(f"{p}.mlp.gate_up_projection.weight")
    state[f"layers.{i}.mlp.down_proj.weight"]       = tensor(f"{p}.mlp.down_projection.weight")
missing, unexpected = low.load_state_dict(state, strict=False)
print("missing:", missing, "unexpected:", unexpected)

torch.manual_seed(11)
P = 8
hidden = torch.randn(1, P, cfg.hidden_size) * 0.5
inject = torch.randn(1, P, cfg.hidden_size) * 0.5
rot = RotaryEmbedding(dim=cfg.hidden_size // cfg.num_heads,
                      max_position_embeddings=P, base=cfg.rope_theta)
with torch.no_grad():
    y = low(hidden, inject, cos_sin=rot())

def dump(name, t): (out / name).write_bytes(t.contiguous().float().numpy().tobytes())
dump("hidden.f32", hidden[0]); dump("inject.f32", inject[0]); dump("expected.f32", y[0])
lines = ["# name\tshape\tdtype\tblob\telements"]
order = []
for i in range(cfg.L_layers):
    p = f"model.inner.low_level_module.layers.{i}"
    order += [(f"{p}.self_attn.qkv_projection.weight"), (f"{p}.self_attn.output_projection.weight"),
              (f"{p}.mlp.gate_up_projection.weight"), (f"{p}.mlp.down_projection.weight")]
for i, name in enumerate(order):
    t = tensor(name); b = f"{i:03d}.bin"
    (out / b).write_bytes(t.T.contiguous().numpy().tobytes())   # column-major stream order
    lines.append(f"{name}\t{t.shape[0]},{t.shape[1]}\tFloatStorage\t{b}\t{t.numel()}")
(out / "manifest.tsv").write_text("\n".join(lines) + "\n")
print(f"positions={P} out range [{y.min():.6f}, {y.max():.6f}]")
