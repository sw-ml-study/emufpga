"""Run the official BDH and dump stream-order weights plus stage outputs.

Seeded random weights: pathwaycom/bdh ships architecture and training
code, not a trained checkpoint. Random init still checks every formula
exactly, which is what actually caught the TRM and HRM bugs.

THE TRANSPOSE IS NOT UNIVERSAL. scripts/extract-checkpoint transposes
every 2-D tensor because a torch Linear stores (out, in) while .spm
wants column-major (out, in) stream order. BDH stores every parameter
as (in, out) instead, so the raw row-major bytes ARE stream order and
the declared shape is simply reversed. Applying the generic transpose
here would reintroduce postmortem defect 8 in mirror image.
"""

import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from bdh import BDH, BDHConfig  # noqa: E402


def dump(path, tensor):
    values = tensor.detach().to(torch.float32).contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    return len(values)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    out.mkdir(parents=True, exist_ok=True)
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    torch.manual_seed(20260831)
    config = BDHConfig()
    model = BDH(config).eval()  # eval() matters: drop is p=0.1

    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    idx = torch.randint(0, config.vocab_size, (1, seq), generator=torch.Generator().manual_seed(7))

    stages = {}
    with torch.no_grad():
        logits, _ = model(idx)
        # Re-run the loop by hand to capture intermediates for bisection.
        x = model.ln(model.embed(idx).unsqueeze(1))
        stages["x_embedded"] = x.clone()
        for level in range(config.n_layer):
            x_sparse = torch.relu(x @ model.encoder)
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x))
            y_sparse = torch.relu(yKV @ model.encoder_v)
            xy = x_sparse * y_sparse
            yMLP = xy.transpose(1, 2).reshape(1, 1, seq, N * nh) @ model.decoder
            x = model.ln(x + model.ln(yMLP))
            if level == 0:
                stages["x_sparse_0"] = x_sparse.clone()
                stages["yKV_0"] = yKV.clone()
                stages["y_sparse_0"] = y_sparse.clone()
                stages["yMLP_0"] = yMLP.clone()
            stages[f"x_after_{level}"] = x.clone()
        stages["logits"] = logits

    # Weights, in CONSUMPTION order: the loop reads encoder, then
    # encoder_v, then decoder, n_layer times; lm_head exactly once
    # after the last level, so it sits after the rotating region.
    rows = []
    for h in range(nh):
        n = dump(out / f"encoder_{h}.f32", model.encoder[h])
        rows.append((f"encoder.{h}", [N, D], f"encoder_{h}.f32", n))
    for h in range(nh):
        n = dump(out / f"encoder_v_{h}.f32", model.encoder_v[h])
        rows.append((f"encoder_v.{h}", [N, D], f"encoder_v_{h}.f32", n))
    n = dump(out / "decoder.f32", model.decoder)
    rows.append(("decoder", [D, nh * N], "decoder.f32", n))
    n = dump(out / "lm_head.f32", model.lm_head)
    rows.append(("lm_head", [config.vocab_size, D], "lm_head.f32", n))

    # Resident by design: an embedding is a gather by token id, not a
    # sweep. Streaming it to serve one token would read the whole table.
    dump(out / "embed.f32", model.embed.weight)

    manifest = ["# name\tshape\tdtype\tblob\tcount"]
    for name, shape, blob, count in rows:
        manifest.append(f"{name}\t{','.join(map(str, shape))}\tf32\t{blob}\t{count}")
    (out / "manifest.tsv").write_text("\n".join(manifest) + "\n")

    for name, tensor in stages.items():
        dump(out / f"stage_{name}.f32", tensor)
    ids = idx.view(-1).tolist()
    (out / "tokens.u32").write_bytes(struct.pack(f"<{len(ids)}I", *ids))

    (out / "meta.json").write_text(json.dumps({
        "n_layer": config.n_layer, "n_embd": D, "n_head": nh, "N": N,
        "vocab_size": config.vocab_size, "seq": seq,
        "rotating_streams": 2 * nh + 1,
        "stages": {k: list(v.shape) for k, v in stages.items()},
    }, indent=2))
    print(f"N={N} seq={seq} streams={len(rows)} rotating={2 * nh + 1}")
    print("logits", tuple(logits.shape), "first", logits.view(-1)[:4].tolist())


if __name__ == "__main__":
    main()
