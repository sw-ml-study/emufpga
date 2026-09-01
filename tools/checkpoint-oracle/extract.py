#!/usr/bin/env python3
"""Extract tensors from a PyTorch .pt into raw blobs plus a manifest.

Standard library only. **torch is not required and not imported** --
a .pt is a ZIP holding a Python pickle and raw storage blobs, and both
are readable without it.

Why this is Python and the rest of the importer is Rust: reading a .pt
in Rust means writing a ZIP reader and a pickle VM, and the pickle
subset torch emits runs to roughly 25 opcodes. That is a lot of code
whose reuse is doubtful -- the DeepSeek R1 1.58-bit quant this project
builds toward is GGUF, not pickle, so a .pt VM would be written once
and retired. The split puts the throwaway half in the language that
already has a pickle implementation.

    scripts/extract-checkpoint <model.pt|model.safetensors> <out-dir>

Both container formats are read. A .pt is a ZIP holding a pickle; a
.safetensors is a JSON header giving names, shapes, dtypes and byte
offsets, followed by the raw data -- no pickle, and far less to go
wrong. The output contract is the same either way.

Writes <out-dir>/manifest.tsv and one <out-dir>/<index>.bin per
tensor, little-endian f32, in .spm STREAM ORDER.

Two conversions happen here, and both are this script's job because
this is the boundary between a framework's conventions and the
format's:

  * bf16 storages are widened to f32. Widening bf16 to f32 is exact.
  * 2-D tensors are TRANSPOSED. PyTorch stores row-major, so W[r][c]
    sits at r*cols + c; .spm stream order is column-major, so index k
    must hold W[k % rows][k / rows]. Without this every matrix in the
    file is its own transpose -- and the bytes still round-trip
    perfectly, because round-tripping bytes says nothing about how
    they are interpreted. It took a numerical comparison against the
    reference implementation to see it.

Doing it here keeps the Rust importer pure framing: it never touches a
value, only frames bytes that already mean what the format says.
"""

import array
import json
import struct
import sys
import zipfile
from pathlib import Path

# Storage type -> (bytes per element, decoder to f32 little-endian).
# Only what TRM's checkpoint actually contains; anything else is
# rejected loudly rather than guessed at.
WIDTHS = {"FloatStorage": 4, "BFloat16Storage": 2, "HalfStorage": 2}

# safetensors names its dtypes differently; map onto the same set so
# the widening and transposing paths below do not have to care which
# container the bytes arrived in.
SAFETENSORS_DTYPES = {"F32": "FloatStorage", "BF16": "BFloat16Storage", "F16": "HalfStorage"}


def read_safetensors(path):
    """Return {name: (dtype, shape, raw bytes)} from a .safetensors."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        body = f.read()
    header.pop("__metadata__", None)
    out = {}
    for name, meta in header.items():
        lo, hi = meta["data_offsets"]
        out[name] = (meta["dtype"], meta["shape"], body[lo:hi])
    return out


def read_index(path):
    """Return (zip, root, {name: (storage_type, key, shape)})."""
    import io
    import pickle

    def rebuild(storage, offset, size, stride, *rest):
        return ("T", storage, tuple(size))

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "OrderedDict":
                import collections

                return collections.OrderedDict
            if "Storage" in name:
                return name
            return lambda *a, **k: None

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            return (stype if isinstance(stype, str) else str(stype), key)

    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    raw = Reader(io.BytesIO(z.read(f"{root}/data.pkl"))).load()
    out = {}
    for name, value in raw.items():
        if not (isinstance(value, tuple) and value and value[0] == "T"):
            continue
        (_, (stype, key), shape) = value
        out[name] = (stype, key, shape)
    return z, root, out


def to_stream_order(raw, shape):
    """Reorder row-major bytes into column-major stream order.

    `raw` is little-endian f32. A 1-D tensor is already in order: a
    single column streams in its natural sequence.
    """
    if len(shape) < 2:
        return raw
    rows = 1
    for d in shape[:-1]:
        rows *= d
    cols = shape[-1]
    src = array.array("f")
    src.frombytes(raw)
    dst = array.array("f", [0.0]) * (rows * cols)
    # src[c::cols] is column c, strided in C rather than in Python.
    for c in range(cols):
        dst[c * rows : (c + 1) * rows] = src[c::cols]
    return dst.tobytes()


def to_f32(raw, stype):
    """Widen a storage's bytes to little-endian f32 bytes."""
    if stype == "FloatStorage":
        return raw
    if stype in ("BFloat16Storage", "HalfStorage"):
        out = bytearray()
        for i in range(0, len(raw), 2):
            bits = struct.unpack_from("<H", raw, i)[0]
            if stype == "BFloat16Storage":
                # bf16 is the top 16 bits of an f32: widening is exact.
                out += struct.pack("<I", bits << 16)
            else:
                out += struct.pack("<f", struct.unpack_from("<e", raw, i)[0])
        return bytes(out)
    raise SystemExit(f"unsupported storage type {stype}")


def to_bf16(raw):
    """Round little-endian f32 bytes to little-endian bf16.

    Round-to-nearest-EVEN, matching spm-codec-bf16. Truncating -- just
    keeping the high two bytes -- is the obvious implementation and is
    biased toward zero, so the two sides of the pipeline would disagree
    on values exactly halfway between two bf16s.
    """
    values = array.array("I")
    values.frombytes(raw)
    out = array.array("H", bytes(2 * len(values)))
    for i, bits in enumerate(values):
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        if exponent == 0xFF and mantissa:  # NaN stays NaN
            out[i] = (bits >> 16) | 0x0040
            continue
        lsb = (bits >> 16) & 1
        out[i] = ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def main():
    args = [a for a in sys.argv[1:] if a != "--bf16"]
    as_bf16 = "--bf16" in sys.argv[1:]
    if len(args) != 2:
        raise SystemExit(
            "usage: extract-checkpoint [--bf16] <model.pt|.safetensors> <out-dir>"
        )
    src, dst = Path(args[0]), Path(args[1])
    dst.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".safetensors":
        index = {
            name: (SAFETENSORS_DTYPES.get(dtype, dtype), shape, raw)
            for name, (dtype, shape, raw) in read_safetensors(src).items()
        }
    else:
        z, root, raw_index = read_index(src)
        index = {
            name: (stype, shape, z.read(f"{root}/data/{key}"))
            for name, (stype, key, shape) in raw_index.items()
        }

    lines = ["# name\tshape\tdtype\tblob\telements"]
    total = 0
    for i, (name, (stype, shape, raw)) in enumerate(sorted(index.items())):
        if stype not in WIDTHS:
            raise SystemExit(f"{name}: unsupported storage {stype}")
        blob = f"{i:03d}.bin"
        # Widen, reorder, then narrow. The transpose works on f32
        # regardless, and bf16 -> f32 -> bf16 is exact, so going
        # through f32 costs nothing and keeps one reorder path.
        ordered = to_stream_order(to_f32(raw, stype), shape)
        (dst / blob).write_bytes(to_bf16(ordered) if as_bf16 else ordered)
        elements = 1
        for d in shape:
            elements *= d
        total += elements
        lines.append(
            "\t".join(
                [
                    name,
                    ",".join(str(d) for d in shape),
                    "bf16" if as_bf16 else "f32",
                    blob,
                    str(elements),
                ]
            )
        )
    (dst / "manifest.tsv").write_text("\n".join(lines) + "\n")
    width = 2 if as_bf16 else 4
    print(f"{len(index)} tensors, {total} parameters, "
          f"{total * width} bytes ({'bf16' if as_bf16 else 'f32'}) -> {dst}")


if __name__ == "__main__":
    main()
