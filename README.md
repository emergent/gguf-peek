# gguf-peek

A small Rust program that opens a GGUF file and counts which quantization type each tensor uses. It has no dependencies beyond `std`.

## What it does

- Reads the GGUF header, metadata, and tensor info
- Tallies tensors by type and by layer
- Compares the layers that llama.cpp's `use_more_bits` should pick with the layers that are actually promoted to a higher-precision type. It compares them both by layer number and by the order the tensors appear in the file
- Converts `Q4_0` and `Q4_K` blocks back to f32 (`src/quant.rs`). The command does not call these functions; the tests do

The command never reads the tensor data, so it also works on a file that has been cut off after the first few megabytes.

## Usage

```bash
cargo run --release -- path/to/model.gguf
```

`output/` holds the results for the files I examined.

| File | Model |
|---|---|
| `Llama-3.2-1B-Instruct-Q4_0.txt` | bartowski's `Q4_0` |
| `Llama-3.2-1B-Instruct-Q4_K_S.txt` | bartowski's `Q4_K_S` |
| `Llama-3.2-1B-Instruct-Q4_K_M.txt` | bartowski's `Q4_K_M` |
| `Llama-3.2-1B-Instruct-Q4_K_L.txt` | bartowski's `Q4_K_L` |
| `unsloth-Llama-3.2-1B-Instruct-Q4_K_M.txt` | unsloth's `Q4_K_M` (only the first 10 MB was downloaded) |
| `gemma-4-26B-A4B-it-Q4_K_M.txt` | lmstudio-community's `Q4_K_M` |

## Tests

```bash
cargo test
```

Two more tests compare the dequantized values bit for bit with [candle](https://github.com/huggingface/candle). They need GGUF files, so they do not run by default. Pass the file paths in environment variables to run them.

```bash
GGUF_Q4_0=path/to/model-Q4_0.gguf GGUF_Q4_K=path/to/model-Q4_K_M.gguf cargo test --release -- --ignored --nocapture
```

On my machine, every value matched candle (`candle-core` 0.11.1) for these three files.

| File | Type | Tensors | Values |
|---|---|---:|---:|
| Llama 3.2 1B `Q4_0` (bartowski) | `Q4_0` | 110 | 939,524,096 |
| Llama 3.2 1B `Q4_K_M` (bartowski) | `Q4_K` | 96 | 830,472,192 |
| Gemma 4 26B-A4B `Q4_K_M` (lmstudio-community) | `Q4_K` | 192 | 16,617,373,696 |

`half` and `candle-core` are used only by the tests.

## Tested with

Rust 1.100.0 (edition 2024).

## References

- [The GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- `src/llama-quant.cpp` and `ggml/src/ggml-quants.c` in [llama.cpp](https://github.com/ggml-org/llama.cpp)

The block layouts and the bit packing follow the llama.cpp (ggml) implementation. `pack_scales_mins` in the tests is a Rust transcription of the corresponding part of ggml's `quantize_row_q4_K` (MIT License, Copyright (c) 2023-2026 The ggml authors).

## License

MIT. See [LICENSE](LICENSE).
