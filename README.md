# gguf-peek

GGUFファイルを開けて、量子化の「配合」を数える小さなRustプログラムです。依存クレートなしで、`std` だけで書いています。

技術同人誌「機械学習の炊いたん12」の章「Q4_K_M の "K_M" って何なんですか — GGUFをRustで開けてみる」のために書きました。

## できること

- GGUFのヘッダ、メタデータ、テンソル情報を読む
- テンソルを型ごと・層ごとに集計する
- llama.cppの `use_more_bits` が選ぶはずの層と、実際に格上げされている層を比べる。層番号の順と、ファイル中に現れる順の両方で比べます
- `Q4_0` と `Q4_K` のブロックをf32に戻す（`src/quant.rs`）。この関数はコマンドからは呼んでおらず、テストで動かします

コマンドはテンソル本体を読まないので、ファイルの先頭だけを切り出したものでも動きます。

## 使い方

```bash
cargo run --release -- path/to/model.gguf
```

`output/` に、本で使ったファイルの実行結果を置いています。

| ファイル | 対象 |
|---|---|
| `Llama-3.2-1B-Instruct-Q4_0.txt` | bartowski版 `Q4_0` |
| `Llama-3.2-1B-Instruct-Q4_K_S.txt` | bartowski版 `Q4_K_S` |
| `Llama-3.2-1B-Instruct-Q4_K_M.txt` | bartowski版 `Q4_K_M` |
| `Llama-3.2-1B-Instruct-Q4_K_L.txt` | bartowski版 `Q4_K_L` |
| `unsloth-Llama-3.2-1B-Instruct-Q4_K_M.txt` | unsloth版 `Q4_K_M`（先頭10MBだけを取得したもの） |
| `gemma-4-26B-A4B-it-Q4_K_M.txt` | lmstudio-community版 `Q4_K_M` |

## テスト

```bash
cargo test
```

逆量子化の結果を [candle](https://github.com/huggingface/candle) とビット単位で比べるテストは、GGUFファイルが要るので既定では実行されません。ファイルのパスを環境変数で渡して実行します。

```bash
GGUF_Q4_0=path/to/model-Q4_0.gguf GGUF_Q4_K=path/to/model-Q4_K_M.gguf cargo test --release -- --ignored --nocapture
```

手元では次の3つのファイルで、すべての値がcandle（`candle-core` 0.11.1）と一致しました。

| ファイル | 型 | テンソル数 | 値の数 |
|---|---|---:|---:|
| Llama 3.2 1B `Q4_0`（bartowski版） | `Q4_0` | 110 | 939,524,096 |
| Llama 3.2 1B `Q4_K_M`（bartowski版） | `Q4_K` | 96 | 830,472,192 |
| Gemma 4 26B-A4B `Q4_K_M`（lmstudio-community版） | `Q4_K` | 192 | 16,617,373,696 |

`half` と `candle-core` はテストでだけ使います。

## 動作を確かめた環境

Rust 1.100.0（edition 2024）。

## 参考にしたもの

- [GGUFの仕様](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) の `src/llama-quant.cpp` と `ggml/src/ggml-quants.c`

ブロックの並びとビットの詰め方は、llama.cpp（ggml）の実装に従っています。テストの中の `pack_scales_mins` は、ggmlの `quantize_row_q4_K` の該当部分をRustに書き写したものです（MIT License, Copyright (c) 2023-2026 The ggml authors）。

## ライセンス

MIT License です。[LICENSE](LICENSE) を見てください。
