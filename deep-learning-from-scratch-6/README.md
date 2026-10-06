# ゼロから作る Deep Learning 6

- Book: https://www.oreilly.co.jp/books/9784814401611/
- GitHub: https://github.com/oreilly-japan/deep-learning-from-scratch-6

## codebot

スクリプトは `codebot/` ディレクトリから実行する（データファイルを `data/` からの相対パスで読み書きするため）。

### BPE トークナイザの学習

`data/tiny_codes.txt` から語彙数 1000 のマージルールを学習し、`data/tiny_codes.pkl` に保存する。
`data/tiny_codes.pkl` が既にある場合は学習をスキップしてそれを読み込む。
あわせて、最初と最後に学習された 10 トークンと、先頭 10,000 文字での圧縮比を表示する。

```bash
cd codebot
uv run --frozen scripts/train_bpe_tiny_codes.py
```

`--frozen` を付けると `uv.lock` を更新せずに実行する。

### データセットのトークン化

学習済みのマージルール `data/tiny_codes.pkl` で `data/tiny_codes.txt` 全体をエンコードし、トークン ID 列を `uint16` のバイナリとして `data/tiny_codes.bin` に保存する。

```bash
cd codebot
uv run --frozen scripts/bpe_encode_tiny_codes.py
```
