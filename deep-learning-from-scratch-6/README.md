# ゼロから作る Deep Learning 6

- Book: https://www.oreilly.co.jp/books/9784814401611/
- GitHub: https://github.com/oreilly-japan/deep-learning-from-scratch-6

## codebot

スクリプトは `codebot/` ディレクトリから実行する（データファイルを `data/` からの相対パスで読み書きするため）。

### BPE トークナイザの学習

`data/tiny_codes.txt` から語彙数 1000 のマージルールを学習し、`data/tiny_codes.pkl` に保存する。

```bash
cd codebot
uv run --frozen scripts/train_bpe_tiny_codes.py
```

`--frozen` を付けると `uv.lock` を更新せずに実行する。
