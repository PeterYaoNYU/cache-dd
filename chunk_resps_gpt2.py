#!/usr/bin/env python3
"""
chunk_resps_gpt2_v2.py

Read a lm-evaluation-harness samples_*.jsonl file, GPT-2-tokenise every
completion in `resps`, split into N-token blocks (default 128), decode each
block back to text, and write one JSON line *per completion* like:

{
  "doc_id": 70,
  "question": "...",                # original prompt question
  "resp_set": 0,
  "resp_idx": 0,
  "blocks": ["first 128-token slice", "second slice", ...],
  "token_blocks": [[50256, 470, ...], ...]   # optional
}

Usage
-----
    python chunk_resps_gpt2_v2.py samples.jsonl
    python chunk_resps_gpt2_v2.py samples.jsonl -n 64 -o out.jsonl --no-tokens
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_CHUNK = 128
DEFAULT_TOK = "gpt2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk `resps` with GPT-2 tokenizer.")
    p.add_argument("jsonl", type=Path, help="lm-eval samples_*.jsonl file")
    p.add_argument("-o", "--out", type=Path, help="output path (.blocks.jsonl)")
    p.add_argument(
        "-n", "--chunk-size", type=int, default=DEFAULT_CHUNK, help="tokens per block"
    )
    p.add_argument(
        "-t", "--tokenizer", default=DEFAULT_TOK, help="HF tokenizer name (default gpt2)"
    )
    p.add_argument(
        "--no-tokens",
        action="store_true",
        help="omit raw token-ID blocks to keep the file small",
    )
    return p.parse_args()


def split_chunks(seq: list[int], size: int):
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main() -> None:
    args = parse_args()
    infile: Path = args.jsonl
    outfile: Path = args.out or infile.with_suffix(".blocks.jsonl")

    if not infile.exists():
        sys.exit(f"Input file not found: {infile}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True)

    with infile.open(encoding="utf-8") as fin, outfile.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            sample = json.loads(raw)
            doc_id = sample.get("doc_id")
            question = sample.get("doc", {}).get("question", "<no-question>")
            answer = sample.get("doc", {}).get("answer", "<no-answer>")
            filtered_resp = sample.get("filtered_resps", "<no-filtered>")

            for set_idx, resp_set in enumerate(sample.get("resps", [])):
                for resp_idx, resp_text in enumerate(resp_set):
                    ids = tok.encode(resp_text, add_special_tokens=False)
                    token_blocks = split_chunks(ids, args.chunk_size)
                    text_blocks = [
                        [tok.decode(block, clean_up_tokenization_spaces=False)]
                        for block in token_blocks
                    ]
                    
                    blocks_len = [len(block) for block in token_blocks]

                    out_line = {
                        "doc_id": doc_id,
                        "question": question,
                        "resp_set": set_idx,
                        "resp_idx": resp_idx,
                        "blocks": text_blocks,
                        "answer": answer, 
                        "filtered_resp": filtered_resp,
                        "blocks_len": blocks_len,
                    }
                    if not args.no_tokens:
                        out_line["token_blocks"] = token_blocks

                    fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    print(f"Wrote block-structured chunks → {outfile}")


if __name__ == "__main__":
    main()
