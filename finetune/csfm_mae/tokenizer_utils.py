"""Offline tokenizer loading and explicit provenance for CSFM adaptation.

The public CSFM implementation does not identify its exact pretrained token-ID
mapping. BERT uncased is an explicit assumption, not an author-confirmed fact.
Only tokenizer assets are bundled; no BERT model weights are used or needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parent / "assets" / "bert-base-uncased"
TEXT_LENGTH = 64
VOCAB_SIZE = 30522
SPECIAL_IDS = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
ASSET_NAMES = ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "config.json", "LICENSE")


def _resolve_dir(tokenizer_dir: str | Path | None) -> Path:
    path = DEFAULT_TOKENIZER_DIR if tokenizer_dir is None else Path(tokenizer_dir).expanduser()
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local tokenizer directory does not exist: {path}")
    for name in ("vocab.txt", "tokenizer.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Required local tokenizer asset is missing: {path / name}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_contract(path: Path) -> dict[str, Any]:
    vocab = (path / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if len(vocab) != VOCAB_SIZE or len(set(vocab)) != VOCAB_SIZE:
        raise ValueError(f"Tokenizer must have {VOCAB_SIZE} unique vocabulary entries; got {len(vocab)}")
    for token, expected in SPECIAL_IDS.items():
        if vocab[expected] != token:
            raise ValueError(f"Tokenizer special token {token} must have ID {expected}")
    files = {name: {"sha256": _sha256(path / name), "bytes": (path / name).stat().st_size}
             for name in ASSET_NAMES if (path / name).is_file()}
    default = path == DEFAULT_TOKENIZER_DIR.resolve()
    provenance_path = DEFAULT_TOKENIZER_DIR / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if default:
        for name, expected in provenance["files"].items():
            if name not in files or files[name] != expected:
                raise ValueError(f"Bundled tokenizer asset failed pinned hash/size verification: {name}")
    return {
        "status": "assumed_bert_uncased_mapping" if default else "user_supplied_mapping_unverified",
        "original_csfm_mapping_verified": False,
        "tokenizer_dir": str(path),
        "tokenizer_class": "BertTokenizerFast",
        "vocab_size": VOCAB_SIZE,
        "text_length": TEXT_LENGTH,
        "special_token_ids": SPECIAL_IDS.copy(),
        "source_repository": provenance["source_repository"] if default else "user_supplied_local_directory",
        "source_revision": provenance["source_revision"] if default else None,
        "original_mapping_ambiguity_url": provenance["original_mapping_ambiguity_url"],
        "assumption": provenance["assumption"],
        "files": files,
        "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def load_tokenizer(tokenizer_dir: str | Path | None = None):
    """Return a fully local fast WordPiece tokenizer; this never downloads files.

    Supports offset_mapping and special_tokens_mask for field/value retention
    audits. Call with max_length=64, truncation=True and padding='max_length'.
    A supplied directory overrides the explicit BERT mapping assumption, but its
    original-CSFM provenance remains unverified until separately documented.
    """
    path = _resolve_dir(tokenizer_dir)
    _asset_contract(path)
    try:
        from transformers import BertTokenizerFast
    except ImportError as exc:
        raise RuntimeError("Use the Python environment from requirements.txt, "
                           "which contains transformers and tokenizers.") from exc
    tokenizer = BertTokenizerFast.from_pretrained(str(path), local_files_only=True,
                                                 model_max_length=TEXT_LENGTH,
                                                 clean_up_tokenization_spaces=False)
    if tokenizer.vocab_size != VOCAB_SIZE or len(tokenizer) != VOCAB_SIZE:
        raise ValueError("Tokenizer vocabulary or added token count is incompatible with the CSFM text weights")
    disk_vocab = (path / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if any(tokenizer.convert_tokens_to_ids(token) != i for i, token in enumerate(disk_vocab)):
        raise ValueError("tokenizer.json token IDs do not match vocab.txt")
    for token, expected in SPECIAL_IDS.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Loaded tokenizer changed special token mapping for {token}")
    return tokenizer


def continuation_ids(tokenizer_dir: str | Path | None = None) -> frozenset[int]:
    """IDs of WordPiece pieces that continue a word, marked '##' in the vocabulary."""
    vocab = (_resolve_dir(tokenizer_dir) / "vocab.txt").read_text(encoding="utf-8").splitlines()
    return frozenset(number for number, piece in enumerate(vocab) if piece.startswith("##"))


def tokenizer_contract(tokenizer_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a JSON-serializable, hash-verified mapping provenance contract."""
    return _asset_contract(_resolve_dir(tokenizer_dir))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    report = tokenizer_contract(args.tokenizer_dir)
    report["smoke_test"] = {
        "vocab_ids_validated": True,
        "encoded_length": len(tokenizer("Heart rhythm: sinus rhythm.", max_length=64,
                                        padding="max_length", truncation=True)["input_ids"]),
    }
    print(json.dumps(report, indent=2))
