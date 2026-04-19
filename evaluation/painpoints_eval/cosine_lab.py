"""Pure pair / one-vs-N cosine on arbitrary strings.

A throwaway tool for understanding how ``text-embedding-3-small`` rates
arbitrary pairs of strings — independent of the DB, the painpoint
fixtures, and the merge logic. Useful when you're trying to reason
about why two specific phrases did or didn't merge in production but
don't want to round-trip through the snapshot.

Modes
-----

``--pair "<a>" "<b>"``
    Embed two strings and print one cosine similarity.

``--query "<q>" --against "<c1>" "<c2>" ...``
    Embed one query string and N candidates; print a sorted list of
    ``(cosine, candidate)`` pairs. Useful for "which of these
    paraphrases is closest to my reference?".

``--repl``
    Drop into an interactive prompt:

    .. code-block:: text

        > pair "API costs are a barrier" "API cost barrier"
        cosine = 0.9213

        > add memory "Need persistent local memory for continuity"
        added: memory

        > sim memory "Memory is killer feature"
        cosine = 0.7841

        > top memory                              # vs. everything cached
        ...

    Embeddings are cached per session by exact-string key, so
    repeating the same input doesn't re-bill the OpenAI API.

By design this tool talks **only** to the embedder — it does *not*
touch ``trends.db``, the painpoints table, or any pipeline state.
That's what ``mega_merge_stress.py`` is for.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from db.embeddings import OpenAIEmbedder

from ._util import cosine_sim

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached embedder
# ---------------------------------------------------------------------------

class CachedEmbedder:
    """Wraps :class:`OpenAIEmbedder` with an exact-string cache.

    Two calls with the same text share one HTTP round-trip — important
    for `--repl` sessions where you typically `sim` against the same
    reference string many times in a row.
    """

    def __init__(self, embedder: Optional[OpenAIEmbedder] = None):
        self._inner = embedder or OpenAIEmbedder()
        self._cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float]:
        if text in self._cache:
            self._hits += 1
            return self._cache[text]
        vec = self._inner.embed(text)
        self._cache[text] = vec
        self._misses += 1
        return vec

    def get_many(self, texts: list[str]) -> list[list[float]]:
        # Honour the cache for the ones we already have, batch the rest.
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self._inner.embed_batch(missing)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
            self._misses += len(missing)
        self._hits += len(texts) - len(missing)
        return [self._cache[t] for t in texts]

    def stats(self) -> str:
        total = self._hits + self._misses
        return f"cache: {self._hits} hit / {self._misses} miss ({total} lookups, {len(self._cache)} unique)"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cmd_pair(emb: CachedEmbedder, a: str, b: str) -> None:
    va, vb = emb.get_many([a, b])
    sim = cosine_sim(va, vb)
    print(f"cosine = {sim:.4f}")


def cmd_query(emb: CachedEmbedder, query: str, candidates: list[str]) -> None:
    vecs = emb.get_many([query] + candidates)
    qv = vecs[0]
    scored = sorted(
        ((cosine_sim(qv, cv), c) for cv, c in zip(vecs[1:], candidates)),
        key=lambda x: x[0],
        reverse=True,
    )
    width = max(len(c) for c in candidates)
    for sim, cand in scored:
        print(f"  {sim:+.4f}   {cand:<{width}}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

REPL_HELP = """\
commands:
  pair "<a>" "<b>"             one-shot cosine of two strings
  add <name> "<text>"          cache a string under <name>
  list                         list named strings
  sim <name> "<text>"          cosine of named string vs. another string
  sim <name1> <name2>          cosine of two named strings
  top <name>                   sorted cosines of <name> vs. every cached string
  stats                        cache hit/miss summary
  help                         show this
  quit / exit / Ctrl-D         leave
"""


def _shlex_split(line: str) -> list[str]:
    """Shell-style tokeniser tolerant of quoted strings with spaces."""
    import shlex
    try:
        return shlex.split(line, posix=True)
    except ValueError as exc:
        raise ValueError(f"unbalanced quotes: {exc}") from exc


def repl(emb: CachedEmbedder) -> None:
    print("cosine_lab REPL — type 'help' for commands, 'quit' to leave.")
    named: dict[str, str] = {}  # name -> text
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        try:
            tokens = _shlex_split(line)
        except ValueError as exc:
            print(f"parse error: {exc}")
            continue
        cmd = tokens[0].lower()

        try:
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(REPL_HELP)
            elif cmd == "stats":
                print(emb.stats())
            elif cmd == "pair":
                if len(tokens) != 3:
                    raise ValueError("pair takes 2 quoted strings")
                cmd_pair(emb, tokens[1], tokens[2])
            elif cmd == "add":
                if len(tokens) != 3:
                    raise ValueError("add takes <name> <quoted-text>")
                name, text = tokens[1], tokens[2]
                named[name] = text
                emb.get(text)  # warm cache
                print(f"added: {name}")
            elif cmd == "list":
                if not named:
                    print("(no named strings)")
                else:
                    width = max(len(n) for n in named)
                    for n, t in named.items():
                        snippet = t if len(t) <= 60 else t[:57] + "..."
                        print(f"  {n:<{width}}  {snippet}")
            elif cmd == "sim":
                if len(tokens) != 3:
                    raise ValueError("sim takes <name> <name|quoted-text>")
                left = named.get(tokens[1])
                if left is None:
                    raise ValueError(f"unknown name: {tokens[1]} — add it first")
                right = named.get(tokens[2], tokens[2])
                vl, vr = emb.get_many([left, right])
                print(f"cosine = {cosine_sim(vl, vr):.4f}")
            elif cmd == "top":
                if len(tokens) != 2:
                    raise ValueError("top takes <name>")
                name = tokens[1]
                if name not in named:
                    raise ValueError(f"unknown name: {name}")
                if len(named) < 2:
                    print("(need at least one other named string)")
                    continue
                texts = [named[name]] + [t for n, t in named.items() if n != name]
                vecs = emb.get_many(texts)
                qv, others = vecs[0], vecs[1:]
                other_names = [n for n in named if n != name]
                scored = sorted(
                    ((cosine_sim(qv, ov), n) for ov, n in zip(others, other_names)),
                    key=lambda x: x[0],
                    reverse=True,
                )
                width = max(len(n) for n in other_names)
                for sim, n in scored:
                    print(f"  {sim:+.4f}   {n:<{width}}   {named[n][:60]}")
            else:
                print(f"unknown command: {cmd!r} — type 'help'")
        except ValueError as exc:
            print(f"error: {exc}")
        except Exception as exc:  # noqa: BLE001 — REPL must not crash
            log.exception("unexpected error in REPL")
            print(f"error: {type(exc).__name__}: {exc}")

    print(emb.stats())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Pure pair / one-vs-N cosine on arbitrary strings.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair", nargs=2, metavar=("A", "B"),
                      help="One-shot cosine of two strings.")
    mode.add_argument("--query", metavar="Q",
                      help="Embed Q and rank candidates passed via --against.")
    mode.add_argument("--repl", action="store_true",
                      help="Drop into an interactive REPL with an embedding cache.")
    parser.add_argument("--against", nargs="+", metavar="C",
                        help="Candidate strings (with --query).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
        return 1

    emb = CachedEmbedder()

    if args.pair:
        cmd_pair(emb, args.pair[0], args.pair[1])
    elif args.query:
        if not args.against:
            print("--query requires --against C1 [C2 ...]", file=sys.stderr)
            return 2
        cmd_query(emb, args.query, args.against)
    elif args.repl:
        repl(emb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
