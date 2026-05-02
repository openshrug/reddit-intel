"""
Painpoint extraction.

Main flow:
1. Load posts and comments from SQLite, or accept caller-provided
   posts-with-comments for the pure in-memory API.
2. Build token-budgeted batches and render each batch with stable post /
   comment IDs for the LLM.
3. Ask the LLM to emit structured pending painpoints against the category
   taxonomy.
4. Postprocess each row's quote evidence via `painpoint_extraction.postprocess`
   so only source-faithful `quoted_text` values continue.
5. Persist verified rows as pending_painpoints, using embedding dedup to link
   near-duplicate observations as extra sources.

    await extract_painpoints([1, 2, 3])
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from db import get_db
from db.categories import get_category_list_flat
from db.embeddings import OpenAIEmbedder
from db.painpoints import save_pending_painpoints_batch
from db.posts import get_comments_for_post, get_posts_by_ids
from llm import OPENAI_COMPLETION_CONCURRENCY, TokenCounter, get_client, llm_call
from painpoint_extraction.postprocess import postprocess_painpoints

log = logging.getLogger(__name__)

# --- Tunables ---
MODEL = "gpt-5-nano"
REASONING_EFFORT = "low"
BATCH_TOKEN_BUDGET = 2_000
# OpenAI request concurrency is enforced in llm.py. This semaphore only
# prevents extractor batches from flooding asyncio.to_thread workers while
# they wait on llm.py's shared completion semaphore.
EXTRACTION_BATCH_CONCURRENCY = OPENAI_COMPLETION_CONCURRENCY


# --- Structured output schema ---

class ExtractedPainpoint(BaseModel):
    title: str = Field(description="Concise name revealing the essence of the pain")
    description: str = Field(description="1-2 sentence explanation of the pain")
    severity: int = Field(ge=1, le=10, description="1 = trivial inconvenience, 3 = recurring annoyance, 5 = significant friction affecting routine, 7 = major disruption or emotional distress, 10 = totally blocking")
    quoted_text: str = Field(description="Short phrase or clause copied verbatim from one source post or comment; keep it under one sentence")
    category_name: str = Field(description="Must match a category from the taxonomy, or 'Uncategorized'")
    post_id: int = Field(description="The [Post N] ID from the input")
    comment_id: int | None = Field(default=None, description="The [Comment N] ID, or null if from the post body")


class ExtractionResult(BaseModel):
    painpoints: list[ExtractedPainpoint]


# --- Instructions ---

EXTRACT_INSTRUCTIONS = """\
You are a painpoint extraction engine for product research. You will \
receive Reddit posts with their comments. Your job is to identify user \
painpoints that someone could build a product, app, or service around.

Keep painpoints that meet BOTH of these:
- APP-ADDRESSABLE: a mobile app, web app, browser extension, API, or \
small service could plausibly solve or ease this pain. The solution \
doesn't have to be technical itself — a content app, matching app, \
tracker, social tool, or consumer utility all count.
- SPECIFIC: a concrete friction, missing feature, broken experience, or \
unmet need — not a vague opinion or abstract commentary.

Bias toward KEEPING painpoints with viral / mass-market potential: \
consumer frustrations, social/dating/relationship friction that a product \
could ease, lifestyle and habit pains, creator/marketing workflow \
struggles, etc. Non-technical pains are welcome if a product could \
address them.

Skip painpoints that are:
- Pure opinions, memes, jokes, or sarcasm with no real pain behind them.
- Philosophical or political takes with no product hook ("AI will take our \
jobs", "society is broken").
- Company drama / platform politics with no product angle.

Rules:
- A single post/comment may contain zero, one, or many painpoints.
- Different comments on the same post may yield different painpoints.
- quoted_text must be copied verbatim from one source post or comment. \
Prefer a short phrase; a short clause is okay when needed, but keep it \
under one sentence. Do not paraphrase, stitch together separate phrases, \
or change numbers.
- If no category fits well, use "Uncategorized".

## Category taxonomy

{taxonomy}"""


# ============================================================
# Public API
# ============================================================

async def extract_painpoints_from_posts(
    posts_with_comments,
    categories,
    *,
    batch_token_budget=BATCH_TOKEN_BUDGET,
):
    """Pure in-memory painpoint extraction — no DB access.

    Dict-in, dict-out. Use this when the caller owns its own storage
    (e.g. the closed backend writes results straight to Postgres rather
    than going through engine's SQLite).

    Args:
        posts_with_comments: list of post dicts. Each post must have the
            same fields as ``reddit_scraper.scrape_subreddit_full``
            output (``id``, ``title``, ``selftext``, ``subreddit``,
            ``score``, ``num_comments``) plus a ``comments`` list of
            comment dicts with ``id``, ``body``, ``score``.

            The ``id`` on posts and comments is referenced by the LLM in
            its output (``post_id`` / ``comment_id`` fields on returned
            pendings). Callers that don't already have persisted IDs can
            assign any stable integers here and map back after.
        categories: list of category dicts with ``path`` (or ``name``)
            and ``description`` keys, used to build the taxonomy section
            of the LLM prompt. Pass ``[]`` to extract with a
            taxonomy-free prompt; the LLM will label everything
            ``Uncategorized`` and the caller can re-assign categories
            later via embedding similarity.
        batch_token_budget: target token count per LLM batch.

    Returns:
        Tuple of (list of pending-painpoint dicts, token usage dict).
        Each pending dict has: ``title``, ``description``, ``severity``,
        ``quoted_text``, ``category_name``, ``post_id``,
        ``comment_id?``. No DB side effects.
    """
    if not posts_with_comments:
        return [], TokenCounter().as_dict()

    # Internal helpers take (post, comments) tuples; callers pass
    # posts-with-nested-comments. One-time unpack here.
    tuples = [(p, p.get("comments", []) or []) for p in posts_with_comments]

    batches = _build_batches(tuples, batch_token_budget)
    log.info("extract: %d posts -> %d batches (budget %dK tokens)",
             len(posts_with_comments), len(batches), batch_token_budget // 1000)

    instructions = _build_instructions_from_categories(categories)
    client = get_client()
    sem = asyncio.Semaphore(EXTRACTION_BATCH_CONCURRENCY)
    counter = TokenCounter()

    batch_results = await asyncio.gather(
        *(_process_batch(i, len(batches), b, client, instructions, sem, counter)
          for i, b in enumerate(batches))
    )

    items = [item for batch_items in batch_results for item in batch_items]

    usage = counter.as_dict()
    log.info(
        "extract: tokens — input: %d (cached: %d, %.1f%% hit), "
        "output: %d (reasoning: %d, text: %d)",
        usage["input_tokens"], usage["cached_input_tokens"],
        usage["cache_hit_pct"], usage["output_tokens"],
        usage["reasoning_tokens"], usage["text_tokens"],
    )

    return items, usage


async def extract_painpoints(post_ids, *, batch_token_budget=BATCH_TOKEN_BUDGET):
    """Extract painpoints from persisted posts via LLM.

    SQLite-backed entry point — reads posts from engine's DB, calls the
    pure extractor, writes pendings back. Used by the solo-agent
    pipeline and existing tests. Downstream consumers that own their
    own storage should call ``extract_painpoints_from_posts`` directly
    instead.

    Args:
        post_ids: list of internal post IDs (from posts.id).
        batch_token_budget: target token count per LLM batch.

    Returns:
        Tuple of (list of pending_painpoints IDs created, token usage dict).
    """
    if not post_ids:
        return [], TokenCounter().as_dict()

    post_ids = _filter_unextracted(post_ids)
    if not post_ids:
        log.info("extract: all posts already extracted, skipping")
        return [], TokenCounter().as_dict()

    # Fetch from engine SQLite, reshape into posts-with-nested-comments
    # so the pure extractor can consume it.
    posts_with_comments = _load_posts_with_comments(post_ids)
    post_dicts = [{**post, "comments": comments}
                  for post, comments in posts_with_comments]
    categories = get_category_list_flat()

    items, usage = await extract_painpoints_from_posts(
        post_dicts, categories, batch_token_budget=batch_token_budget,
    )

    if not items:
        log.info("extract: no painpoints found")
        return [], usage

    # Pass an embedder so near-duplicate observations across batches and
    # within this batch collapse into one pending row (with extra sources
    # linked via pending_painpoint_sources) instead of piling up as
    # near-copies the frontend would render as duplicates.
    ids = save_pending_painpoints_batch(items, embedder=OpenAIEmbedder())
    unique_ids = len(set(ids))
    log.info(
        "extract: saved %d pending painpoints (%d unique; %d deduped)",
        len(ids), unique_ids, len(ids) - unique_ids,
    )
    return ids, usage


# ============================================================
# Per-batch processing
# ============================================================

async def _process_batch(batch_idx, total, batch, client, instructions, sem, counter):
    """LLM extract -> validate source evidence -> return corrected rows."""
    async with sem:
        batch_text = _format_batch(batch)
        log.info("extract: batch %d/%d (%d posts)",
                 batch_idx + 1, total, len(batch))
        try:
            result = await asyncio.to_thread(
                llm_call, client, instructions, batch_text,
                response_model=ExtractionResult, model=MODEL,
                max_tokens=None, reasoning_effort=REASONING_EFFORT,
                token_counter=counter,
            )
        except Exception as exc:
            log.warning("extract: batch %d/%d failed: %s",
                        batch_idx + 1, total, exc)
            return []

        items = [pp.model_dump() for pp in result.painpoints]
        items, stats = postprocess_painpoints(items, batch)
        log.info(
            "extract: batch %d/%d -> %d painpoints kept "
            "(%d attribution fixes, %d fuzzy quote repairs, "
            "%d numeric/entity repairs, %d quote drops)",
            batch_idx + 1, total, stats["kept"], stats["attribution_fixed"],
            stats["fuzzy_repaired"], stats["numeric_entity_repaired"],
            stats["dropped"],
        )
        return items


# ============================================================
# Private helpers
# ============================================================

def _filter_unextracted(post_ids):
    """Return only post_ids that have no pending_painpoints rows yet."""
    if not post_ids:
        return []
    conn = get_db()
    placeholders = ",".join("?" * len(post_ids))
    existing = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT post_id FROM pending_painpoints"
            f" WHERE post_id IN ({placeholders})",
            list(post_ids),
        ).fetchall()
    }
    conn.close()
    filtered = [pid for pid in post_ids if pid not in existing]
    skipped = len(post_ids) - len(filtered)
    if skipped:
        log.info("extract: skipping %d already-extracted posts", skipped)
    return filtered


def _load_posts_with_comments(post_ids):
    """Fetch posts and their comments from the DB.

    Returns a list of (post_dict, comments_list) tuples, ordered by
    engagement descending (inherited from get_posts_by_ids ordering).
    """
    posts_map = get_posts_by_ids(post_ids)
    result = []
    for pid in posts_map:
        post = posts_map[pid]
        comments = get_comments_for_post(pid)
        result.append((post, comments))
    return result


def _build_batches(posts_with_comments, budget):
    """Greedy bin-packing: add posts to the current batch until the next
    one would exceed the token budget, then start a new batch."""
    batches = []
    current_batch = []
    current_tokens = 0

    for post, comments in posts_with_comments:
        tokens = _estimate_tokens(post, comments)
        if current_batch and current_tokens + tokens > budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append((post, comments))
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def _build_instructions():
    """Legacy: build instructions with taxonomy read from engine SQLite."""
    return _build_instructions_from_categories(get_category_list_flat())


def _build_instructions_from_categories(categories):
    """Pure variant — caller supplies the category list.

    Accepts either ``{path, description}`` (as returned by engine's
    ``get_category_list_flat``) or ``{name, description}`` (convenient
    for callers that don't build a hierarchical path). Missing keys
    fall back gracefully so both shapes work.
    """
    if categories:
        taxonomy_lines = [
            f"- {c.get('path') or c.get('name', '?')}: "
            f"{c.get('description') or '(no description)'}"
            for c in categories
        ]
        taxonomy = "\n".join(taxonomy_lines)
    else:
        taxonomy = "- Uncategorized"
    return EXTRACT_INSTRUCTIONS.format(taxonomy=taxonomy)


def _estimate_tokens(post, comments):
    """Rough token estimate: ~4 chars per token for English text."""
    text_len = len(post.get("title", "")) + len(post.get("selftext", "") or "")
    text_len += sum(len(c.get("body", "") or "") for c in comments)
    return max(1, text_len // 4)


def _format_batch(batch):
    """Render a batch of (post, comments) tuples into LLM-ready text."""
    return "\n\n".join(_format_post(post, comments) for post, comments in batch)


def _format_post(post, comments):
    """Render one post + its comments with DB IDs the LLM can reference."""
    lines = [
        f"### [Post {post['id']}] \"{post['title']}\""
        f" (r/{post['subreddit']}, score: {post['score']},"
        f" {post['num_comments']} comments)"
    ]
    if post.get("selftext"):
        lines.append(post["selftext"])
    if comments:
        lines.append("\nComments:")
        for c in comments:
            lines.append(
                f"  [Comment {c['id']}] (score: {c['score']}) {c['body']}"
            )
    return "\n".join(lines)
