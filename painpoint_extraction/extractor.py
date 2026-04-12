"""
Painpoint extraction — batch posts from the DB, send to an LLM with the
category taxonomy, and save extracted painpoints as pending_painpoints.

    await extract_painpoints([1, 2, 3])
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from db import get_db
from db.categories import get_category_list_flat
from db.painpoints import save_pending_painpoints_batch
from db.posts import get_posts_by_ids, get_comments_for_post
from llm import get_client, llm_call, TokenCounter

log = logging.getLogger(__name__)

# --- Tunables ---
MODEL = "gpt-5-nano"
REASONING_EFFORT = "low"
BATCH_TOKEN_BUDGET = 5_000
LLM_CONCURRENCY = 100


# --- Structured output schema ---

class ExtractedPainpoint(BaseModel):
    title: str = Field(description="Concise name revealing the essence of the pain")
    description: str = Field(description="1-2 sentence explanation of the pain")
    severity: int = Field(ge=1, le=10, description="1 = minor annoyance, 10 = blocking/critical, 5 - moderate annoyance, but not blocking")
    quoted_text: str = Field(description="Verbatim substring from the source post or comment")
    category_name: str = Field(description="Must match a category from the taxonomy, or 'Uncategorized'")
    post_id: int = Field(description="The [Post N] ID from the input")
    comment_id: int | None = Field(default=None, description="The [Comment N] ID, or null if from the post body")


class ExtractionResult(BaseModel):
    painpoints: list[ExtractedPainpoint]


# --- Instructions ---

EXTRACT_INSTRUCTIONS = """\
You are a painpoint extraction engine for developer-tool product research. \
You will receive Reddit posts with their comments. Your job is to identify \
user painpoints that a developer could build a product or tool around.

Focus on painpoints that are:
- ACTIONABLE: someone could build an app, tool, extension, API, or service \
to solve this problem.
- SPECIFIC: a concrete workflow friction, missing feature, or broken \
experience — not a vague complaint or social commentary.
- TECHNICAL: related to software development, tooling, infrastructure, \
developer workflow, or developer experience.

Skip painpoints that are:
- Pure opinions, memes, jokes, or sarcasm with no real pain behind them.
- Pricing/business model complaints (unless they reveal a feature gap).
- Social/career anxieties (e.g. "AI will take our jobs").
- Platform politics or company drama.

Rules:
- A single post/comment may contain zero, one, or many painpoints.
- Different comments on the same post may yield different painpoints.
- quoted_text must be a verbatim substring from the source text.
- If no category fits well, use "Uncategorized".

## Category taxonomy

{taxonomy}"""


# ============================================================
# Public API
# ============================================================

async def extract_painpoints(post_ids, *, batch_token_budget=BATCH_TOKEN_BUDGET):
    """Extract painpoints from persisted posts via LLM.

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

    posts_with_comments = _load_posts_with_comments(post_ids)
    batches = _build_batches(posts_with_comments, batch_token_budget)
    log.info("extract: %d posts -> %d batches (budget %dK tokens)",
             len(post_ids), len(batches), batch_token_budget // 1000)

    instructions = _build_instructions()
    client = get_client()
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    counter = TokenCounter()

    async def _process_batch(batch_idx, batch):
        async with sem:
            batch_text = _format_batch(batch)
            log.info("extract: batch %d/%d (%d posts)",
                     batch_idx + 1, len(batches), len(batch))
            try:
                result = await asyncio.to_thread(
                    llm_call, client, instructions, batch_text,
                    response_model=ExtractionResult, model=MODEL,
                    max_tokens=None, reasoning_effort=REASONING_EFFORT,
                    token_counter=counter,
                )
            except Exception as exc:
                log.warning("extract: batch %d/%d failed: %s",
                            batch_idx + 1, len(batches), exc)
                return []
            log.info("extract: batch %d/%d -> %d painpoints",
                     batch_idx + 1, len(batches), len(result.painpoints))
            return result.painpoints

    batch_results = await asyncio.gather(
        *(_process_batch(i, b) for i, b in enumerate(batches))
    )

    items = [pp.model_dump() for pps in batch_results for pp in pps]

    usage = counter.as_dict()
    log.info("extract: tokens — input: %d, output: %d (reasoning: %d, text: %d)",
             usage["input_tokens"], usage["output_tokens"],
             usage["reasoning_tokens"], usage["text_tokens"])

    if not items:
        log.info("extract: no painpoints found")
        return [], usage

    ids = save_pending_painpoints_batch(items)
    log.info("extract: saved %d pending painpoints", len(ids))
    return ids, usage


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
    """Build the system-level instructions with the current category taxonomy."""
    categories = get_category_list_flat()
    if categories:
        taxonomy_lines = [
            f"- {c['path']}: {c['description'] or '(no description)'}"
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
