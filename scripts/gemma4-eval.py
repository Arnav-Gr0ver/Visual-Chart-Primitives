import asyncio
import json
import os
import time
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

os.environ["GEMINI_API_KEY"] = "KEY HERE"
client = genai.Client()

# Load dataset
dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
dataset_sample = dataset.select(range(500))

# Free tier: 30 requests/minute for this model. Use concurrency=1 and a fixed
# delay between dispatches so we never exceed the limit, rather than sleeping
# after the call completes (which doesn't bound the dispatch rate).
MIN_INTERVAL = 2.1  # seconds between request starts -> ~28.5 req/min, safely under 30
_last_dispatch = 0.0
_dispatch_lock = asyncio.Lock()

results = []

async def rate_limited_wait():
    global _last_dispatch
    async with _dispatch_lock:
        now = time.time()
        wait = MIN_INTERVAL - (now - _last_dispatch)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_dispatch = time.time()

def split_reasoning_and_final(response):
    """Thinking mode returns multiple parts; parts marked with thought=True
    are the reasoning trace, the rest is the final answer."""
    reasoning_text = ""
    final_text = ""
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError):
        return "", (response.text or "").strip()

    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            reasoning_text += text
        else:
            final_text += text

    return reasoning_text.strip(), final_text.strip()

async def process_item(idx, item, max_retries=5):
    image = item["image"]
    question = item["query"]
    ground_truth = item["label"]

    prompt = f"Answer the following question about the chart concisely:\n{question}"

    for attempt in range(max_retries):
        await rate_limited_wait()
        try:
            response = await client.aio.models.generate_content(
                model="gemma-4-31b-it",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH",
                        include_thoughts=True,
                    )
                ),
            )
            reasoning, prediction = split_reasoning_and_final(response)
            return {
                "id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "reasoning": reasoning,
                "prediction": prediction,
            }
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str and attempt < max_retries - 1:
                # back off extra hard on rate limit hits, then retry
                await asyncio.sleep(5 * (attempt + 1))
                continue
            return {
                "id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "reasoning": "",
                "prediction": f"Error: {e}",
            }

async def main():
    # concurrency=1: the rate_limited_wait lock already serializes dispatch,
    # so higher concurrency here would just contend on the lock with no benefit
    tasks = [process_item(idx, item) for idx, item in enumerate(dataset_sample)]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating Async ChartQA (thinking on)")

    with open("gemma4_31b_chartqa_results_thinking.json", "w") as f:
        json.dump(results, f, indent=2)

    errors = sum(1 for r in results if r["prediction"].startswith("Error:"))
    with_reasoning = sum(1 for r in results if r["reasoning"])
    print(f"\nDone. {len(results) - errors}/{len(results)} succeeded, {errors} errors.")
    print(f"{with_reasoning}/{len(results)} returned a non-empty reasoning trace.")

asyncio.run(main())