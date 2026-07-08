"""First-principles Gemini latency probe: isolate thinking vs concurrency vs rate-limit.
Bypasses the whole ControlArena pipeline. Run on the droplet with GOOGLE_API_KEY in env."""
import os, time, concurrent.futures as cf
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
M = "gemini-2.5-flash"
NOTHINK = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
THINK = types.GenerateContentConfig()  # default (thinking on)

def one(cfg, tag, i):
    t = time.time()
    try:
        r = client.models.generate_content(model=M, contents="Reply with the single word: ok.", config=cfg)
        return f"  {tag} {i}: {time.time()-t:5.1f}s  ok={bool(getattr(r,'text',None))}"
    except Exception as e:
        return f"  {tag} {i}: {time.time()-t:5.1f}s  ERR {type(e).__name__}: {str(e)[:90]}"

print("=== SEQUENTIAL, thinking OFF ===")
for i in range(3): print(one(NOTHINK, "seq-nothink", i))
print("=== SEQUENTIAL, thinking ON (default) ===")
for i in range(3): print(one(THINK, "seq-think", i))
print("=== CONCURRENT x6, thinking OFF (rate-limit test) ===")
t0 = time.time()
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(lambda i: one(NOTHINK, "conc", i), range(6)): print(r)
print(f"  (wall for 6 concurrent: {time.time()-t0:.1f}s)")
