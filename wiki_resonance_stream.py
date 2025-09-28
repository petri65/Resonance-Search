#!/usr/bin/env python3
"""
wiki_resonance_stream.py — Multi-seed AND resonance over Wikipedia via MediaWiki APIs.

Default behavior:
- Only keep pages that LITERALLY mention ALL seeds (any of their aliases) somewhere in
  the page title or REST summary. (Disable with --no_literal_and if needed.)
- Under each result, print ONE sentence PER SEED where that seed is mentioned.

Other features:
- Semantic scoring with geo AND aggregator (default) and fuzzy q=0.90
- Optional link/backlink local gain (on by default; disable with --no_gain)
- Optional strict mode that ALSO requires semantic-AND in addition to literal AND (--strict_gate)
- Multilingual title resolution for seeds
"""

import argparse, time, re
from typing import List, Iterable, Optional, Dict, Tuple, Set
import numpy as np
import requests
import unicodedata  # robust literal matching

# ---------- Config ----------
UA = {"User-Agent": "resonance-stream/1.8 (research; respectful)"}
CANDIDATE_LANGS = ["fi","en","sv","de","fr","es","it","nl","ru","pl","ja","zh"]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------- Terminal hyperlinks ----------
def hyperlink(text: str, url: str, enable: bool = True) -> str:
    if not enable:
        return url
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"

# ---------- Endpoints ----------
def endpoints(lang: str) -> Tuple[str, str, str]:
    base = f"https://{lang}.wikipedia.org"
    api = f"{base}/w/api.php"
    rest_sum = f"{base}/api/rest_v1/page/summary/"
    return base, api, rest_sum

# ---------- Utils ----------
def safe_array(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = safe_array(X)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return X / n

def dot_sane(A: np.ndarray, B_T: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        Z = safe_array(A) @ safe_array(B_T)
    return safe_array(Z)

def normalize_title(t: str) -> str:
    return t.replace("_"," ").strip()

def title_to_url(lang: str, t: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/" + t.replace(" ", "_")

def api_call(api_url: str, params: dict, sleep: float = 0.03) -> dict:
    p = dict(params); p["format"] = "json"; p["formatversion"] = 2
    r = requests.get(api_url, params=p, headers=UA, timeout=20)
    time.sleep(sleep)
    r.raise_for_status()
    return r.json()

def get_summary_text(rest_sum: str, title: str, max_chars: int = 6000) -> str:
    url = rest_sum + requests.utils.quote(title.replace(" ", "_"))
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200: return ""
        return (r.json().get("extract") or "")[:max_chars].strip()
    except Exception:
        return ""

def get_links(api_url: str, title: str, limit_per_dir: int = 200) -> List[str]:
    title = normalize_title(title)
    out: List[str] = []; cont = None
    while len(out) < limit_per_dir:
        q = {"action":"query","prop":"links","titles":title,"plnamespace":0,"pllimit":"max"}
        if cont: q["plcontinue"] = cont
        try:
            j = api_call(api_url, q)
        except Exception:
            break
        for l in j.get("query", {}).get("pages", [{}])[0].get("links", []):
            t = l.get("title")
            if t: out.append(t)
        cont = j.get("continue", {}).get("plcontinue")
        if not cont: break
    return out[:limit_per_dir]

def get_backlinks(api_url: str, title: str, limit_per_dir: int = 120) -> List[str]:
    title = normalize_title(title)
    out: List[str] = []; cont = None
    while len(out) < limit_per_dir:
        q = {"action":"query","list":"backlinks","bltitle":title,"blnamespace":0,"bllimit":"max"}
        if cont: q["blcontinue"] = cont
        try:
            j = api_call(api_url, q)
        except Exception:
            break
        out += [b["title"] for b in j.get("query", {}).get("backlinks", []) if "title" in b]
        cont = j.get("continue", {}).get("blcontinue")
        if not cont: break
    return out[:limit_per_dir]

# ---------- Language detection & robust page resolution ----------
def detect_lang(seed: str) -> Optional[str]:
    try:
        from langdetect import detect
        return detect(seed)
    except Exception:
        return None

def _title_exists(api_url: str, title: str) -> bool:
    try:
        j = api_call(api_url, {"action":"query","titles":title,"prop":"info","inprop":"url"})
        pages = j.get("query",{}).get("pages",[])
        if not pages: return False
        p = pages[0]
        return ("missing" not in p) and (p.get("ns",0) == 0)
    except Exception:
        return False

def _exact_with_summary(lang: str, title: str) -> Optional[Tuple[str,str]]:
    _, api, rest = endpoints(lang)
    if _title_exists(api, title):
        s = get_summary_text(rest, title)
        if s: return (lang, title)
    return None

def _search_nearmatch(lang: str, seed: str) -> Optional[Tuple[str,str]]:
    _, api, rest = endpoints(lang)
    try:
        js = api_call(api, {
            "action": "query",
            "list": "search",
            "srsearch": seed,
            "srwhat": "nearmatch",
            "srlimit": 1
        })
        hits = js.get("query", {}).get("search", [])
        if not hits: return None
        title = hits[0].get("title")
        if not title: return None
        s = get_summary_text(rest, title)
        if s: return (lang, title)
        return None
    except Exception:
        return None

def _boost_lang_order(seed: str, prefer_lang: Optional[str]) -> List[str]:
    base = CANDIDATE_LANGS[:]
    if prefer_lang and prefer_lang in base:
        base.remove(prefer_lang); base.insert(0, prefer_lang)
    tok = seed.strip()
    if tok and " " not in tok and tok.lower().endswith("nen") and "fi" in base:
        base.remove("fi"); base.insert(0, "fi")
    return base

def search_best_title(seed: str, prefer_lang: Optional[str]) -> Tuple[str, str]:
    seed_norm = normalize_title(seed)
    langs = _boost_lang_order(seed_norm, prefer_lang)
    variants = {seed_norm, seed_norm[:1].upper() + seed_norm[1:]}
    for lang in langs:
        for v in variants:
            hit = _exact_with_summary(lang, v)
            if hit: return hit
    for lang in langs:
        hit = _search_nearmatch(lang, seed_norm)
        if hit: return hit
    return "en", seed_norm  # last resort

# ---------- Embedding ----------
class Embedder:
    _singleton = None
    def __new__(cls, model_name: Optional[str] = None):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
            name = model_name or DEFAULT_MODEL
            print(f"[resonance] Loading embedding model '{name}'… (first run may download ~100MB)")
            from sentence_transformers import SentenceTransformer
            cls._singleton.model = SentenceTransformer(name)
        return cls._singleton
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size
        )
        return safe_array(embs)
    def encode_seed(self, seed: str) -> np.ndarray:
        v = self.model.encode([seed], convert_to_numpy=True, normalize_embeddings=True)[0]
        return safe_array(v)

# ---------- Fuzzy & scoring ----------
def fuzzy_from_probes(X: np.ndarray, probes: np.ndarray, q: float = 0.98) -> np.ndarray:
    X = normalize_rows(X); P = normalize_rows(probes)
    sims = dot_sane(X, P.T)
    return np.quantile(sims, q, axis=1)

def topical_prior(strict: np.ndarray, tau: float = 0.28) -> np.ndarray:
    prior = (strict - tau) / max(1e-9, (1.0 - tau))
    return np.clip(prior, 0.0, 1.0)

def best_sentence(text: str, seed_vec: np.ndarray, emb: 'Embedder', max_len: int = 220) -> str:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sents: return ""
    embs = emb.encode_texts(sents, batch_size=64)
    sims = dot_sane(embs, seed_vec[:, None]).ravel()
    s = sents[int(np.argmax(sims))]
    return (s[:max_len] + "…") if len(s) > max_len else s

# ---------- NEW: Literal helpers & snippets ----------
def _norm_text(s: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", s.strip())

def text_mentions_all(title: str, summary: str, aliases: List[List[str]]) -> bool:
    """Every seed must have at least one alias present in title+summary."""
    page = _norm_text(title) + " " + _norm_text(summary or "")
    for alist in aliases:
        ok = any(_norm_text(a) in page for a in alist if a)
        if not ok:
            return False
    return True

def find_sentence_with_any_alias(text: str, aliases: List[str], max_len: int = 220) -> str:
    """Return a sentence (or 2-sentence window) containing ANY alias from this seed."""
    if not text:
        return ""
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    # single sentence
    for s in sents:
        blob = _norm_text(s)
        if any(_norm_text(a) in blob for a in aliases if a):
            return s if len(s) <= max_len else (s[:max_len] + "…")
    # two-sentence window
    for i in range(len(sents) - 1):
        joined = sents[i].strip() + " " + sents[i+1].strip()
        blob = _norm_text(joined)
        if any(_norm_text(a) in blob for a in aliases if a):
            return joined if len(joined) <= max_len else (joined[:max_len] + "…")
    return ""

# ---------- Local gain (optional) ----------
def local_gain(seed_title: str, titles: List[str], lang: str, k_out: int, k_back: int) -> np.ndarray:
    _, api, _ = endpoints(lang)
    try:
        seed_neigh = set(get_links(api, seed_title, k_out)) | set(get_backlinks(api, seed_title, k_back))
    except Exception:
        seed_neigh = set()
    deg_seed = max(1, len(seed_neigh))
    scores = []
    for t in titles:
        if t == seed_title:
            scores.append(1.0); continue
        try:
            neigh = set(get_links(api, t, max(1, k_out // 2))) | set(get_backlinks(api, t, max(1, k_back // 2)))
        except Exception:
            neigh = set()
        if not neigh:
            scores.append(0.0); continue
        direct = 1.0 if t in seed_neigh else 0.0
        jacc = float(len(seed_neigh & neigh)) / max(1.0, float(np.sqrt(deg_seed * len(neigh))))
        scores.append(0.6 * direct + 0.4 * jacc)
    return np.asarray(scores, dtype=float)

# ---------- Single-seed scorer ----------
class ResonanceOne:
    def __init__(self, seed_lang: str, seed_title: str, seed_text: str, emb: 'Embedder',
                 tau: float = 0.28, per_node: int = 80, probe_k: int = 64,
                 use_gain: bool = True, fast_fuzzy: bool = False):
        self.lang = seed_lang
        self.seed_title = normalize_title(seed_title)
        self.seed_text  = seed_text
        self.tau = tau
        self.per_node = per_node
        self.probe_k = probe_k
        self.emb = emb
        self.seed_vec = self.emb.encode_seed(seed_text)
        self.use_gain = use_gain
        self.fast_fuzzy = fast_fuzzy

    def neighborhood(self) -> List[Tuple[str,str]]:
        K1 = self.per_node; K2 = max(1, self.per_node // 2)
        _, api, _ = endpoints(self.lang)
        out = list((set(get_links(api, self.seed_title, K1)) |
                    set(get_backlinks(api, self.seed_title, K2))) - {self.seed_title})
        return [(self.lang, self.seed_title)] + [(self.lang, t) for t in out]

    def score_on_matrix(self, X: np.ndarray, titles: List[Tuple[str,str]], texts: List[str]) -> Dict[Tuple[str,str], dict]:
        seed_vec = self.seed_vec
        strict = dot_sane(X, seed_vec[:, None]).ravel()

        # Probes: seed + top-K strict neighbors (if any)
        if len(titles) > 1:
            neigh_scores = strict[1:]
            top_idx = np.argsort(-neigh_scores)[:min(self.probe_k, len(neigh_scores))]
            probes = np.vstack([seed_vec[np.newaxis,:], X[1:][top_idx]])
        else:
            probes = seed_vec[np.newaxis,:]

        if self.fast_fuzzy:
            sims = dot_sane(X, probes.T)
            fuzzy = np.maximum(strict, sims.mean(axis=1))
        else:
            # Make fuzzy genuinely looser than strict
            fuzzy = np.maximum(strict, fuzzy_from_probes(X, probes, q=0.90))

        if self.use_gain:
            same_lang_idx = [i for i,(lg,_) in enumerate(titles) if lg == self.lang]
            g_local = np.zeros(len(titles), dtype=float)
            if same_lang_idx:
                tl = [titles[i][1] for i in same_lang_idx]
                g_vals = local_gain(self.seed_title, tl, self.lang, self.per_node, max(1,self.per_node//2))
                g_local[same_lang_idx] = g_vals
        else:
            g_local = np.zeros(len(titles), dtype=float)

        amp = 1.0 + np.clip(g_local, 0.0, 1.0)
        resonance = fuzzy * amp
        reveal = np.clip(resonance - strict, 0.0, None)
        prior = topical_prior(strict, self.tau)
        final = reveal * prior

        out: Dict[Tuple[str,str], dict] = {}
        for i, tt in enumerate(titles):
            snippet_text = texts[i] or ""
            snippet = ""
            if snippet_text:
                snippet = best_sentence(snippet_text, seed_vec, self.emb, max_len=220)
            out[tt] = {
                "strict": float(strict[i]),
                "fuzzy": float(fuzzy[i]),
                "gain": float(g_local[i]),
                "reveal": float(reveal[i]),
                "final": float(final[i]),
                "snippet": snippet,
            }
        return out

# ---------- Aggregation (AND) with literal gate ----------
def aggregate_AND(per_seed: List[Dict[Tuple[str,str], dict]],
                  titles: List[Tuple[str,str]],
                  texts: List[str],
                  aliases_per_seed: List[List[str]],
                  tau: float = 0.28,
                  agg: str = "geo",
                  require_literal: bool = True,
                  require_sem_literal: bool = False) -> List[dict]:
    """
    require_literal: if True, drop pages that don't literally contain ALL seed aliases.
    require_sem_literal: if True, require BOTH semantic-AND and literal gate.
    """
    keep = []
    for idx, tt in enumerate(titles):
        # (A) semantic AND: each seed passes (final>0 OR strict>=tau)
        semantic_vals = []
        all_pass = True
        for ps in per_seed:
            if tt not in ps:
                all_pass = False; break
            s = ps[tt]["strict"]; f = ps[tt]["final"]
            pass_one = (f > 0.0) or (s >= tau)
            if not pass_one:
                all_pass = False; break
            semantic_vals.append(max(0.0, f))

        # (B) literal ALL-seeds in title/summary
        co_mention = text_mentions_all(tt[1], texts[idx] or "", aliases_per_seed)

        # Enforce modes
        if require_literal and not co_mention:
            continue
        if require_sem_literal and not (all_pass and co_mention):
            continue

        # Default rule with literal required by default: we already ensured co_mention True.
        if semantic_vals:
            if   agg == "mean": score = float(sum(semantic_vals)/len(semantic_vals))
            elif agg == "geo":  score = float(np.prod([max(v,1e-12) for v in semantic_vals]) ** (1.0/len(semantic_vals)))
            else:               score = float(min(semantic_vals))
        else:
            # Should be rare (literal-only with no semantic pass)
            score = 0.02  # non-zero floor

        keep.append({"lt": tt, "score": score})
    keep.sort(key=lambda r: r["score"], reverse=True)
    return keep

# ---------- Parse seeds ----------
def parse_seed_args(seed_list: Iterable[str], seeds_csv: Optional[str]) -> List[str]:
    seeds: List[str] = []
    if seed_list:
        for s in seed_list:
            s = s.strip()
            if s: seeds.append(s)
    if seeds_csv:
        for s in seeds_csv.split(","):
            s = s.strip()
            if s: seeds.append(s)
    seen = set(); out: List[str] = []
    for s in seeds:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="append", help='Seed title (repeatable), e.g. --seed "Ibuprofen"')
    ap.add_argument("--seeds", help='Comma-separated seeds, e.g. "Rantanen, harmonikka"')
    ap.add_argument("--seed_text", default=None, help="Optional custom phrase; defaults to each seed title")
    # scoring & neighborhood
    ap.add_argument("--tau", type=float, default=0.28)
    ap.add_argument("--per_node", type=int, default=80)
    ap.add_argument("--probe_k", type=int, default=64)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--agg", choices=["min","mean","geo"], default="geo")  # default: geo
    # speed / gating toggles
    ap.add_argument("--no_gain", action="store_true")
    ap.add_argument("--fast_fuzzy", action="store_true")
    ap.add_argument("--strict_gate", action="store_true",
                    help="Require BOTH semantic-AND and literal-AND to pass")
    ap.add_argument("--no_literal_and", action="store_true",
                    help="Disable default literal ALL-seeds requirement")
    ap.add_argument("--gate_cap", type=int, default=400)
    # output
    ap.add_argument("--no_ansi", action="store_true", help="Disable clickable links")
    # model override
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    # Default: literal AND is ON unless explicitly disabled
    literal_required = not args.no_literal_and

    seeds = parse_seed_args(args.seed or [], args.seeds)
    if not seeds:
        typed = input("Enter seeds (comma-separated): ").strip()
        seeds = [s.strip() for s in typed.split(",") if s.strip()]
    if not seeds:
        print("[error] No seeds provided."); return

    # Resolve language+title per seed (with exact-title bias)
    resolved: List[Tuple[str,str,str]] = []  # (lang, title, seed_text)
    aliases_per_seed: List[List[str]] = []
    for s in seeds:
        guessed = detect_lang(s)  # may be None
        lang, title = search_best_title(s, guessed)
        seed_text = (args.seed_text or s)
        resolved.append((lang, title, seed_text))
        # aliases for literal detection: raw seed, resolved title, underscored and spaced variants
        alist = {s, title, title.replace("_"," "), s.replace("_"," ")}
        aliases_per_seed.append(sorted(alist))
        print(f"[resonance] Seed '{s}' → {lang}:{title}")

    # One shared embedder/model
    emb = Embedder(args.model)

    # Build neighborhoods per seed and union
    per_seed_titles: Dict[Tuple[str,str], List[Tuple[str,str]]] = {}
    union_titles_set: Set[Tuple[str,str]] = set()
    for lang, title, seed_text in resolved:
        R1 = ResonanceOne(lang, title, seed_text, emb=emb,
                          tau=args.tau, per_node=args.per_node,
                          probe_k=args.probe_k, use_gain=(not args.no_gain),
                          fast_fuzzy=args.fast_fuzzy)
        neigh = R1.neighborhood()
        per_seed_titles[(lang,title)] = neigh
        union_titles_set.update(neigh)

    # Preserve order
    union_titles: List[Tuple[str,str]] = []
    for key in per_seed_titles:
        for tt in per_seed_titles[key]:
            if tt not in union_titles:
                union_titles.append(tt)
    for tt in list(union_titles_set):
        if tt not in union_titles:
            union_titles.append(tt)

    # Fetch summaries
    print(f"[resonance] Fetching {len(union_titles)} summaries (union of neighborhoods)…")
    texts: List[str] = []
    for (lg, t) in union_titles:
        _, _, rest = endpoints(lg)
        texts.append(get_summary_text(rest, t))

    # Encode once
    print("[resonance] Encoding summaries…")
    X = normalize_rows(emb.encode_texts([tx if tx else "" for tx in texts]))

    # Optional strict-only semantic prefilter (independent of final gating)
    survivors_mask = np.ones(len(union_titles), dtype=bool)
    if args.strict_gate and len(resolved) > 1:
        print("[resonance] Stage A: strict-only AND prefilter…")
        masks = []
        for (_, _, seed_text) in resolved:
            sv = emb.encode_seed(seed_text)
            masks.append(dot_sane(X, sv[:, None]).ravel() >= args.tau)
        survivors_mask = np.logical_and.reduce(masks)
        if survivors_mask.sum() > args.gate_cap:
            sums = np.zeros(len(union_titles), dtype=float)
            for (_, _, seed_text) in resolved:
                sv = emb.encode_seed(seed_text)
                sums += dot_sane(X, sv[:, None]).ravel()
            idx = np.argsort(-sums)[:args.gate_cap]
            survivors_mask = np.zeros(len(union_titles), dtype=bool); survivors_mask[idx] = True

    sel_idx = np.where(survivors_mask)[0]
    titles_sel = [union_titles[i] for i in sel_idx]
    texts_sel = [texts[i] for i in sel_idx]
    X_sel = X[sel_idx]

    # Build a quick lookup for printing per-seed sentences later
    text_by_page = {titles_sel[i]: (texts_sel[i] or "") for i in range(len(titles_sel))}

    # Score per seed
    per_seed_scores: List[Dict[Tuple[str,str], dict]] = []
    for (lg, title, seed_text) in resolved:
        R1 = ResonanceOne(lg, title, seed_text, emb=emb,
                          tau=args.tau, per_node=args.per_node,
                          probe_k=args.probe_k, use_gain=(not args.no_gain),
                          fast_fuzzy=args.fast_fuzzy)
        per_seed_scores.append(R1.score_on_matrix(X_sel, titles_sel, texts_sel))

    # Aggregate with literal AND required by default (+ optional semantic AND)
    rows = aggregate_AND(
        per_seed_scores, titles_sel, texts_sel, aliases_per_seed,
        tau=args.tau, agg=args.agg,
        require_literal=literal_required,
        require_sem_literal=args.strict_gate
    )

    hyperlink_on = not args.no_ansi
    seeds_label = ", ".join([f"{lg}:{ti}" for (lg,ti,_) in resolved])
    print(f"\n=== AND resonance for seeds: {seeds_label} (agg={args.agg}) ===")
    if not rows:
        print("  (no pages satisfy the configured gates under current thresholds)")
        return

    # Print results + ONE sentence PER SEED where that seed is mentioned
    for r in rows[:args.topk]:
        lang, title = r["lt"]
        url = title_to_url(lang, title)
        link = hyperlink(url, url, enable=hyperlink_on)

        page_text = text_by_page.get((lang, title), "")

        print(f"{r['score']:.3f}  {title}  {link}")
        # one verification line per seed
        for i, seed_aliases in enumerate(aliases_per_seed, start=1):
            sent = find_sentence_with_any_alias(page_text, seed_aliases, max_len=220)
            label = seed_aliases[0] if seed_aliases else f"seed{i}"
            if sent:
                print(f"   • {label}: {sent}")
            else:
                print(f"   • {label}: (no sentence found in summary; mention exists elsewhere in title/summary)")

if __name__ == "__main__":
    main()
