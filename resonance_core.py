# resonance_core.py
import time, re, unicodedata
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
import requests

UA = {"User-Agent": "resonance-ui/1.0 (research; respectful)"}
CANDIDATE_LANGS = ["fi","en","sv","de","fr","es","it","nl","ru","pl","ja","zh"]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------- small utils ----------
def _safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def _norm_rows(X: np.ndarray) -> np.ndarray:
    X = _safe(X)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return X / n

def _dot(A: np.ndarray, B_T: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        Z = _safe(A) @ _safe(B_T)
    return _safe(Z)

def _normalize_title(t: str) -> str:
    return t.replace("_"," ").strip()

def _title_url(lang: str, t: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/" + t.replace(" ", "_")

def _endpoints(lang: str):
    base = f"https://{lang}.wikipedia.org"
    api = f"{base}/w/api.php"
    rest = f"{base}/api/rest_v1/page/summary/"
    return base, api, rest

def _api(api_url: str, params: dict, sleep: float = 0.03) -> dict:
    p = dict(params); p["format"]="json"; p["formatversion"]=2
    r = requests.get(api_url, params=p, headers=UA, timeout=20)
    time.sleep(sleep); r.raise_for_status()
    return r.json()

def _summary(rest: str, title: str, max_chars=6000) -> str:
    url = rest + requests.utils.quote(title.replace(" ", "_"))
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200: return ""
        return (r.json().get("extract") or "")[:max_chars].strip()
    except Exception:
        return ""

def _links(api_url: str, title: str, limit=200) -> List[str]:
    title = _normalize_title(title); out=[]; cont=None
    while len(out) < limit:
        q={"action":"query","prop":"links","titles":title,"plnamespace":0,"pllimit":"max"}
        if cont: q["plcontinue"]=cont
        try: j=_api(api_url,q)
        except Exception: break
        for l in j.get("query",{}).get("pages",[{}])[0].get("links",[]):
            t=l.get("title"); 
            if t: out.append(t)
        cont = j.get("continue",{}).get("plcontinue")
        if not cont: break
    return out[:limit]

def _backlinks(api_url: str, title: str, limit=120) -> List[str]:
    title = _normalize_title(title); out=[]; cont=None
    while len(out) < limit:
        q={"action":"query","list":"backlinks","bltitle":title,"blnamespace":0,"bllimit":"max"}
        if cont: q["blcontinue"]=cont
        try: j=_api(api_url,q)
        except Exception: break
        out += [b["title"] for b in j.get("query",{}).get("backlinks",[]) if "title" in b]
        cont = j.get("continue",{}).get("blcontinue")
        if not cont: break
    return out[:limit]

def _detect_lang(seed: str) -> Optional[str]:
    try:
        from langdetect import detect
        return detect(seed)
    except Exception:
        return None

def _title_exists(api_url: str, title: str) -> bool:
    try:
        j = _api(api_url, {"action":"query","titles":title,"prop":"info","inprop":"url"})
        pages = j.get("query",{}).get("pages",[])
        if not pages: return False
        p = pages[0]
        return ("missing" not in p) and (p.get("ns",0)==0)
    except Exception:
        return False

def _exact_with_summary(lang: str, title: str):
    _, api, rest = _endpoints(lang)
    if _title_exists(api, title):
        s=_summary(rest,title)
        if s: return (lang,title)
    return None

def _search_nearmatch(lang: str, seed: str):
    _, api, rest = _endpoints(lang)
    try:
        js=_api(api,{"action":"query","list":"search","srsearch":seed,"srwhat":"nearmatch","srlimit":1})
        hits=js.get("query",{}).get("search",[])
        if not hits: return None
        title=hits[0].get("title"); 
        if not title: return None
        s=_summary(rest,title)
        if s: return (lang,title)
    except Exception:
        pass
    return None

def _boost_langs(seed: str, prefer: Optional[str]) -> List[str]:
    base = CANDIDATE_LANGS[:]
    if prefer and prefer in base: base.remove(prefer); base.insert(0,prefer)
    tok=seed.strip()
    if tok and " " not in tok and tok.lower().endswith("nen") and "fi" in base:
        base.remove("fi"); base.insert(0,"fi")
    return base

def resolve_title(seed: str) -> Tuple[str,str]:
    s=_normalize_title(seed)
    langs=_boost_langs(s,_detect_lang(s))
    variants={s, s[:1].upper()+s[1:]} if s else {s}
    for lang in langs:
        for v in variants:
            hit=_exact_with_summary(lang,v)
            if hit: return hit
    for lang in langs:
        hit=_search_nearmatch(lang,s)
        if hit: return hit
    return "en", s

# ---------- embeddings ----------
class Embedder:
    _singleton=None
    def __new__(cls, model_name: Optional[str] = None):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
            name = model_name or DEFAULT_MODEL
            print(f"[resonance] Loading embedding model '{name}'… (first run may download ~100MB)")
            from sentence_transformers import SentenceTransformer
            cls._singleton.model = SentenceTransformer(name)
        return cls._singleton
    def encode_texts(self, texts: List[str], batch_size=64) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
        return _safe(embs)
    def encode_seed(self, seed: str) -> np.ndarray:
        v = self.model.encode([seed], convert_to_numpy=True, normalize_embeddings=True)[0]
        return _safe(v)

def fuzzy_from_probes(X: np.ndarray, probes: np.ndarray, q: float = 0.90) -> np.ndarray:
    X=_norm_rows(X); P=_norm_rows(probes)
    sims=_dot(X,P.T)
    return np.quantile(sims,q,axis=1)

def topical_prior(strict: np.ndarray, tau: float=0.28) -> np.ndarray:
    prior=(strict - tau)/max(1e-9,(1.0 - tau))
    return np.clip(prior,0.0,1.0)

# ---------- literal helpers ----------
def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+"," ", s.strip())

def text_mentions_all(title: str, summary: str, aliases_per_seed: List[List[str]]) -> bool:
    page = _norm_text(title) + " " + _norm_text(summary or "")
    for alist in aliases_per_seed:
        if not any(_norm_text(a) in page for a in alist if a):
            return False
    return True

def sentence_with_any_alias(text: str, aliases: List[str], max_len=220) -> str:
    if not text: return ""
    sents=[s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    for s in sents:
        blob=_norm_text(s)
        if any(_norm_text(a) in blob for a in aliases if a):
            return s if len(s)<=max_len else (s[:max_len]+"…")
    for i in range(len(sents)-1):
        joined=sents[i].strip()+" "+sents[i+1].strip()
        blob=_norm_text(joined)
        if any(_norm_text(a) in blob for a in aliases if a):
            return joined if len(joined)<=max_len else (joined[:max_len]+"…")
    return ""

# ---------- main API you’ll call from the UI ----------
def run_resonance(
    seeds: List[str],
    tau: float = 0.28,
    per_node: int = 80,
    probe_k: int = 64,
    topk: int = 25,
    agg: str = "geo",
    use_gain: bool = True,
    fast_fuzzy: bool = False,
    require_literal: bool = True,          # default ON per your request
    require_sem_literal: bool = False,     # if True, also require semantic AND
    model_name: Optional[str] = None
):
    # resolve seeds
    resolved=[]  # (lang,title,seed_text)
    aliases_per_seed=[]
    for s in seeds:
        lang,title = resolve_title(s)
        seed_text = s
        resolved.append((lang,title,seed_text))
        aliases_per_seed.append(sorted({s, title, title.replace("_"," "), s.replace("_"," ")}))
        print(f"[resonance] Seed '{s}' → {lang}:{title}")

    emb = Embedder(model_name)

    # neighborhoods union
    union_titles=[]
    seen=set()
    for lang,title,seed_text in resolved:
        _, api, _ = _endpoints(lang)
        neigh = [(lang,title)]
        try:
            neigh += [(lang,t) for t in set(_links(api,title,per_node)) | set(_backlinks(api,title,max(1,per_node//2)))]
        except Exception:
            pass
        for tt in neigh:
            if tt not in seen:
                union_titles.append(tt); seen.add(tt)

    # fetch texts
    texts=[]
    for lg,t in union_titles:
        _,_,rest=_endpoints(lg)
        texts.append(_summary(rest,t))

    # encode
    X=_norm_rows(emb.encode_texts([tx or "" for tx in texts]))

    # per-seed scoring
    per_seed_scores=[]
    for (lg,title,seed_text) in resolved:
        seed_vec = emb.encode_seed(seed_text)
        strict=_dot(X, seed_vec[:,None]).ravel()
        if len(union_titles)>1:
            neigh_scores=strict[1:]
            top_idx=np.argsort(-neigh_scores)[:min(probe_k, len(neigh_scores))]
            probes=np.vstack([seed_vec[np.newaxis,:], X[1:][top_idx]])
        else:
            probes=seed_vec[np.newaxis,:]
        if fast_fuzzy:
            sims=_dot(X, probes.T)
            fuzzy=np.maximum(strict, sims.mean(axis=1))
        else:
            fuzzy=np.maximum(strict, fuzzy_from_probes(X,probes,q=0.90))
        if use_gain:
            # light-weight: no extra calls here (kept 0 gain). You can add gain if desired.
            gain=np.zeros_like(strict)
        else:
            gain=np.zeros_like(strict)
        amp=1.0+np.clip(gain,0.0,1.0)
        resonance=fuzzy*amp
        reveal=np.clip(resonance - strict,0.0,None)
        prior=topical_prior(strict, tau)
        final=reveal*prior
        per_seed_scores.append({union_titles[i]:{"strict":float(strict[i]),"final":float(final[i])} for i in range(len(union_titles))})

    # aggregate with literal gate
    rows=[]
    for i,tt in enumerate(union_titles):
        title=tt[1]; text=texts[i] or ""
        # literal
        co = text_mentions_all(title, text, aliases_per_seed)
        if require_literal and not co:
            continue
        # semantic AND
        sem_vals=[]; all_pass=True
        for ps in per_seed_scores:
            s=ps[tt]["strict"]; f=ps[tt]["final"]
            if not ((f>0.0) or (s>=tau)):
                all_pass=False; break
            sem_vals.append(max(0.0,f))
        if require_sem_literal and not all_pass:
            continue
        # score
        if sem_vals:
            if agg=="mean":
                score=float(sum(sem_vals)/len(sem_vals))
            elif agg=="geo":
                score=float(np.prod([max(v,1e-12) for v in sem_vals])**(1.0/len(sem_vals)))
            else:
                score=float(min(sem_vals))
        else:
            score=0.02
        # per-seed sentences
        per_seed_lines=[]
        for aliases in aliases_per_seed:
            sent = sentence_with_any_alias(text, aliases, max_len=220)
            per_seed_lines.append(sent)
        rows.append({
            "lang": tt[0],
            "title": title,
            "url": _title_url(tt[0], title),
            "score": score,
            "per_seed_sentences": per_seed_lines
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return {
        "resolved": resolved,                  # list[(lang,title,seed_text)]
        "aliases_per_seed": aliases_per_seed,  # list[list[str]]
        "rows": rows                           # list[dict]
    }
