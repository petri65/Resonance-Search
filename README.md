RESONANCE SEARCH

- Find Wikipedia pages that connect multiple seed terms (AND-style) using multilingual embeddings and lightweight graph cues.
- Supports a friendly Streamlit UI with clickable results and per-seed verification sentences.


✨ What this tool does?
- Takes seeds (e.g., trump, president) and resolves them to Wikipedia titles.
- Expands a union neighborhood (links + backlinks) around each seed.
- Computes semantic resonance per seed and aggregates across seeds (geo by default).
- Literal gate (on by default in the UI and core): only show pages that mention all seeds (in title or summary).

- Under each result, shows one sentence per seed from the page summary where that seed is mentioned—easy to verify.
- The first run downloads a ~100 MB multilingual MiniLM model; later runs use the cached copy.

  

Requirements:
- Python 3.9+ recommended
- macOS / Linux / Windows supported

- Install dependencies (from your preferred shell): 
(from the repo root)
python3 -m pip install --upgrade pip    # from the repo root
python3 -m pip install sentence-transformers requests langdetect numpy streamlit

- Tip (recommended): use a virtual environment outside the repo (to avoid committing it):
python3 -m venv ~/.venvs/resonance
source ~/.venvs/resonance/bin/activate  # macOS/Linux
(Windows:  %USERPROFILE%\.venvs\resonance\Scripts\activate)



Quick Start (Streamlit UI):
The UI gives you a clean, interactive window with instructions, options, and clickable links.
(from the repo root)
python3 -m streamlit run resonance_ui.py

Then open the browser tab that appears. You’ll see:
- A text box for seeds (e.g., trump, president)
- Toggles for strict semantic AND, literal gate, etc.
- Buttons to run and view results
- For each result: a score, clickable page title, and one sentence per seed as verification

  
NOTE: macOS double-click launcher (optional):

Create a file named Launch Resonance.command in the repo root with:
#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install --user streamlit sentence-transformers requests langdetect numpy
python3 -m streamlit run resonance_ui.py

Make it executable once:
chmod +x "Launch Resonance.command"
Now you can double-click it in Finder to open the UI (no Terminal commands needed).

Command-line Mode (advanced):
If you prefer the terminal, use the original CLI script:
python3 wiki_resonance_stream.py --seeds "trump, president" --topk 25

Useful flags:
--strict_gate — require both semantic-AND and literal-AND.
--no_literal_and — disable the default literal ALL-seeds requirement (UI keeps literal ON by default).
--tau 0.28 — semantic threshold; lower for more results.
--per_node 120 — neighborhood size (links/backlinks) per seed.
--probe_k 128 — more probes for fuzzy scoring (slower, higher quality).
--agg geo|min|mean — AND aggregator across seeds (geo is default/sensible).

Example (very strict):
python3 wiki_resonance_stream.py --seeds "trump, president" --strict_gate --topk 20
The CLI also prints, under each result, one sentence per seed showing where that seed is mentioned.



Project Layout:
.
├── wiki_resonance_stream.py  # CLI script (prints results + per-seed sentences)
├── resonance_core.py         # Core logic (used by the UI)
├── resonance_ui.py           # Streamlit app (interactive, clickable)
├── Launch Resonance.command  # macOS launcher (optional)
└── README.md

If you only see the CLI script, add resonance_core.py and resonance_ui.py from this repo commit.



Tips for Better Results:
- Use specific seeds (e.g., “Donald Trump” instead of “Trump”).
- Add a language variant seed if helpful (e.g., ice hockey along with jääkiekko).
- If results are sparse, try:
    - lowering τ slightly (--tau 0.25)
    - increasing neighborhood size (--per_node 120)
    - increasing probes (--probe_k 128)
- Keep literal ALL-seeds ON (default in the UI) to ensure each page really mentions every seed.



Troubleshooting:
Links aren’t clickable in Terminal
Use the Streamlit UI (python3 -m streamlit run resonance_ui.py)—links are standard HTML there.

Push to GitHub fails: file > 100 MB
You likely committed a virtualenv or large cache.
- Remove venv/, .venv*/, etc.
- Add a .gitignore (already included in this repo).
- Re-init and push again (see the issue logs for exact steps).
  
First run is slow:
Model download happens once (~100 MB). Later runs use the cached model.



Authentication to GitHub:
Use GitHub CLI browser login:
brew install gh
gh auth login   # choose GitHub.com → HTTPS → Login with a web browser




License & Contributions:
- Licensed under MIT (or your choice—edit this line if you prefer a different license).
- PRs welcome! Please:
    - Keep virtualenvs and large model files out of the repo.
    - Include a short description and a quick way to test your change.

      
 
Acknowledgements:
- Sentence embeddings via sentence-transformers
- Wikipedia summaries via public REST endpoints (respectful usage)

  


One-liner Recap:
(Install and run the UI)
python3 -m pip install sentence-transformers requests langdetect numpy streamlit
python3 -m streamlit run resonance_ui.py


Happy exploring!
