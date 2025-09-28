# resonance_ui.py
import streamlit as st
from typing import List
from resonance_core import run_resonance

st.set_page_config(page_title="Wikipedia Resonance", layout="wide")

# Title
st.title("🔍 Wikipedia Resonance Explorer")

# Main explanation
st.markdown("""
Welcome to the **Wikipedia Resonance Explorer**!  
This tool helps you find Wikipedia pages that **connect multiple seed terms** 
(e.g., people, topics, events) across languages.  
Only pages that literally mention *all* the seeds are shown by default.  

---
""")

# Instructions in a collapsible section
with st.expander("📖 How to use this tool", expanded=True):
    st.markdown("""
    1. **Enter seeds** in the box below, separated by commas.  
       Example: `aho, kirjailija` or `rantanen, harmonikka`.
    2. **Adjust options** if you want (semantic threshold τ, neighborhood size, etc.).
    3. Press **Run**. Results will appear below with:
       - A **score** showing how strongly the page resonates with all seeds.  
       - A **clickable link** to the Wikipedia page.  
       - **One sentence per seed** where the seed is explicitly mentioned.
    4. If a seed doesn’t appear in the summary, you’ll see a note.  
       (Sometimes it’s only in the page title or deeper in the page.)
    ---
    **Tips**:
    - Lower τ if you want *more results* (looser semantic filter).  
    - Increase neighborhood size if you want to explore *further connections*.  
    - Use **strict semantic gate** if you want *both literal and semantic agreement*.  
    """)

# Input box
seeds_text = st.text_input("✏️ Enter your seeds (comma-separated)", value="aho, kirjailija")

# Options
colL, colR = st.columns([2,1])
with colL:
    tau = st.slider("Semantic threshold (τ)", 0.10, 0.60, 0.28, 0.01)
    per_node = st.slider("Neighborhood size per seed", 40, 200, 80, 5)
    probe_k = st.slider("Probe K for fuzzy", 16, 256, 64, 16)
    topk = st.slider("Results to show", 5, 50, 25, 1)
with colR:
    agg = st.selectbox("AND aggregator", ["geo","min","mean"], index=0)
    strict_gate = st.checkbox("Require semantic AND (strict)", value=False)
    literal_required = st.checkbox("Require literal ALL-seeds", value=True)
    fast_fuzzy = st.checkbox("Fast fuzzy (approximate)", value=False)

# Run button
run_btn = st.button("🚀 Run resonance search")

if run_btn:
    seeds: List[str] = [s.strip() for s in seeds_text.split(",") if s.strip()]
    if not seeds:
        st.error("⚠️ Please enter at least one seed.")
    else:
        with st.spinner("Fetching, embedding, and ranking… please wait."):
            result = run_resonance(
                seeds=seeds,
                tau=tau,
                per_node=per_node,
                probe_k=probe_k,
                topk=topk,
                agg=agg,
                use_gain=True,
                fast_fuzzy=fast_fuzzy,
                require_literal=literal_required,
                require_sem_literal=strict_gate,
                model_name=None
            )
        resolved_disp = ", ".join([f"{lg}:{ti}" for (lg,ti,_) in result["resolved"]])
        st.subheader(f"Results for seeds: {resolved_disp} (agg={agg})")

        rows = result["rows"][:topk]
        if not rows:
            st.info("ℹ️ No pages satisfy the configured gates under current thresholds.")
        else:
            aliases = result["aliases_per_seed"]
            for row in rows:
                st.markdown(f"### {row['score']:.3f} — [{row['title']}]({row['url']})")
                for idx, sent in enumerate(row["per_seed_sentences"]):
                    label = aliases[idx][0] if aliases[idx] else f"seed{idx+1}"
                    if sent:
                        st.markdown(f"&nbsp;&nbsp;• **{label}:** {sent}")
                    else:
                        st.markdown(f"&nbsp;&nbsp;• **{label}:** *(no summary sentence; check title or full page)*")
                st.markdown("---")
