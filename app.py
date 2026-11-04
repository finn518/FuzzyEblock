import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from fuzzy_core import SETS, RULES, POOL, run

st.set_page_config(page_title="E-Block Adaptive Variable Ratio", layout="wide")
st.title("Adaptive Variable Ratio Reward Scheduling")
st.caption("E-Block | Fuzzy Sugeno (scikit-fuzzy)")

# =============== 1. PROTOTIPE ===============
st.header("1. Prototipe")
c1, c2 = st.columns(2)
with c1:
    score = st.number_input("Skor saat ini (0-100)", 0.0, 100.0, 80.0, 1.0)
    time_min = st.number_input("Waktu pengerjaan (menit)", 0.0, value=5.0, step=0.5)
    ref_time = st.number_input("Rata-rata waktu pengerjaan / acuan (menit)", 0.1, value=10.0, step=0.5)
with c2:
    has_prev = st.checkbox("Ada skor sebelumnya (percobaan ke-2 dst.)", value=False)
    prev_score = st.number_input("Skor sebelumnya (0-100)", 0.0, 100.0, 0.0, 1.0, disabled=not has_prev)
    attempts = st.slider("Attempts (1-5)", 1, 5, 1)

if st.button("Hitung", type="primary"):
    st.session_state["res"] = run(score, time_min, ref_time, prev_score if has_prev else None, attempts)

res = st.session_state.get("res")
if res:
    # --- Input
    st.subheader("Input")
    st.dataframe(pd.DataFrame({
        "Variabel": ["Skor", "Waktu (menit)", "Rata-rata waktu (menit)", "Skor sebelumnya",
                     "Performance", "Improvement", "Attempts"],
        "Nilai": [f"{score:.2f}", f"{time_min:.2f}", f"{ref_time:.2f}",
                  f"{prev_score:.2f}" if has_prev else "-",
                  f"{res['perf']:.2f}", f"{res['imp']:.2f}", str(res["att"])],
        "Keterangan": ["", "", "Acuan waktu", "Kosong -> Improvement = 0",
                       "skor x (1 - waktu / (2 x rata-rata))", "max(0, skor - skor sebelumnya)", ""],
    }), hide_index=True, width="stretch")

    # --- Keys (fuzzifikasi)
    st.subheader("Keys (Fuzzifikasi)")
    cols = st.columns(3)
    for col, (var, terms) in zip(cols, res["mem"].items()):
        with col:
            st.markdown(f"**{var}**")
            st.dataframe(pd.DataFrame({"Himpunan": list(terms), "Derajat": [round(v, 4) for v in terms.values()]}),
                         hide_index=True, width="stretch")
            active = [t for t, v in terms.items() if v > 0]
            st.write("Aktif: " + (", ".join(f"{t} ({terms[t]:.2f})" for t in active) or "-"))

    # --- Inferensi
    st.subheader("Inferensi")
    df = pd.DataFrame(res["rows"])
    act = df[df["alpha (MIN)"] > 0].copy()
    st.dataframe(act.round(4), hide_index=True, width="stretch")
    st.write(f"Jumlah aturan aktif: **{len(act)}**")

    # --- Defuzzifikasi
    st.subheader("Defuzzifikasi")
    st.code(
        f"Σ(α·z) = {res['num']:.4f}\n"
        f"Σα     = {res['den']:.4f}\n"
        f"W      = Σ(α·z) / Σα = {res['num']:.4f} / {res['den']:.4f} = {res['w']:.4f}\n"
        f"N      = round(W) = {res['n']}"
    )

    # --- Arti
    st.subheader("Arti / Keputusan")
    (st.success if res["n"] == 0 else st.info)(f"N = {res['n']} → {res['text']}")
    st.caption("Reward dipilih dari pool: poin langsung (+25/+50/+75) atau multiplier skor (x1.25/x1.5/x1.75).")

    # --- Plot posisi input
    st.subheader("Posisi input pada fungsi keanggotaan")
    vals = {"Performance": res["perf"], "Improvement": res["imp"], "Attempts": res["att"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    for ax, (var, cfg) in zip(axes, SETS.items()):
        lo, hi = cfg["universe"]
        x = np.linspace(lo, hi, 400)
        for term, (_, p) in cfg["sets"].items():
            ax.plot(x, fuzz.trimf(x, p), label=term)
        ax.axvline(vals[var], color="k", ls="--")
        ax.set_title(f"{var} = {vals[var]:.2f}")
        ax.legend(fontsize=7)
    st.pyplot(fig)

# =============== 2. ACUAN ===============
st.divider()
st.header("2. Acuan: Variabel Input, Himpunan, dan Domain")
rows = []
for var, cfg in SETS.items():
    for term, (shape, p) in cfg["sets"].items():
        rows.append({"Variabel": var, "Himpunan": term, "Bentuk": shape, "Domain [a - b - c]": " - ".join(map(str, p))})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("**Output Sugeno:** jumlah challenge yang harus diselesaikan untuk mendapat reward tambahan (0-3). 0 = langsung mendapat reward.")
st.markdown("**Pool reward:**")
st.dataframe(pd.DataFrame(POOL), hide_index=True)

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 3))
for ax, (var, cfg) in zip(axes2, SETS.items()):
    lo, hi = cfg["universe"]
    x = np.linspace(lo, hi, 400)
    for term, (_, p) in cfg["sets"].items():
        ax.plot(x, fuzz.trimf(x, p), label=term)
    ax.set_title(var)
    ax.legend(fontsize=7)
st.pyplot(fig2)

st.subheader("Rule Base (27 aturan)")
st.dataframe(pd.DataFrame(
    [{"Rule": f"R{i}", "Performance": p, "Improvement": im, "Attempts": a, "Output (z)": z}
     for i, (p, im, a, z) in enumerate(RULES, start=1)]
), hide_index=True, width="stretch")
