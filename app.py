
import random

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from fuzzy_core import SETS, RULES, POOL, run


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="E-Block Adaptive Variable Ratio",
    layout="wide"
)

st.title("Adaptive Variable Ratio Reward Scheduling")
st.caption("E-Block | Fuzzy Sugeno (scikit-fuzzy)")


# ==========================================
# FUNGSI GRAFIK MEMBERSHIP
# ==========================================

def plot_membership(vals=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    for ax, (variabel, cfg) in zip(axes, SETS.items()):
        domain = cfg["domain"]

        for himpunan in cfg:
            if himpunan == "domain":
                continue

            ax.plot(
                domain,
                fuzz.trimf(domain, cfg[himpunan]),
                label=himpunan
            )

        if vals is not None:
            ax.axvline(
                vals[variabel],
                color="k",
                linestyle="--"
            )
            ax.set_title(f"{variabel} = {vals[variabel]:.2f}")
        else:
            ax.set_title(variabel)

        ax.set_xlabel("Nilai")
        ax.set_ylabel("Derajat Keanggotaan")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 1. PROTOTIPE
# ==========================================

st.header("1. Prototipe")

c1, c2 = st.columns(2)

with c1:
    score = st.number_input(
        "Skor saat ini (0-100)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    time_min = st.number_input(
        "Waktu pengerjaan (menit)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    avg_time = st.number_input(
        "Rata-rata waktu pengerjaan / acuan (menit)",
        min_value=0.1,
        value=10.0,
        step=0.5
    )

with c2:
    has_prev = st.checkbox(
        "Ada skor sebelumnya (percobaan ke-2 dst.)",
        value=False
    )

    prev_score = st.number_input(
        "Skor sebelumnya (0-100)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        disabled=not has_prev
    )

    attempts = st.slider(
        "Attempts (1-5)",
        min_value=1,
        max_value=5,
        value=1
    )


# ==========================================
# HITUNG FUZZY SUGENO
# ==========================================

if st.button("Hitung", type="primary"):

    try:
        hasil = run(
            score,
            time_min,
            avg_time,
            prev_score if has_prev else None,
            attempts
        )

        st.session_state["res"] = hasil

        # Reset progress untuk hasil perhitungan baru
        st.session_state["progress"] = 0

        # Reset reward yang sebelumnya diperoleh
        st.session_state["claimed_reward"] = None

    except ValueError as e:
        st.session_state.pop("res", None)
        st.session_state.pop("progress", None)
        st.session_state.pop("claimed_reward", None)

        st.error(str(e))


# ==========================================
# HASIL PERHITUNGAN
# ==========================================

res = st.session_state.get("res")

if res and "score" not in res:
    res = None
    st.session_state.pop("res", None)


if res:

    # ======================================
    # INPUT
    # ======================================

    st.subheader("Input")

    prev = res["prev_score"]

    st.dataframe(
        pd.DataFrame({
            "Variabel": [
                "Skor",
                "Waktu (menit)",
                "Rata-rata waktu (menit)",
                "Skor sebelumnya",
                "Performance",
                "Improvement",
                "Attempts"
            ],

            "Nilai": [
                f"{res['score']:.2f}",
                f"{res['time_min']:.2f}",
                f"{res['ref_time']:.2f}",
                f"{prev:.2f}" if prev is not None else "-",
                f"{res['perf']:.2f}",
                f"{res['imp']:.2f}",
                str(res["att"])
            ],

            "Keterangan": [
                "",
                "",
                "Acuan waktu",
                "Kosong -> Improvement = 0",
                "Skor x (1 - waktu / (2 x rata-rata))",
                "max(0, skor - skor sebelumnya)",
                ""
            ]
        }),
        hide_index=True,
        width="stretch"
    )


    # ======================================
    # FUZZIFIKASI
    # ======================================

    st.subheader("Fuzzifikasi")

    cols = st.columns(3)

    for col, (variabel, memberships) in zip(
        cols,
        res["mem"].items()
    ):

        with col:
            st.markdown(f"**{variabel}**")

            st.dataframe(
                pd.DataFrame({
                    "Himpunan": list(memberships.keys()),
                    "Derajat Keanggotaan": [
                        float(nilai)
                        for nilai in memberships.values()
                    ]
                }).round(4),
                hide_index=True,
                width="stretch"
            )

            aktif = [
                himpunan
                for himpunan, nilai in memberships.items()
                if nilai > 0
            ]

            st.write(
                "Himpunan aktif: " +
                (
                    ", ".join(
                        f"{himpunan} ({memberships[himpunan]:.2f})"
                        for himpunan in aktif
                    )
                    if aktif else "-"
                )
            )


    # ======================================
    # INFERENSI
    # ======================================

    st.subheader("Inferensi")

    df = pd.DataFrame(res["rows"])

    rule_aktif = df[df["alpha (MIN)"] > 0].copy()

    st.dataframe(
        rule_aktif.round(4),
        hide_index=True,
        width="stretch"
    )

    st.write(
        f"Jumlah aturan aktif: **{len(rule_aktif)} dari {len(RULES)}**"
    )


    # ======================================
    # DEFUZZIFIKASI
    # ======================================

    st.subheader("Defuzzifikasi")

    st.code(
        f"Σ(α × z) = {res['num']:.4f}\n"
        f"Σα      = {res['den']:.4f}\n"
        f"W       = Σ(α × z) / Σα\n"
        f"        = {res['num']:.4f} / {res['den']:.4f}\n"
        f"        = {res['w']:.4f}\n\n"
        f"N       = round(W) = {res['n']}"
    )

    st.caption(
        "W merupakan output asli Fuzzy Sugeno, sedangkan N "
        "merupakan jumlah challenge setelah konversi menggunakan round()."
    )


    # ======================================
    # KEPUTUSAN
    # ======================================

    st.subheader("Arti / Keputusan")

    if res["n"] == 0:
        st.success(f"N = {res['n']} → {res['text']}")
    else:
        st.info(f"N = {res['n']} → {res['text']}")

    st.caption(
        "Pool reward: poin langsung (+25/+50/+75) "
        "atau multiplier skor (x1.25/x1.5/x1.75)."
    )



    # ======================================
    # PROGRESS CHALLENGE DAN KLAIM REWARD
    # ======================================

    st.divider()
    st.subheader("Progress Reward")

    target = res["n"]

    if "progress" not in st.session_state:
        st.session_state["progress"] = 0

    if "claimed_reward" not in st.session_state:
        st.session_state["claimed_reward"] = None

    progress = st.session_state["progress"]

    # Dua kolom progress dan tombol challenge
    col_progress, col_challenge = st.columns(2)

    with col_progress:
        st.write("**Progress Challenge**")

        if target == 0:
            st.success("Tidak perlu menyelesaikan challenge tambahan.")
            st.progress(1.0)
            st.write("Progress: 0 / 0")

        else:
            st.write(f"Progress: **{progress} / {target}**")
            st.progress(progress / target)

            # Stamp
            kolom_stamp = st.columns(target)

            for i, kolom in enumerate(kolom_stamp):
                with kolom:
                    if i < progress:
                        st.success(f"✓ Stamp {i + 1}")
                    else:
                        st.warning(f"○ Stamp {i + 1}")

    with col_challenge:
        st.write("**Penyelesaian Challenge**")

        if progress < target:
            st.write(
                f"Selesaikan {target - progress} challenge lagi "
                "untuk membuka reward."
            )

            if st.button(
                "Selesaikan 1 Challenge",
                use_container_width=True,
                type="primary"
            ):
                st.session_state["progress"] += 1
                st.rerun()

        else:
            st.success("Semua challenge telah diselesaikan!")


    # ======================================
    # KLAIM REWARD
    # ======================================

    st.write("")

    if st.session_state["progress"] >= target:
        tombol_klaim_aktif = True
    else:
        tombol_klaim_aktif = False

    if st.button(
        "🎁 Klaim Reward",
        type="primary",
        use_container_width=True,
        disabled=not tombol_klaim_aktif or
        st.session_state["claimed_reward"] is not None
    ):
        daftar_reward = [
            (jenis, nilai)
            for jenis, daftar in POOL.items()
            for nilai in daftar
        ]

        st.session_state["claimed_reward"] = random.choice(
            daftar_reward
        )

        st.rerun()


    # Menampilkan hasil reward
    if st.session_state["claimed_reward"] is not None:
        jenis, nilai = st.session_state["claimed_reward"]

        st.success(
            f"🎉 Selamat! Kamu mendapatkan reward: **{nilai}**"
        )

        st.write(f"Jenis reward: **{jenis}**")


    # ======================================
    # GRAFIK MEMBERSHIP
    # ======================================

    st.subheader("Posisi Input pada Fungsi Keanggotaan")

    plot_membership({
        "Performance": res["perf"],
        "Improvement": res["imp"],
        "Attempts": res["att"]
    })


# ==========================================
# 2. ACUAN
# ==========================================

st.divider()

st.header("2. Acuan: Variabel Input, Himpunan, dan Domain")


# ==========================================
# TABEL HIMPUNAN FUZZY
# ==========================================

rows = []

for variabel, cfg in SETS.items():

    for himpunan in cfg:

        if himpunan == "domain":
            continue

        rows.append({
            "Variabel": variabel,
            "Himpunan": himpunan,
            "Domain": (
                f"{cfg['domain'][0]} - "
                f"{cfg['domain'][-1]}"
            ),
            "Parameter MF (a, b, c)": cfg[himpunan]
        })


st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch"
)


# ==========================================
# OUTPUT SUGENO DAN POOL REWARD
# ==========================================

st.markdown(
    "**Output Sugeno:** jumlah challenge yang harus "
    "diselesaikan untuk mendapat reward tambahan (0-3). "
    "0 = langsung mendapat reward."
)

st.markdown("**Pool reward:**")

st.dataframe(
    pd.DataFrame(POOL),
    hide_index=True,
    width="stretch"
)


# ==========================================
# GRAFIK SELURUH MEMBERSHIP FUNCTION
# ==========================================

st.subheader("Grafik Fungsi Keanggotaan")

plot_membership()


# ==========================================
# RULE BASE
# ==========================================

st.subheader(f"Rule Base ({len(RULES)} aturan)")

st.dataframe(
    pd.DataFrame([
        {
            "Rule": f"R{i}",
            "Performance": performance,
            "Improvement": improvement,
            "Attempts": attempts,
            "Output (z)": z
        }

        for i, (
            performance,
            improvement,
            attempts,
            z
        ) in enumerate(RULES, start=1)
    ]),
    hide_index=True,
    width="stretch"
)
