"""Inti Fuzzy Sugeno (scikit-fuzzy) untuk Adaptive Variable Ratio Reward Scheduling E-Block.

Alur mengikuti format jurnal: fuzzifikasi input -> evaluasi aturan (MIN)
-> agregasi output (rata-rata tertimbang) -> hasil akhir.
"""
import math
import numpy as np
import skfuzzy as fuzz

# ---------- Himpunan fuzzy & domain (sesuai catatan) ----------
SETS = {
    "Performance": {
        "universe": (0, 100),
        "sets": {
            "Rendah": ("Bahu kiri", [0, 0, 40]),
            "Baik": ("Segitiga", [30, 50, 70]),
            "Sangat_Baik": ("Bahu kanan", [60, 100, 100]),
        },
    },
    "Improvement": {
        "universe": (0, 100),
        "sets": {
            "Rendah": ("Bahu kiri", [0, 0, 40]),
            "Cukup": ("Segitiga", [30, 50, 70]),
            "Tinggi": ("Bahu kanan", [60, 100, 100]),
        },
    },
    "Attempts": {
        "universe": (1, 5),
        "sets": {
            "Rendah": ("Bahu kiri", [1, 1, 3]),
            "Cukup": ("Segitiga", [2, 3, 4]),
            "Tinggi": ("Bahu kanan", [3, 5, 5]),
        },
    },
}

# ---------- Rule base (Performance, Improvement, Attempts, z) ----------
RULES = [
    ("Rendah", "Rendah", "Rendah", 3), ("Rendah", "Rendah", "Cukup", 2), ("Rendah", "Rendah", "Tinggi", 1),
    ("Rendah", "Cukup", "Rendah", 2), ("Rendah", "Cukup", "Cukup", 2), ("Rendah", "Cukup", "Tinggi", 1),
    ("Rendah", "Tinggi", "Rendah", 0), ("Rendah", "Tinggi", "Cukup", 2), ("Rendah", "Tinggi", "Tinggi", 3),
    ("Baik", "Rendah", "Rendah", 2), ("Baik", "Rendah", "Cukup", 2), ("Baik", "Rendah", "Tinggi", 3),
    ("Baik", "Cukup", "Rendah", 1), ("Baik", "Cukup", "Cukup", 2), ("Baik", "Cukup", "Tinggi", 2),
    ("Baik", "Tinggi", "Rendah", 0), ("Baik", "Tinggi", "Cukup", 1), ("Baik", "Tinggi", "Tinggi", 2),
    ("Sangat_Baik", "Rendah", "Rendah", 0), ("Sangat_Baik", "Rendah", "Cukup", 1), ("Sangat_Baik", "Rendah", "Tinggi", 2),
    ("Sangat_Baik", "Cukup", "Rendah", 0), ("Sangat_Baik", "Cukup", "Cukup", 0), ("Sangat_Baik", "Cukup", "Tinggi", 1),
    ("Sangat_Baik", "Tinggi", "Rendah", 0), ("Sangat_Baik", "Tinggi", "Cukup", 1), ("Sangat_Baik", "Tinggi", "Tinggi", 2),
]

POOL = {
    "Poin langsung": ["+25", "+50", "+75"],
    "Multiplier skor": ["x1.25", "x1.5", "x1.75"],
}


# ---------- Format angka gaya Indonesia (koma desimal) ----------
def fmt(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


def fmt_in(v):
    """Angka masukan: maksimal 2 desimal, nol di belakang dibuang."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def label(term):
    return term.replace("_", " ")


# ---------- Variabel turunan ----------
def calculate_performance(score, time_min, ref_time):
    """Performance = skor * (1 - waktu / (2 * rata-rata waktu)), dibatasi 0-100."""
    if not 0 <= score <= 100:
        raise ValueError("Skor harus 0-100.")
    if time_min < 0:
        raise ValueError("Waktu tidak boleh negatif.")
    if ref_time <= 0:
        raise ValueError("Rata-rata waktu harus > 0.")
    return float(max(0.0, min(100.0, score * (1 - time_min / (2 * ref_time)))))


def calculate_improvement(score, prev_score=None):
    if prev_score is None:
        return 0.0
    return float(max(0.0, score - prev_score))


# ---------- Fuzzifikasi ----------
def mu(x, params):
    """Derajat keanggotaan via scikit-fuzzy trimf (dibulatkan 4 desimal)."""
    return round(float(fuzz.trimf(np.array([x], dtype=float), params)[0]), 4)


def mu_steps(x, shape, p):
    """Teks langkah perhitungan μ(x) seperti pada jurnal."""
    a, b, c = p
    X = fmt_in(x)
    if shape == "Bahu kiri":
        if x <= b:
            return "= 1, karena x ≤ " + str(b)
        if x >= c:
            return "= 0, karena x ≥ " + str(c)
        return f"= ({c} − {X}) / ({c} − {b}) = {fmt(round((c - x) / (c - b), 4))}"
    if shape == "Bahu kanan":
        if x >= b:
            return "= 1, karena x ≥ " + str(b)
        if x <= a:
            return "= 0, karena x ≤ " + str(a)
        return f"= ({X} − {a}) / ({b} − {a}) = {fmt(round((x - a) / (b - a), 4))}"
    # Segitiga
    if x <= a:
        return "= 0, karena x ≤ " + str(a)
    if x >= c:
        return "= 0, karena x ≥ " + str(c)
    if x == b:
        return "= 1, karena x = " + str(b)
    if x < b:
        return f"= ({X} − {a}) / ({b} − {a}) = {fmt(round((x - a) / (b - a), 4))}"
    return f"= ({c} − {X}) / ({c} − {b}) = {fmt(round((c - x) / (c - b), 4))}"


def fuzzify(perf, imp, att):
    """Kembalikan {var: {term: {'mu': float, 'text': str}}}."""
    values = {"Performance": perf, "Improvement": imp, "Attempts": att}
    out = {}
    for var, cfg in SETS.items():
        x = values[var]
        out[var] = {}
        for term, (shape, p) in cfg["sets"].items():
            out[var][term] = {
                "mu": mu(x, p),
                "text": f"μ{label(term)}({fmt_in(x)}) {mu_steps(x, shape, p)}",
            }
    return out


# ---------- Evaluasi aturan & agregasi ----------
def evaluate_rules(mem):
    """alpha = MIN dari derajat keanggotaan antecedent."""
    rows = []
    for i, (p, im, a, z) in enumerate(RULES, start=1):
        mp, mi, ma = mem["Performance"][p]["mu"], mem["Improvement"][im]["mu"], mem["Attempts"][a]["mu"]
        alpha = min(mp, mi, ma)
        base = (f"Performance {label(p)} ({fmt(mp)}) AND Improvement {label(im)} ({fmt(mi)}) "
                f"AND Attempts {label(a)} ({fmt(ma)})")
        if alpha > 0:
            text = f"{base} = MIN({fmt(mp)}; {fmt(mi)}; {fmt(ma)}) = {fmt(alpha)}"
        else:
            text = f"{base} = 0"
        rows.append({"no": i, "text": text, "alpha": alpha, "z": z, "active": alpha > 0})
    return rows


def aggregate(rows):
    active = [r for r in rows if r["active"]]
    if not active:
        raise ValueError("Tidak ada aturan aktif.")
    num = sum(r["alpha"] * r["z"] for r in active)
    den = sum(r["alpha"] for r in active)
    return active, num, den, num / den


def lx(v, nd=4):
    return fmt(v, nd).replace(",", "{,}")


def aggregation_latex(active, num, den, w):
    top = " + ".join(f"({lx(r['alpha'])} \\times {r['z']})" for r in active)
    bot = " + ".join(lx(r["alpha"]) for r in active)
    return (r"W = \frac{\sum(\alpha \cdot z)}{\sum \alpha}"
            f" = \\frac{{{top}}}{{{bot}}}"
            f" = \\frac{{{lx(num)}}}{{{lx(den)}}} = {lx(w)}")


def decision(w):
    """N = 0 jika W<0,5; 1 jika 0,5<=W<1,5; 2 jika 1,5<=W<2,5; 3 jika W>=2,5."""
    n = max(0, min(3, int(math.floor(w + 0.5))))
    if n == 0:
        text = "Reward diberikan LANGSUNG pada challenge yang baru selesai."
    else:
        text = f"Reward diberikan setelah {n} challenge berikutnya diselesaikan."
    return n, text


# ---------- Rumus fungsi keanggotaan (untuk bagian acuan) ----------
def mf_latex(shape, p):
    a, b, c = p
    if shape == "Bahu kiri":
        rows = [f"1, & x \\le {b}",
                f"\\frac{{{c}-x}}{{{c}-{b}}}, & {b} \\le x \\le {c}",
                f"0, & x \\ge {c}"]
    elif shape == "Bahu kanan":
        rows = [f"0, & x \\le {a}",
                f"\\frac{{x-{a}}}{{{b}-{a}}}, & {a} \\le x \\le {b}",
                f"1, & x \\ge {b}"]
    else:
        rows = [f"0, & x \\le {a} \\text{{ atau }} x \\ge {c}",
                f"\\frac{{x-{a}}}{{{b}-{a}}}, & {a} \\le x \\le {b}",
                f"\\frac{{{c}-x}}{{{c}-{b}}}, & {b} \\le x \\le {c}"]
    return r"\mu[x]=\begin{cases}" + r"\\".join(rows) + r"\end{cases}"


# ---------- Pipeline ----------
def run(score, time_min, ref_time, prev_score, attempts):
    perf = round(calculate_performance(score, time_min, ref_time), 2)
    imp = round(calculate_improvement(score, prev_score), 2)
    mem = fuzzify(perf, imp, attempts)
    rows = evaluate_rules(mem)
    active, num, den, w = aggregate(rows)
    n, text = decision(w)
    return dict(perf=perf, imp=imp, att=attempts, mem=mem, rows=rows, active=active,
                num=num, den=den, w=w, n=n, text=text,
                latex=aggregation_latex(active, num, den, w))
