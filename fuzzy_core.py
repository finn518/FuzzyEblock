"""Inti Fuzzy Sugeno (scikit-fuzzy) untuk Adaptive Variable Ratio Reward Scheduling E-Block."""
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


def calculate_performance(score, time_min, ref_time):
    """Performance = skor * (1 - waktu / (2 * waktu_acuan)), dibatasi 0-100."""
    if not 0 <= score <= 100:
        raise ValueError("Skor harus 0-100.")
    if time_min < 0:
        raise ValueError("Waktu tidak boleh negatif.")
    if ref_time <= 0:
        raise ValueError("Waktu acuan/rata-rata harus > 0.")
    return float(max(0.0, min(100.0, score * (1 - time_min / (2 * ref_time)))))


def calculate_improvement(score, prev_score=None):
    if prev_score is None:
        return 0.0
    return float(max(0.0, score - prev_score))


def mu(x, params):
    """Derajat keanggotaan via scikit-fuzzy trimf."""
    return float(fuzz.trimf(np.array([x], dtype=float), params)[0])


def fuzzify(perf, imp, att):
    values = {"Performance": perf, "Improvement": imp, "Attempts": att}
    return {
        var: {term: mu(values[var], p) for term, (_, p) in cfg["sets"].items()}
        for var, cfg in SETS.items()
    }


def infer(memberships):
    """Inferensi Sugeno: alpha = MIN, output = sum(alpha*z)/sum(alpha)."""
    rows = []
    num = den = 0.0
    for i, (p, im, a, z) in enumerate(RULES, start=1):
        alpha = min(
            memberships["Performance"][p],
            memberships["Improvement"][im],
            memberships["Attempts"][a],
        )
        rows.append({"Rule": f"R{i}", "Performance": p, "Improvement": im,
                     "Attempts": a, "z": z, "alpha (MIN)": alpha, "alpha x z": alpha * z})
        if alpha > 0:
            num += alpha * z
            den += alpha
    if den == 0:
        raise ValueError("Tidak ada aturan aktif.")
    return rows, num, den, num / den


def decision(w):
    n = max(0, min(3, int(round(w))))
    if n == 0:
        text = "Reward diberikan LANGSUNG pada challenge yang baru selesai."
    else:
        text = f"Reward diberikan setelah {n} challenge berikutnya diselesaikan."
    return n, text


def run(score, time_min, ref_time, prev_score, attempts):
    perf = calculate_performance(score, time_min, ref_time)
    imp = calculate_improvement(score, prev_score)
    mem = fuzzify(perf, imp, attempts)
    rows, num, den, w = infer(mem)
    n, text = decision(w)
    return dict(perf=perf, imp=imp, att=attempts, mem=mem, rows=rows,
                num=num, den=den, w=w, n=n, text=text)
