import numpy as np
import skfuzzy as fuzz

# Himpunan fuzzy dan domain
SETS = {
    "Performance": {
        "domain": np.arange(0, 101),
        "Rendah": [0, 0, 40],
        "Baik": [30, 50, 70],
        "Sangat_Baik": [60, 100, 100],
    },
    "Improvement": {
        "domain": np.arange(0, 101),
        "Rendah": [0, 0, 40],
        "Cukup": [30, 50, 70],
        "Tinggi": [60, 100, 100],
    },
    "Attempts": {
        "domain": np.arange(1, 6),
        "Rendah": [1, 1, 3],
        "Cukup": [2, 3, 4],
        "Tinggi": [3, 5, 6],
    },
}

# Rule base: (Performance, Improvement, Attempts, z)
RULES = [
    ("Rendah", "Rendah", "Rendah", 3),
    ("Rendah", "Rendah", "Cukup", 2),
    ("Rendah", "Rendah", "Tinggi", 1),
    ("Rendah", "Cukup", "Rendah", 2),
    ("Rendah", "Cukup", "Cukup", 2),
    ("Rendah", "Cukup", "Tinggi", 1),
    ("Rendah", "Tinggi", "Rendah", 0),
    ("Rendah", "Tinggi", "Cukup", 2),
    ("Rendah", "Tinggi", "Tinggi", 3),

    ("Baik", "Rendah", "Rendah", 2),
    ("Baik", "Rendah", "Cukup", 2),
    ("Baik", "Rendah", "Tinggi", 3),
    ("Baik", "Cukup", "Rendah", 1),
    ("Baik", "Cukup", "Cukup", 2),
    ("Baik", "Cukup", "Tinggi", 2),
    ("Baik", "Tinggi", "Rendah", 0),
    ("Baik", "Tinggi", "Cukup", 1),
    ("Baik", "Tinggi", "Tinggi", 2),

    ("Sangat_Baik", "Rendah", "Rendah", 0),
    ("Sangat_Baik", "Rendah", "Cukup", 1),
    ("Sangat_Baik", "Rendah", "Tinggi", 2),
    ("Sangat_Baik", "Cukup", "Rendah", 0),
    ("Sangat_Baik", "Cukup", "Cukup", 0),
    ("Sangat_Baik", "Cukup", "Tinggi", 1),
    ("Sangat_Baik", "Tinggi", "Rendah", 0),
    ("Sangat_Baik", "Tinggi", "Cukup", 1),
    ("Sangat_Baik", "Tinggi", "Tinggi", 2),
]

POOL = {
    "Poin langsung": ["+25", "+50", "+75"],
    "Multiplier skor": ["x1.25", "x1.5", "x1.75"],
}


# Menghitung nilai Performance
def calculate_performance(score, time_min, avg_time):
    if not 0 <= score <= 100:
        raise ValueError("Skor harus 0-100.")

    if time_min < 0:
        raise ValueError("Waktu tidak boleh negatif.")

    if avg_time <= 0:
        raise ValueError("Waktu rata-rata harus > 0.")

    return float(max(
        0.0,
        min(100.0, score * (1 - time_min / (2 * avg_time)))
    ))


# Menghitung nilai Improvement
def calculate_improvement(score, prev_score=None):
    if not 0 <= score <= 100:
        raise ValueError("Skor harus 0-100.")

    if prev_score is None:
        return 0.0

    if not 0 <= prev_score <= 100:
        raise ValueError("Skor sebelumnya harus 0-100.")

    return float(max(0.0, score - prev_score))

# contoh penggunaan, data_input = {"Performance" : x, "Improvement": x, "Attempts" : x}
# Proses fuzzifikasi
def fuzzifikasi(data_input):
    memberships = {}

    for variabel in data_input:
        memberships[variabel] = {}

        for himpunan in SETS[variabel]:
            if himpunan == "domain":
                continue

            memberships[variabel][himpunan] = fuzz.trimf(
                np.array([data_input[variabel]], dtype=float),
                SETS[variabel][himpunan]
            )[0]

    return memberships


# Proses inferensi: menghitung alpha seluruh rule
def inferensi(memberships):
    hasil_rule = []

    for i, rule in enumerate(RULES, start=1):
        performance, improvement, attempts, z = rule

        alpha = min(
            memberships["Performance"][performance],
            memberships["Improvement"][improvement],
            memberships["Attempts"][attempts]
        )

        hasil_rule.append({
            "Rule": f"R{i}",
            "Performance": performance,
            "Improvement": improvement,
            "Attempts": attempts,
            "z": z,
            "alpha (MIN)": alpha,
            "alpha x z": alpha * z
        })

    return hasil_rule


# Proses defuzzifikasi: Weighted Average
def defuzifikasi(hasil_rule):
    pembilang = sum(
        rule["alpha x z"]
        for rule in hasil_rule
    )

    penyebut = sum(
        rule["alpha (MIN)"]
        for rule in hasil_rule
    )

    if penyebut == 0:
        raise ValueError("Tidak ada aturan aktif.")

    hasil = pembilang / penyebut

    return pembilang, penyebut, hasil

# Menentukan jumlah challenge sebelum reward
def decision(w):
    jumlah = max(0, min(3, round(w)))

    if jumlah == 0:
        keterangan = "Reward diberikan langsung."
    else:
        keterangan = (
            f"Reward diberikan setelah {jumlah} challenge berikutnya."
        )

    return jumlah, keterangan


# Menggabungkan seluruh proses
def run(score, time_min, avg_time, prev_score, attempts):
    if not 1 <= attempts <= 5:
        raise ValueError("Attempts harus 1-5.")

    perf = calculate_performance(score, time_min, avg_time)
    imp = calculate_improvement(score, prev_score)

    data_input = {
        "Performance": perf,
        "Improvement": imp,
        "Attempts": attempts
    }

    memberships = fuzzifikasi(data_input)

    hasil_rule = inferensi(memberships)

    pembilang, penyebut, w = defuzifikasi(hasil_rule)

    jumlah, keterangan = decision(w)

    return {
        "score": score,
        "time_min": time_min,
        "ref_time": avg_time,
        "prev_score": prev_score,
        "perf": perf,
        "imp": imp,
        "att": attempts,
        "mem": memberships,
        "rows": hasil_rule,
        "num": pembilang,
        "den": penyebut,
        "w": w,
        "n": jumlah,
        "text": keterangan
    }

