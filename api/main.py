from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fuzzy import run

app = FastAPI(
    title="E-Block Adaptive Variable Ratio API",
    description="Fuzzy Sugeno: menentukan jumlah challenge sebelum reward.",
    version="1.0.0",
)


class EvaluateRequest(BaseModel):
    score: float = Field(..., ge=0, le=100,
                         description="Skor saat ini (0-100)", examples=[80])
    time_min: float = Field(..., ge=0,
                            description="Waktu pengerjaan (menit)", examples=[5])
    avg_time: float = Field(..., gt=0,
                            description="Rata-rata waktu / acuan (menit)", examples=[10])
    prev_score: Optional[float] = Field(
        None, ge=0, le=100,
        description="Skor sebelumnya (0-100). Isi null jika percobaan pertama.",
        examples=[60],
    )
    attempts: int = Field(..., ge=1, le=6,
                          description="Jumlah percobaan (1-6)", examples=[1])


class EvaluateResponse(BaseModel):
    performance: float = Field(..., description="Nilai performance hasil perhitungan")
    improvement: float = Field(..., description="Nilai improvement hasil perhitungan")
    attempts: int = Field(..., description="Jumlah percobaan")
    required_challenges: int = Field(
        ..., description="Jumlah challenge yang harus diselesaikan sebelum reward (0 = langsung)"
    )
    message: str = Field(..., description="Keterangan hasil")


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Fuzzy"])
def evaluate(req: EvaluateRequest):
    try:
        hasil = run(
            req.score,
            req.time_min,
            req.avg_time,
            req.prev_score,
            req.attempts,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "performance": round(hasil["perf"], 2),
        "improvement": round(hasil["imp"], 2),
        "attempts": hasil["att"],
        "required_challenges": hasil["n"],
        "message": hasil["text"],
    }