"""kiwi-service: FastAPI sidecar that tokenizes Korean text with Kiwi.

POST /tokenize accepts a batch of texts plus an allow-list of POS tags and
returns the filtered tokens per text. GET /healthz is for container health
checks.

Long compound proper nouns (NNP) are recursively expanded by re-analyzing
them so that hybrid search gets finer-grained sub-tokens (e.g. the company
name "한국주택금융공사" is decomposed into "한국", "주택", "금융", "공사").
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from kiwipiepy import Kiwi
from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)
    filterPos: List[str] = Field(default_factory=lambda: ["NNG", "NNP", "SL", "SH", "SN"])


class TokenizeResponse(BaseModel):
    tokens: List[List[str]]


app = FastAPI(title="kiwi-service", version="1.0.0")
_kiwi = Kiwi()

_NNP_EXPAND_THRESHOLD = 15.0  # max log-prob distance from best analysis
_NNP_MIN_LEN = 3               # only try to split NNP tokens longer than this


def _expand_nnp(form: str, allow: set, depth: int = 0) -> List[str]:
    """Recursively attempt to split a compound NNP into sub-tokens.

    Kiwi recognises many Korean organisation/place names as single NNP
    entries.  For search tokenisation we want the component morphemes, so we
    re-analyse the surface form and accept any parses whose score is within
    *_NNP_EXPAND_THRESHOLD* log-prob units of the MAP parse and that yield
    more than one token in the allow-list.
    """
    if depth >= 3 or len(form) <= _NNP_MIN_LEN:
        return [form]

    results = list(_kiwi.analyze(form, top_n=10))
    best_score: float = results[0][1]

    for analysis, score in results[1:]:
        if score < best_score - _NNP_EXPAND_THRESHOLD:
            break
        candidate = [t for t in analysis if str(t.tag) in allow]
        if len(candidate) > 1:
            out: List[str] = []
            for t in candidate:
                if str(t.tag) == "NNP" and len(t.form) > _NNP_MIN_LEN:
                    out.extend(_expand_nnp(t.form, allow, depth + 1))
                else:
                    out.append(t.form)
            return out

    return [form]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    if not req.texts:
        return TokenizeResponse(tokens=[])

    allow = set(req.filterPos)
    out: List[List[str]] = []

    for analysis in _kiwi.tokenize(req.texts):
        filtered: List[str] = []
        for t in analysis:
            if str(t.tag) in allow:
                if str(t.tag) == "NNP" and len(t.form) > _NNP_MIN_LEN:
                    filtered.extend(_expand_nnp(t.form, allow))
                else:
                    filtered.append(t.form)
        out.append(filtered)

    return TokenizeResponse(tokens=out)
