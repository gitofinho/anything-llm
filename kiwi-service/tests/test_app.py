from fastapi.testclient import TestClient
from kiwi_service.app import app

client = TestClient(app)


def test_healthz_ok():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_tokenize_basic_korean():
    r = client.post(
        "/tokenize",
        json={"texts": ["한국주택금융공사 채용계획"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "tokens" in body
    assert len(body["tokens"]) == 1
    tokens = body["tokens"][0]
    # Kiwi should split the input into at least 4 of these expected nominal tokens.
    assert sum(1 for t in tokens if t in {"한국", "주택", "금융", "공사", "채용", "계획"}) >= 4


def test_tokenize_filters_unwanted_pos():
    r = client.post(
        "/tokenize",
        json={"texts": ["나는 학교에 갔다"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    tokens = r.json()["tokens"][0]
    assert "학교" in tokens
    assert "나" not in tokens


def test_tokenize_empty_batch():
    r = client.post("/tokenize", json={"texts": [], "filterPos": ["NNG"]})
    assert r.status_code == 200
    assert r.json() == {"tokens": []}


def test_tokenize_non_korean_passes_through_empty():
    r = client.post(
        "/tokenize",
        json={"texts": ["hello world 123"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    assert r.json()["tokens"][0] == []
