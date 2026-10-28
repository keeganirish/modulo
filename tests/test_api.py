from io import BytesIO
from dataclasses import replace
import pytest
from backend.app import create_app
from tests.drawing import encode, sequence
from tests.test_intake import SVG_FLOWCHART

@pytest.fixture
def client(settings):
    app=create_app(settings); app.config["TESTING"]=True
    return app.test_client()

def test_frontend_routes(client):
    assert client.get("/").status_code==200
    assert b"Bring your own flowchart" in client.get("/").data
    assert client.get("/js/app.js").status_code==200
    assert client.get("/manus-routes.json").json["routes"]==[{"path":"/","title":"Modulo"}]

@pytest.mark.parametrize("data,expected", [({},"FILE_REQUIRED"),({"file":(BytesIO(b"text"),"chart.png")},"CONTENT_TYPE_MISMATCH"),({"file":(BytesIO(b""),"chart.png")},"EMPTY_FILE")])
def test_bad_uploads(client,data,expected):
    response=client.post("/api/analyze",data=data)
    assert response.status_code>=400 and response.json["error"]==expected

def test_unknown_analysis(client):
    assert client.get("/api/analysis/unknown/image").status_code==404
    assert client.post("/api/generate",json={"analysis_id":"unknown"}).status_code==404

def test_client_cannot_supply_analysis_truth(client):
    response=client.post("/api/generate",json={"analysis_id":"x","ir":{"malicious":True}})
    assert response.status_code==400

@pytest.mark.integration
def test_real_upload_review_and_reset(client):
    response=client.post("/api/analyze",data={"file":(BytesIO(encode(sequence("amount = 31"))),"new-upload.png")})
    assert response.status_code==200
    payload=response.json
    assert payload["status"]=="ready",payload
    assert payload["generation_available"] is True
    assert payload["nodes"][1]["text"]=="amount = 31"
    identifier=payload["analysis_id"]
    image=client.get(payload["page"]["image_url"])
    assert image.status_code==200 and image.mimetype=="image/png"
    assert image.headers["Cache-Control"]=="no-store"
    assert client.post("/api/generate",json={"analysis_id":identifier}).status_code==422
    assert client.post("/api/approve",json={"analysis_id":identifier}).status_code==200
    generated=client.post("/api/generate",json={"analysis_id":identifier})
    assert generated.status_code==200 and generated.json["status"]=="done"
    assert "def main():" in "\n".join(line["text"] for line in generated.json["code"])
    assert client.delete(f"/api/analysis/{identifier}").status_code==204
    assert client.get(payload["page"]["image_url"]).status_code==404

def test_oversized_request(settings):
    client=create_app(replace(settings,MAX_UPLOAD_MB=1)).test_client()
    response=client.post("/api/analyze",data={"file":(BytesIO(b"x"*(3*1024*1024)),"large.png")})
    assert response.status_code==413

def test_cross_origin_request_rejected(client):
    response=client.post("/api/analyze",headers={"Origin":"https://other.example"})
    assert response.status_code==403

def test_bad_json_request(client):
    assert client.post("/api/generate",data="{broken",content_type="application/json").status_code==400


def test_svg_upload_is_accepted(client):
    response = client.post("/api/analyze", data={"file": (BytesIO(SVG_FLOWCHART), "diagram.svg")})
    assert response.status_code == 200, response.json
    assert response.json["page"]["width"] > 0
