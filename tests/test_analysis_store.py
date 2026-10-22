from dataclasses import replace
from backend.analysis_store import AnalysisStore
from backend.models import PipelineContext

def test_store_random_ids_and_get(settings):
    store=AnalysisStore(settings)
    one,_=store.put(PipelineContext(),b"image")
    two,_=store.put(PipelineContext(),b"image")
    assert one!=two and len(one)>=32
    assert store.get(one).image_png==b"image"

def test_expiry_refresh_and_removal(settings):
    now=[1000.0]
    store=AnalysisStore(replace(settings,STORE_TTL_SECONDS=10),clock=lambda:now[0])
    identifier,_=store.put(PipelineContext(),b"image")
    now[0]+=5;assert store.get(identifier)
    now[0]+=9;assert store.get(identifier)
    now[0]+=11;assert store.get(identifier) is None

def test_oldest_evicted(settings):
    store=AnalysisStore(replace(settings,STORE_MAX_ENTRIES=1))
    first,_=store.put(PipelineContext(),b"a")
    second,_=store.put(PipelineContext(),b"b")
    assert store.get(first) is None and store.get(second)

def test_byte_capacity(settings):
    store=AnalysisStore(replace(settings,STORE_MAX_BYTES=8))
    first,_=store.put(PipelineContext(),b"12345")
    second,_=store.put(PipelineContext(),b"67890")
    assert store.get(first) is None and store.get(second)

def test_delete_idempotent(settings):
    store=AnalysisStore(settings)
    identifier,_=store.put(PipelineContext(),b"x")
    assert store.delete(identifier)
    assert not store.delete(identifier)
    assert store.get(identifier) is None
