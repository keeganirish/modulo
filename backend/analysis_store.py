"""Temporary session state. No disk writes, background threads, or persistence."""
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
import secrets
import time
from backend.config import Settings
from backend.models import PipelineContext

@dataclass
class Entry:
    context: PipelineContext
    expires: float
    image_png: bytes
    size: int

class AnalysisStore:
    def __init__(self, settings: Settings | None = None, clock=time.time):
        self.settings = settings or Settings()
        self.clock = clock
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.lock = RLock()

    def _cleanup(self, now: float) -> None:
        for key in list(self.entries):
            if self.entries[key].expires <= now:
                del self.entries[key]

    @staticmethod
    def expiry(entry: Entry) -> str:
        return datetime.fromtimestamp(entry.expires, timezone.utc).isoformat().replace("+00:00", "Z")

    def put(self, context: PipelineContext, image_png: bytes) -> tuple[str, str]:
        size = len(image_png) + sum(array.nbytes for array in (context.page_image, context.grayscale, context.binary, context.stroke_mask) if array is not None) + len(context.uploaded_bytes)
        if size > self.settings.STORE_MAX_BYTES:
            raise ValueError("Analysis exceeds the in-memory capacity")
        with self.lock:
            now = self.clock()
            self._cleanup(now)
            while self.entries and (len(self.entries) >= self.settings.STORE_MAX_ENTRIES or sum(entry.size for entry in self.entries.values()) + size > self.settings.STORE_MAX_BYTES):
                self.entries.popitem(last=False)
            identifier = secrets.token_urlsafe(24)
            entry = Entry(context, now + self.settings.STORE_TTL_SECONDS, image_png, size)
            self.entries[identifier] = entry
            return identifier, self.expiry(entry)

    def get(self, identifier: str) -> Entry | None:
        with self.lock:
            now = self.clock()
            self._cleanup(now)
            entry = self.entries.get(identifier)
            if entry is not None:
                entry.expires = now + self.settings.STORE_TTL_SECONDS
            return entry

    def delete(self, identifier: str) -> bool:
        with self.lock:
            self._cleanup(self.clock())
            return self.entries.pop(identifier, None) is not None
