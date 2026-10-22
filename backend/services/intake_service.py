"""Decode uploads into a common light-background BGR page image."""
from io import BytesIO
from pathlib import Path
import warnings
import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError
from backend.config import Settings
from backend.models import PipelineContext

class UploadError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

def inspect_upload(data: bytes, filename: str, settings: Settings) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in settings.ALLOWED_EXTENSIONS or extension not in ("png", "jpg", "jpeg", "pdf"):
        raise UploadError("UNSUPPORTED_FILE_TYPE", "Choose a PNG, JPG, or single-page PDF.", 415)
    if not data:
        raise UploadError("EMPTY_FILE", "The file is empty. Export your diagram again.")
    if len(data) > settings.max_upload_bytes:
        raise UploadError("FILE_TOO_LARGE", f"Choose a file no larger than {settings.MAX_UPLOAD_MB} MB.", 413)
    expected = "pdf" if data.startswith(b"%PDF-") else "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg" if data.startswith(b"\xff\xd8\xff") else None
    claimed = "jpg" if extension == "jpeg" else extension
    if expected != claimed:
        raise UploadError("CONTENT_TYPE_MISMATCH", "The contents do not match the filename. Export a genuine PNG, JPG, or PDF.", 415)
    return claimed

def decode_page(data: bytes, kind: str, settings: Settings) -> np.ndarray:
    if kind == "pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if document.needs_pass:
                    raise UploadError("ENCRYPTED_PDF", "Export a PDF without password protection.")
                if document.page_count != 1:
                    raise UploadError("MULTIPAGE_PDF", "Only one page is supported. Export just the flowchart page.")
                page = document[0]
                scale = settings.PDF_RENDER_DPI / 72
                if page.rect.width * page.rect.height * scale * scale > settings.MAX_IMAGE_PIXELS:
                    raise UploadError("IMAGE_TOO_LARGE", "This PDF page is too large. Use a smaller page size.", 413)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
                rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except UploadError:
            raise
        except (RuntimeError, ValueError) as error:
            raise UploadError("INVALID_PDF", "This PDF cannot be opened. Export it again.") from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > settings.MAX_IMAGE_PIXELS:
                    raise UploadError("IMAGE_TOO_LARGE", "The image has too many pixels. Export it at a lower resolution.", 413)
                if getattr(image, "n_frames", 1) != 1:
                    raise UploadError("MULTIFRAME_IMAGE", "Use a still, single-frame image.")
                if image.width < 80 or image.height < 80:
                    raise UploadError("IMAGE_TOO_SMALL", "This image is too small to read. Export a larger diagram.")
                image = ImageOps.exif_transpose(image).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                background.alpha_composite(image)
                return cv2.cvtColor(np.asarray(background.convert("RGB")), cv2.COLOR_RGB2BGR)
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise UploadError("INVALID_IMAGE", "This image is damaged or unsafe to decode. Export it again.") from error

class IntakeService:
    name = "intake"
    def run(self, context: PipelineContext) -> PipelineContext:
        settings = context.settings or Settings()
        context.settings = settings
        try:
            kind = inspect_upload(context.uploaded_bytes, context.filename, settings)
            page = decode_page(context.uploaded_bytes, kind, settings)
            height, width = page.shape[:2]
            scale = min(settings.WORKING_IMAGE_WIDTH / width, settings.MAX_WORKING_HEIGHT / height)
            dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
            if dimensions[0] * dimensions[1] > settings.MAX_IMAGE_PIXELS:
                raise UploadError("IMAGE_TOO_LARGE", "The normalized image is too large. Use a less elongated page.", 413)
            context.page_image = cv2.resize(page, dimensions, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            context.grayscale = cv2.cvtColor(context.page_image, cv2.COLOR_BGR2GRAY)
            _, context.binary = cv2.threshold(context.grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if np.count_nonzero(context.binary) / context.binary.size > .45:
                context.diagnose("error", "UNSUPPORTED_BACKGROUND", "The diagram does not have dark lines on a light background.", "Export a clean diagram on white; photographs are not supported.")
        except UploadError as error:
            context.diagnose("error", error.code, error.message, "Choose a new export of the original digitally drawn diagram.")
        return context

def run(context: PipelineContext) -> PipelineContext:
    return IntakeService().run(context)
