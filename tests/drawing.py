"""Synthetic fixtures built from geometry, never used by production services."""
import cv2
import numpy as np
import pymupdf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def canvas(width=900, height=1000):
    return np.full((height, width, 3), 255, np.uint8)


def text(image, value, center, size=.7):
    candidates = [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                  Path('C:/Windows/Fonts/arial.ttf')]
    path = next((path for path in candidates if path.exists()), None)
    if path is None:
        (width, height), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        cv2.putText(image, value, (int(center[0] - width / 2), int(center[1] + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 1, cv2.LINE_AA)
        return
    font = ImageFont.truetype(str(path), round(size * 32))
    rendered = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rendered)
    draw.text(center, value, font=font, fill='black', anchor='mm')
    image[:] = cv2.cvtColor(np.asarray(rendered), cv2.COLOR_RGB2BGR)


def shape(image, kind, box, value):
    x, y, width, height = box
    if kind == "terminal":
        radius = min(20, height // 2)
        cv2.line(image, (x + radius, y), (x + width - radius, y), (0, 0, 0), 2)
        cv2.line(image, (x + radius, y + height), (x + width - radius, y + height), (0, 0, 0), 2)
        cv2.line(image, (x, y + radius), (x, y + height - radius), (0, 0, 0), 2)
        cv2.line(image, (x + width, y + radius), (x + width, y + height - radius), (0, 0, 0), 2)
        for center, start in [((x + radius, y + radius), 180), ((x + width - radius, y + radius), 270), ((x + width - radius, y + height - radius), 0), ((x + radius, y + height - radius), 90)]:
            cv2.ellipse(image, center, (radius, radius), 0, start, start + 90, (0, 0, 0), 2)
    elif kind == "process":
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), 2)
    elif kind == "decision":
        polygon = [(x + width // 2, y), (x + width, y + height // 2), (x + width // 2, y + height), (x, y + height // 2)]
        cv2.polylines(image, [np.asarray(polygon)], True, (0, 0, 0), 2)
    elif kind in ("input", "output"):
        polygon = [(x + 25, y), (x + width, y), (x + width - 25, y + height), (x, y + height)]
        cv2.polylines(image, [np.asarray(polygon)], True, (0, 0, 0), 2)
    elif kind == "iteration":
        polygon = [(x + 30, y), (x + width - 30, y), (x + width, y + height // 2), (x + width - 30, y + height), (x + 30, y + height), (x, y + height // 2)]
        cv2.polylines(image, [np.asarray(polygon)], True, (0, 0, 0), 2)
    elif kind == "connector":
        cv2.circle(image, (x + width // 2, y + height // 2), width // 2, (0, 0, 0), 2)
    text(image, value, (x + width // 2, y + height // 2), .65 if width < 75 else .7)


def arrow(image, points, head=True):
    for first, second in zip(points[:-2], points[1:-1]):
        cv2.line(image, first, second, (0, 0, 0), 2)
    a, b = points[-2:]
    if head:
        length = np.linalg.norm(np.asarray(a) - np.asarray(b))
        cv2.arrowedLine(image, a, b, (0, 0, 0), 2, tipLength=min(.4, 15 / max(1, length)))
    else:
        cv2.line(image, a, b, (0, 0, 0), 2)


def encode(image, extension=".png"):
    success, buffer = cv2.imencode(extension, image)
    assert success
    return buffer.tobytes()


def sequence(value="amount = 17", horizontal=False):
    if horizontal:
        image = canvas(1200, 400)
        shape(image, "terminal", (50, 140, 180, 75), "Start")
        shape(image, "process", (410, 140, 320, 75), value)
        shape(image, "terminal", (930, 140, 180, 75), "End")
        arrow(image, [(230, 177), (410, 177)])
        arrow(image, [(730, 177), (930, 177)])
        return image
    image = canvas(900, 850)
    specifications = [("terminal", "Start"), ("process", value), ("output", "Output amount"), ("terminal", "End")]
    for index, (kind, content) in enumerate(specifications):
        y = 60 + index * 190
        shape(image, kind, (290, y, 320, 85), content)
        if index:
            arrow(image, [(450, y - 105), (450, y)])
    return image


def branch():
    image = canvas(1200, 1100)
    shape(image, "terminal", (440, 40, 320, 80), "Start")
    shape(image, "input", (440, 220, 320, 85), "Input score:int")
    shape(image, "decision", (430, 415, 340, 160), "score > 60")
    shape(image, "output", (110, 720, 320, 85), "Output 1")
    shape(image, "output", (790, 720, 320, 85), "Output 0")
    shape(image, "terminal", (440, 980, 320, 80), "End")
    arrow(image, [(600, 120), (600, 220)])
    arrow(image, [(600, 305), (600, 415)])
    arrow(image, [(430, 495), (270, 495), (270, 720)])
    arrow(image, [(770, 495), (950, 495), (950, 720)])
    arrow(image, [(270, 805), (270, 1020), (440, 1020)])
    arrow(image, [(950, 805), (950, 1020), (760, 1020)])
    text(image, "Yes", (320, 465))
    text(image, "No", (865, 465))
    return image


def loop():
    image = canvas(1000, 1100)
    shape(image, "terminal", (340, 40, 320, 80), "Start")
    shape(image, "process", (340, 220, 320, 80), "count = 3")
    shape(image, "iteration", (325, 430, 350, 110), "WHILE count > 0")
    shape(image, "process", (340, 720, 320, 85), "count = count - 1")
    shape(image, "terminal", (60, 960, 260, 80), "End")
    arrow(image, [(500, 120), (500, 220)])
    arrow(image, [(500, 300), (500, 430)])
    arrow(image, [(500, 540), (500, 720)])
    arrow(image, [(325, 485), (190, 485), (190, 960)])
    arrow(image, [(660, 762), (850, 762), (850, 485), (675, 485)])
    text(image, "Yes", (545, 630))
    text(image, "No", (250, 455))
    return image


def independent_pdf():
    # A separately authored vector document, not a raster/sample round-trip.
    document = pymupdf.open()
    page = document.new_page(width=500, height=600)
    page.draw_oval(pymupdf.Rect(160, 30, 340, 95), color=(0, 0, 0), width=1.5)
    page.draw_rect(pymupdf.Rect(120, 235, 380, 315), color=(0, 0, 0), width=1.5)
    page.draw_oval(pymupdf.Rect(160, 480, 340, 545), color=(0, 0, 0), width=1.5)
    for box, value in [(pymupdf.Rect(175, 50, 325, 90), "Start"), (pymupdf.Rect(135, 260, 365, 305), "result = 42"), (pymupdf.Rect(180, 502, 320, 535), "End")]:
        page.insert_textbox(box, value, fontsize=16, align=1)
    for start, end in [((250, 95), (250, 235)), ((250, 315), (250, 480))]:
        page.draw_line(start, end, color=(0, 0, 0), width=1.5)
        page.draw_line((end[0] - 9, end[1] - 15), end, color=(0, 0, 0), width=1.5)
        page.draw_line((end[0] + 9, end[1] - 15), end, color=(0, 0, 0), width=1.5)
    data = document.tobytes()
    document.close()
    return data
