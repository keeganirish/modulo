"""Attach actual connector strokes; never invent reading-order connections."""
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
import math
import cv2
import numpy as np

from backend.models import Arrow, Edge, FlowGraph, Node, PipelineContext
from backend.notation import normalize_label

@dataclass
class Port:
    node: Node
    point: tuple[int, int]
    head: bool


def arrowhead_evidence(points: np.ndarray, point: tuple[int, int]) -> bool:
    """Detect a flared V/triangle near a tip followed by a narrower shaft."""
    delta = points.astype(float) - np.asarray(point)
    radius = np.linalg.norm(delta, axis=1)
    for span in (56, 100, 160):
        shaft = delta[(radius >= span * .55) & (radius <= span * 1.2)]
        if len(shaft) < 4:
            continue
        outward = np.median(shaft, axis=0)
        norm = np.linalg.norm(outward)
        if norm < 8:
            continue
        direction = outward / norm
        perpendicular = np.array([-direction[1], direction[0]])
        longitudinal, transverse = delta @ direction, delta @ perpendicular
        widths, flares = [], []
        step = max(4, span // 14)
        for start in range(0, span, step):
            selected = transverse[(longitudinal >= start) & (longitudinal < start + step) & (radius < span * 1.4)]
            widths.append(float(np.ptp(selected)) + 1 if len(selected) else 0)
            flares.append(bool(len(selected) and selected.min() < -4 and selected.max() > 4))
        boundary = max(2, int(len(widths) * .63))
        early = max((widths[i] for i in range(1, boundary) if flares[i]), default=0)
        reference = [width for width in widths[boundary + 1:] if width > 0]
        if reference:
            narrow = float(np.median(reference))
            if early > max(9, narrow * 1.8 + 3) and narrow < 18:
                return True
    return False


def contacts(points: np.ndarray, nodes: list[Node], shape: tuple[int, int], distance: int) -> list[Port]:
    ports = []
    height, width = shape
    for node in nodes:
        # Distance is measured from the actual polygon, not its enclosing rectangle.
        bx, by, bw, bh = node.bbox.to_list()
        nearby = points[(points[:, 0] >= bx - distance) & (points[:, 0] <= bx + bw + distance) &
                        (points[:, 1] >= by - distance) & (points[:, 1] <= by + bh + distance)]
        if len(nearby) == 0:
            continue
        x0, y0 = max(0, bx - distance - 3), max(0, by - distance - 3)
        x1, y1 = min(width, bx + bw + distance + 3), min(height, by + bh + distance + 3)
        exterior = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        polygon = np.asarray(node.polygon or [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], np.int32)
        cv2.fillPoly(exterior, [polygon - np.array([x0, y0])], 0)
        distances = cv2.distanceTransform(exterior, cv2.DIST_L2, 5)
        values = distances[nearby[:, 1] - y0, nearby[:, 0] - x0]
        near = nearby[values <= distance]
        if len(near) == 0:
            continue
        band = np.zeros_like(exterior)
        band[near[:, 1] - y0, near[:, 0] - x0] = 255
        count, labels = cv2.connectedComponents(band, connectivity=8)
        for component in range(1, count):
            ys, xs = np.where(labels == component)
            if len(xs) < 3:
                continue
            d = distances[ys, xs]
            if float(d.min()) > distance:
                continue
            closest = np.column_stack((xs[d <= d.min() + 1.5] + x0, ys[d <= d.min() + 1.5] + y0))
            center = closest.mean(axis=0)
            point_array = closest[np.argmin(np.sum((closest - center) ** 2, axis=1))]
            point = int(point_array[0]), int(point_array[1])
            ports.append(Port(node, point, arrowhead_evidence(points, point)))
    return sorted(ports, key=lambda port: (port.point[1], port.point[0], port.node.id))


def stroke_path(points: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    width, height = int(x1 - x0 + 1), int(y1 - y0 + 1)
    mask = np.zeros((height, width), np.uint8)
    mask[points[:, 1] - y0, points[:, 0] - x0] = 1
    sx, sy = int(start[0] - x0), int(start[1] - y0)
    tx, ty = int(end[0] - x0), int(end[1] - y0)
    begin, finish = sy * width + sx, ty * width + tx
    queue = deque([begin])
    parent = np.full(width * height, -1, np.int32)
    parent[begin] = begin
    found = False
    while queue:
        current = queue.popleft()
        if current == finish:
            found = True
            break
        cy, cx = divmod(current, width)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                index = ny * width + nx
                if mask[ny, nx] and parent[index] == -1:
                    parent[index] = current
                    queue.append(index)
    if not found:
        return []
    route = []
    current = finish
    while current != begin:
        cy, cx = divmod(current, width)
        route.append((int(cx + x0), int(cy + y0)))
        current = int(parent[current])
    route.append(start)
    route.reverse()
    simplified = cv2.approxPolyDP(np.asarray(route, np.int32).reshape(-1, 1, 2), 3, False)
    return [(int(x), int(y)) for x, y in simplified.reshape(-1, 2)]


def distance_to_path(point: tuple[float, float], path: list[tuple[int, int]]) -> float:
    p = np.asarray(point)
    distances = []
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        vector = b - a
        t = np.clip(np.dot(p - a, vector) / max(1, np.dot(vector, vector)), 0, 1)
        distances.append(float(np.linalg.norm(p - (a + t * vector))))
    return min(distances, default=math.inf)


class GraphService:
    name = "graph"

    def run(self, context: PipelineContext) -> PipelineContext:
        graph = FlowGraph(nodes=context.elements)
        context.graph = graph
        mask = context.stroke_mask
        if mask is None:
            context.diagnose("error", "MISSING_CONNECTORS", "No connector geometry is available.", "Run vision before graph construction.")
            return context
        # Diagram editors often leave a deliberate gap around YES/NO text at
        # an elbow. Apply this only to solid-color diagrams; outlined inputs
        # retain the stricter original connector evidence.
        page = context.page_image
        if page is not None:
            hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
            if float(np.mean(hsv[:, :, 1] > 25)) > .02:
                horizontal = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 81), np.uint8))
                vertical = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((81, 1), np.uint8))
                mask = cv2.max(horizontal, vertical)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = sorted(range(1, count), key=lambda index: (int(stats[index, cv2.CC_STAT_TOP]), int(stats[index, cv2.CC_STAT_LEFT])))
        candidates = []
        for index in components:
            if stats[index, cv2.CC_STAT_AREA] < 20 or max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]) < 18:
                continue
            x, y, width, height = stats[index, :4]
            ys, xs = np.where(labels[y:y + height, x:x + width] == index)
            points = np.column_stack((xs + x, ys + y))
            ports = contacts(points, graph.nodes, mask.shape, context.settings.ARROW_ATTACH_DISTANCE)
            ids = list(dict.fromkeys(port.node.id for port in ports))
            if len(ports) < 2:
                context.diagnose("error", "DANGLING_ARROW", "A connector does not reach two shape boundaries.", "Attach each arrow to its source and destination; remove stray lines.", ids)
                continue
            sources, targets = [port for port in ports if not port.head], [port for port in ports if port.head]
            if not targets:
                context.diagnose("error", "AMBIGUOUS_ARROW_DIRECTION", "A connector has no reliably visible arrowhead.", "Use a larger, visible V-shaped or filled triangular arrowhead. Direction is never inferred from layout.", ids)
                continue
            if not sources or (len(sources) > 1 and len(targets) > 1):
                context.diagnose("error", "AMBIGUOUS_CONNECTOR", "A crossing or multi-way connector cannot be resolved safely.", "Separate crossing lines and use labeled connector circles for distant paths.", ids)
                continue
            for source in sources:
                for target in targets:
                    path = stroke_path(points, source.point, target.point)
                    if not path:
                        context.diagnose("error", "DANGLING_ARROW", "A connector path is interrupted.", "Reconnect the line in the source diagram.", [source.node.id, target.node.id])
                        continue
                    candidates.append((source, target, path))
        candidates.sort(key=lambda item: (min(point[1] for point in item[2]), min(point[0] for point in item[2]), item[0].node.id, item[1].node.id))
        for index, (source, target, path) in enumerate(candidates, 1):
            edge = Edge(f"e{index}", source.node.id, target.node.id, points=path)
            graph.edges.append(edge)
            context.arrows.append(Arrow(edge.id, path, target.point, source_hint=edge.source, target_hint=edge.target))
        for word in context.edge_words:
            if not any(character.isalnum() for character in word.text):
                continue
            distances = sorted((distance_to_path(word.bbox.center, edge.points), index) for index, edge in enumerate(graph.edges))
            if not distances or distances[0][0] > max(75, word.bbox.height * 3):
                continue
            if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) < 5:
                if normalize_label(word.text):
                    context.diagnose("error", "AMBIGUOUS_EDGE_LABEL", "A branch label is equally close to multiple arrows.", "Move each Yes/No label closer to its own branch.", [graph.edges[distances[0][1]].id, graph.edges[distances[1][1]].id])
                continue
            edge = graph.edges[distances[0][1]]
            label = normalize_label(word.text) or word.text
            if edge.label:
                context.diagnose("error", "AMBIGUOUS_EDGE_LABEL", "Multiple text labels are attached to one arrow.", "Use one clearly positioned label for each branch.", [edge.id])
            else:
                edge.label = label
                if word.confidence * 100 < context.settings.OCR_MIN_CONFIDENCE:
                    context.diagnose("warning", "LOW_OCR_CONFIDENCE", "An arrow label may have been misread.", "Verify the label or export larger text.", [edge.id])
        # Circle pairs bridge an incoming physical arrow to an outgoing physical arrow.
        pairs = defaultdict(list)
        for node in graph.nodes:
            if node.type == "connector":
                pairs[node.text.strip().casefold()].append(node)
        for label in sorted(pairs):
            pair = pairs[label]
            if len(pair) != 2:
                continue  # Validation supplies the matching-pair diagnostic.
            first, second = pair
            incoming_first = any(edge.target == first.id for edge in graph.edges)
            outgoing_first = any(edge.source == first.id for edge in graph.edges)
            incoming_second = any(edge.target == second.id for edge in graph.edges)
            outgoing_second = any(edge.source == second.id for edge in graph.edges)
            if incoming_first and not outgoing_first and outgoing_second and not incoming_second:
                graph.edges.append(Edge(f"e{len(graph.edges) + 1}", first.id, second.id, structural=True))
            elif incoming_second and not outgoing_second and outgoing_first and not incoming_first:
                graph.edges.append(Edge(f"e{len(graph.edges) + 1}", second.id, first.id, structural=True))
            else:
                context.diagnose("error", "AMBIGUOUS_CONNECTOR_PAIR", "A connector pair has ambiguous entry and exit arrows.", "One circle receives flow; its matching circle sends flow onward.", [first.id, second.id])
        starts = [node.id for node in graph.nodes if node.type == "terminal" and node.text.strip().casefold() == "start"]
        graph.start_id = starts[0] if len(starts) == 1 else None
        graph.end_ids = [node.id for node in graph.nodes if node.type == "terminal" and node.text.strip().casefold() == "end"]
        return context


def run(context: PipelineContext) -> PipelineContext:
    return GraphService().run(context)
