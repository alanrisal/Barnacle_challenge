"""A bounded OpenCV window with source-coordinate corner editing."""

from dataclasses import dataclass

import cv2
import numpy as np

from .regions import manual_region


@dataclass
class ViewTransform:
    scale: float
    offset: np.ndarray

    def to_image(self, point) -> np.ndarray:
        return (np.asarray(point, dtype=float) - self.offset) / self.scale

    def to_screen(self, point) -> np.ndarray:
        return np.asarray(point, dtype=float) * self.scale + self.offset

    def zoom(self, point, factor: float) -> None:
        anchor = self.to_image(point)
        self.scale = float(np.clip(self.scale * factor, 0.05, 16.0))
        self.offset = np.asarray(point, dtype=float) - anchor * self.scale


def select_corners(image: np.ndarray, initial: dict | None = None) -> dict:
    width, height = 1100, 760
    ih, iw = image.shape[:2]

    def fitted_view():
        scale = min(width / iw, (height - 60) / ih, 1.0)
        return ViewTransform(scale, np.array([(width - iw * scale) / 2,
                                              60 + (height - 60 - ih * scale) / 2]))

    view = fitted_view()
    points = [np.array(p, dtype=float) for p in initial["polygon_xy"]] if initial else []
    state = {"dirty": True, "drag_corner": None, "pan": None}
    title = "Barnacle region - Enter saves, Esc cancels"

    def mouse(event, x, y, flags, _):
        cursor = np.array([x, y], dtype=float)
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 32768:
                delta -= 65536
            view.zoom(cursor, 1.25 if delta > 0 else 0.8)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pan"] = cursor
        elif event == cv2.EVENT_RBUTTONUP:
            state["pan"] = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            source = np.clip(view.to_image(cursor), [0, 0], [iw - 1, ih - 1])
            distances = [np.linalg.norm(view.to_screen(p) - cursor) for p in points]
            if distances and min(distances) < 18:
                state["drag_corner"] = int(np.argmin(distances))
            elif len(points) < 4 and y >= 60:
                points.append(source)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag_corner"] = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["pan"] is not None:
                view.offset += cursor - state["pan"]
                state["pan"] = cursor
            if state["drag_corner"] is not None:
                points[state["drag_corner"]] = np.clip(view.to_image(cursor), [0, 0], [iw - 1, ih - 1])
        state["dirty"] = True

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    error = ""
    try:
        while True:
            if state["dirty"]:
                affine = np.array([[view.scale, 0, view.offset[0]], [0, view.scale, view.offset[1]]])
                canvas = cv2.warpAffine(image, affine, (width, height), flags=cv2.INTER_LINEAR)
                projected = np.array([view.to_screen(p) for p in points], dtype=np.int32)
                if len(projected) >= 2:
                    cv2.polylines(canvas, [projected], len(points) == 4, (0, 220, 255), 2)
                for i, point in enumerate(projected):
                    cv2.circle(canvas, tuple(point), 7, (0, 220, 255), -1)
                    cv2.putText(canvas, str(i + 1), tuple(point + [10, -10]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                canvas[:60] = (25, 25, 25)
                text = "Click 4 corners / drag points | Wheel: zoom | Right-drag: pan | F: fit | C: clear"
                cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (240, 240, 240), 1)
                cv2.putText(canvas, error or "Enter: save region   Esc: cancel", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 220, 255), 1)
                cv2.imshow(title, canvas)
                state["dirty"] = False
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Region selection cancelled; no selection saved")
            if key in (10, 13):
                try:
                    return manual_region(image, points)
                except ValueError as exc:
                    error = str(exc)
                    state["dirty"] = True
            if key in (ord("c"), ord("C")):
                points.clear()
                error = ""
                state["dirty"] = True
            if key in (ord("f"), ord("F")):
                view = fitted_view()
                state["dirty"] = True
    finally:
        cv2.destroyWindow(title)
