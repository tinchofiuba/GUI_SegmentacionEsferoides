"""Geometría del lazo: una curva suave y cerrada que pasa por los puntos que marca el
usuario. Sin Qt: el canvas la dibuja y la ventana la rellena en la máscara."""

import numpy as np

MIN_POINT_GAP = 1e-3     # puntos más juntos que esto (px) se tratan como uno solo


def _dedupe(points):
    """Puntos como array (N, 2) sin repetidos consecutivos (ni el último igual al primero)."""
    kept = []
    for p in np.asarray(points, dtype=float).reshape(-1, 2):
        if not kept or np.hypot(*(p - kept[-1])) > MIN_POINT_GAP:
            kept.append(p)
    if len(kept) > 1 and np.hypot(*(kept[0] - kept[-1])) <= MIN_POINT_GAP:
        kept.pop()
    return np.array(kept).reshape(-1, 2)


def smooth_closed_curve(points, per_segment=12):
    """Curva cerrada y suave (spline de Catmull-Rom centrípeta) que pasa por TODOS los
    puntos, en el orden dado y volviendo al primero. Devuelve los puntos de la curva
    como un array (M, 2); cada punto original aparece en ella. La parametrización
    centrípeta evita lazos y picos aunque los puntos estén muy desparejos. Con menos
    de 3 puntos no hay curva cerrada: devuelve los puntos tal cual."""
    pts = _dedupe(points)
    n = len(pts)
    if n < 3:
        return pts
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.hypot(*(p1 - p0)) ** 0.5
        t2 = t1 + np.hypot(*(p2 - p1)) ** 0.5
        t3 = t2 + np.hypot(*(p3 - p2)) ** 0.5
        t = np.linspace(t1, t2, per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def polygon_area(polygon):
    """Área (shoelace) de un polígono dado como (N, 2)."""
    p = np.asarray(polygon, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
