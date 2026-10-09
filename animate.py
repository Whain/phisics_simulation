# -*- coding: utf-8 -*-
"""
Анимация бросков треугольного футляра (matplotlib, 3D).

  * animate_throw  — один бросок подробно: 3D-вид с камерой, следящей за футляром,
                     график высоты и график «какая грань смотрит вниз»;
                     момент первого касания замедлен и выделен.
  * animate_series — несколько бросков подряд со «счётчиком», как в реальном
                     эксперименте: частоты считаются по фактически выпавшим граням.

fx=True — «боевой» режим: неоновая сцена, лазерное сопровождение, прицел на точке
падения, взрыв с искрами и ударными волнами в момент первого касания, тряска камеры,
огненный след и HUD. Это только визуальные эффекты: физика и исходы те же самые.
fx=False — спокойная светлая версия (удобно для отчёта).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from case_sim import CaseModel, FACES, quat_to_mat
from experiment import FACE_COLORS, INK, INK_2, GRID, SURFACE, PHYS_COUNTS

THEMES = {
    "light": dict(bg=SURFACE, panel=SURFACE, floor="#efede7", floor_line="#d9d6cd", ink=INK, ink2=INK_2,
                  grid=GRID, edge="#2b2b2b", edge_w=0.9, cap="#a19f97", faces=FACE_COLORS, shadow_alpha=0.28),
    # тёмная тема: цвета граней — те же оттенки, подобранные для тёмного фона
    "neon": dict(bg="#07080f", panel="#0d0f1c", floor="#130a28", floor_line="#ff2bd6", ink="#f4f4f8",
                 ink2="#a9abc3", grid="#262a40", edge="#9ff6ff", edge_w=1.2, cap="#5b5f7a",
                 faces={1: "#3987e5", 2: "#d95926", 3: "#199e70"}, shadow_alpha=0.55),
}
LASER_COLORS = ("#ff3b3b", "#39ff14", "#00e5ff")
HOT = np.array([[1.0, 0.95, 0.69], [1.0, 0.82, 0.25], [1.0, 0.55, 0.10], [1.0, 0.24, 0.10]])
BOOM_WORDS = ("БАБАХ!", "БУМ!", "БДЫЩ!", "КРАШ!", "ТЫДЫЩ!")


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _hull_2d(pts):
    """Выпуклая оболочка точек на плоскости (монотонная цепь Эндрю)."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _circle(c, r, z=2e-4, n=48):
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)])


# ---------------------------------------------------------------------------
# Спецэффекты
# ---------------------------------------------------------------------------
class Fx:
    """Лазеры, прицел, взрыв, ударные волны, вспышка, огненный след, HUD, звёзды."""

    def __init__(self, fig, ax, th, seed=0):
        self.fig, self.ax, self.th = fig, ax, th
        self.rng = np.random.default_rng(seed)
        glow = ((10, 0.08), (5, 0.22), (1.6, 0.95))
        self.beams = [[ax.plot([], [], [], color=c, lw=w, alpha=a, zorder=4, solid_capstyle="round")[0]
                       for w, a in glow] for c in LASER_COLORS]
        self.emit, = ax.plot([], [], [], "o", ms=6, color="#ffffff", mec="#ff2bd6", mew=1.5, ls="none", zorder=4)
        self.reticle = Line3DCollection([np.zeros((2, 3))], colors="#ff3b3b", linewidths=1.6, zorder=3)
        self.rings = Line3DCollection([np.zeros((2, 3))], linewidths=2.2, zorder=3)
        ax.add_collection3d(self.reticle)
        ax.add_collection3d(self.rings)
        self.comet = ax.scatter([0], [0], [0], s=[0], depthshade=False, zorder=4, linewidths=0)
        self.fire = ax.scatter([0], [0], [0], s=[0], depthshade=False, zorder=6, linewidths=0)
        self.sparks = ax.scatter([0], [0], [0], s=[0], depthshade=False, zorder=7, linewidths=0)
        self.boom = ax.text2D(0.5, 0.80, "", transform=ax.transAxes, ha="center", va="center", color="#ffe14d",
                              fontweight="bold", rotation=-7, zorder=10)
        self.boom.set_path_effects([pe.withStroke(linewidth=8, foreground="#ff2d2d"), pe.Normal()])
        self.hud = ax.text2D(0.02, 0.03, "", transform=ax.transAxes, ha="left", va="bottom", color="#39ff14",
                             family="monospace", fontsize=10, zorder=10)
        self.glow = ax.scatter([0], [0], [0], s=[0], depthshade=False, zorder=7, linewidths=0)
        pos = ax.get_position()
        self.flash = Rectangle((pos.x0, pos.y0), pos.width, pos.height, transform=fig.transFigure,
                               color="#fff2b0", alpha=0.0, zorder=20)
        fig.add_artist(self.flash)
        # звёздное небо — на отдельных осях под 3D-сценой
        self.sky = fig.add_axes(pos, zorder=-1)
        self.sky.set_axis_off()
        self.sky.set_xlim(0, 1); self.sky.set_ylim(0, 1)
        n = 140
        self.star_xy = self.rng.random((n, 2))
        self.star_s = self.rng.uniform(0.5, 6, n)
        self.star_ph = self.rng.uniform(0, 2 * np.pi, n)
        self.stars = self.sky.scatter(self.star_xy[:, 0], self.star_xy[:, 1], s=self.star_s, c="#cfd6ff",
                                      alpha=0.8, linewidths=0)
        ax.set_facecolor((0, 0, 0, 0))
        self.new_throw(None)

    def new_throw(self, target):
        """Новый бросок: точка падения (для прицела и взрыва) и случайные частицы взрыва."""
        rng = self.rng
        self.target = None if target is None else np.asarray(target, float)
        self.word = BOOM_WORDS[rng.integers(len(BOOM_WORDS))]
        n = 170
        az = rng.uniform(0, 2 * np.pi, n)
        el = rng.uniform(np.radians(15), np.radians(80), n)
        sp = rng.uniform(0.15, 0.75, n)
        self.sp_v = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]) * sp[:, None]
        self.sp_life = rng.uniform(0.5, 1.4, n)
        self.sp_col = HOT[rng.integers(0, len(HOT), n)]
        nf = 55
        d = rng.normal(size=(nf, 3))
        d[:, 2] = np.abs(d[:, 2]) * 1.4
        self.fi_v = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0.02, 0.12, nf)[:, None]
        self.fi_s = rng.uniform(0.6, 1.4, nf)

    def shake(self, age):
        """Тряска камеры после удара: (Δelev, Δazim) в градусах."""
        if age is None or age < 0 or age > 0.6:
            return 0.0, 0.0
        amp = 4.0 * np.exp(-age / 0.15)
        return amp * np.sin(age * 97.0), amp * np.cos(age * 73.0)

    def update(self, frame_idx, x, H, flight, age, trail, hud_lines, face_color):
        ax = self.ax
        # звёзды мерцают
        self.stars.set_sizes(self.star_s * (0.65 + 0.35 * np.sin(self.star_ph + 0.35 * frame_idx)))

        # --- лазерное сопровождение и прицел (пока футляр в полёте) ---
        tgt = self.target
        if flight and tgt is not None:
            for k, layers in enumerate(self.beams):
                a = np.radians(90 + 120 * k + 0.8 * frame_idx)
                e = tgt + np.array([0.24 * np.cos(a), 0.24 * np.sin(a), 0.0])
                for ln in layers:
                    ln.set_data_3d([e[0], x[0]], [e[1], x[1]], [0.002, x[2]])
            em = np.array([tgt + [0.24 * np.cos(np.radians(90 + 120 * k + 0.8 * frame_idx)),
                                  0.24 * np.sin(np.radians(90 + 120 * k + 0.8 * frame_idx)), 0.002] for k in range(3)])
            self.emit.set_data_3d(em[:, 0], em[:, 1], em[:, 2])
            r = 0.05 * (1 + 0.15 * np.sin(frame_idx * 0.5))
            rot = np.radians(3 * frame_idx)
            segs = [_circle(tgt, r), _circle(tgt, 0.45 * r)]
            for k in range(4):
                a = rot + k * np.pi / 2
                d = np.array([np.cos(a), np.sin(a), 0])
                segs.append(np.array([tgt + 0.6 * r * d, tgt + 1.5 * r * d]) + [0, 0, 2e-4])
            self.reticle.set_segments(segs)
        else:
            for layers in self.beams:
                for ln in layers:
                    ln.set_data_3d([], [], [])
            self.emit.set_data_3d([], [], [])
            self.reticle.set_segments([])

        # --- огненный след центра масс ---
        if trail is not None and len(trail) > 1 and flight:
            m = len(trail)
            u = np.linspace(0, 1, m)
            col = np.column_stack([np.ones(m), 0.25 + 0.7 * u, 0.1 + 0.5 * u ** 3, 0.1 + 0.8 * u])
            self.comet._offsets3d = (trail[:, 0], trail[:, 1], trail[:, 2])
            self.comet.set_sizes(4 + 60 * u ** 2)
            self.comet.set_facecolors(col)
        else:
            self.comet._offsets3d = ([], [], [])

        # --- взрыв ---
        if age is not None and age >= 0 and tgt is not None:
            c = tgt
            # искры: баллистика с «мультяшной» гравитацией, отскок от пола
            t = np.minimum(age, self.sp_life)
            p = c + self.sp_v * t[:, None] + np.array([0, 0, -0.6]) * (t[:, None] ** 2) / 2
            p[:, 2] = np.abs(p[:, 2])
            alive = age < self.sp_life
            fade = np.clip(1 - age / self.sp_life, 0, 1)
            col = np.column_stack([self.sp_col, fade])
            self.sparks._offsets3d = (p[alive, 0], p[alive, 1], p[alive, 2])
            self.sparks.set_sizes((4 + 26 * fade[alive]) * 1.0)
            self.sparks.set_facecolors(col[alive])
            # огненный шар → дым (полупрозрачный, чтобы футляр было видно)
            if age < 1.4:
                q = c + self.fi_v * (age ** 0.6) + np.array([0, 0, 0.05 * age])
                s = self.fi_s * (150 + 1600 * age ** 0.7) * (0.42 / H) ** 2
                w = min(1.0, age / 0.4)
                base = (1 - w) * np.array([1.0, 0.72, 0.18]) + w * np.array([0.30, 0.28, 0.36])
                base = 0.8 * base + 0.2 * _rgb(face_color)
                a = (0.65 - 0.35 * w) * max(0.0, 1 - age / 1.4)
                self.fire._offsets3d = (q[:, 0], q[:, 1], q[:, 2])
                self.fire.set_sizes(s)
                self.fire.set_facecolors(np.tile(np.append(base, a), (len(q), 1)))
            else:
                self.fire._offsets3d = ([], [], [])
            # светящийся шар в точке удара
            if age < 0.8:
                k = np.exp(-age / 0.25)
                self.glow._offsets3d = ([c[0]] * 3, [c[1]] * 3, [c[2] + 0.005] * 3)
                self.glow.set_sizes(np.array([9000, 3500, 900]) * (0.5 + 0.5 * k) * (0.42 / H) ** 2)
                self.glow.set_facecolors([[1, 0.85, 0.4, 0.12 * k], [1, 0.9, 0.6, 0.3 * k], [1, 1, 0.92, 0.9 * k]])
            else:
                self.glow._offsets3d = ([], [], [])
            # ударные волны по полу
            segs, cols = [], []
            for delay, color in ((0.0, "#ffe14d"), (0.10, "#ff5a1f"), (0.22, "#00e5ff")):
                a_ = age - delay
                if 0 <= a_ < 0.9:
                    segs.append(_circle(c, 0.02 + 0.32 * a_ ** 0.7))
                    cols.append(np.append(_rgb(color), max(0.0, 1 - a_ / 0.9)))
            self.rings.set_segments(segs if segs else [])
            if cols:
                self.rings.set_color(cols)
            # вспышка и надпись
            self.flash.set_alpha(float(0.22 * np.exp(-age / 0.06)))
            pop = min(1.0, age / 0.10)
            size = 14 + 50 * pop * (1.15 - 0.15 * min(1.0, age / 0.6))
            self.boom.set_text(self.word if age < 1.3 else "")
            self.boom.set_fontsize(size)
            self.boom.set_alpha(float(np.clip(1.6 - age / 0.8, 0, 1)))
        else:
            self.sparks._offsets3d = ([], [], [])
            self.fire._offsets3d = ([], [], [])
            self.glow._offsets3d = ([], [], [])
            self.rings.set_segments([])
            self.flash.set_alpha(0.0)
            self.boom.set_text("")

        self.hud.set_text("\n".join(hud_lines))


# ---------------------------------------------------------------------------
# 3D-сцена: пол, тень, футляр с цифрами на гранях, след центра масс
# ---------------------------------------------------------------------------
class Scene3D:
    def __init__(self, ax, model: CaseModel, base_size=0.42, elev=22, azim=-58, spin=0.12, zoom=1.35,
                 theme="light", fx=False, fig=None, seed=0):
        self.ax, self.model = ax, model
        self.th = THEMES[theme]
        th = self.th
        self.base_size = base_size          # минимальный размер видимой области, м
        self.elev, self.azim0, self.spin = elev, azim, spin
        self.center = None
        ax.computed_zorder = False          # порядок: пол → тень → след → футляр → цифры
        ax.set_proj_type("persp", focal_length=1.6)
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_axis_off()
        ax.set_facecolor(th["bg"])

        self.floor = Poly3DCollection([np.zeros((4, 3))], facecolor=th["floor"], edgecolor="none", zorder=0)
        lw, alpha = (1.0, 0.75) if fx else (0.8, 1.0)
        self.grid = Line3DCollection([np.zeros((2, 3))], colors=th["floor_line"], linewidths=lw, alpha=alpha, zorder=1)
        self.grid_glow = Line3DCollection([np.zeros((2, 3))], colors=th["floor_line"], linewidths=4,
                                          alpha=0.12 if fx else 0.0, zorder=1)
        self.shadow = Poly3DCollection([np.zeros((3, 3))], facecolor="#000000", alpha=0.18,
                                       edgecolor="none", zorder=2)
        self.ring = Line3DCollection([np.zeros((2, 3))], colors=th["ink"], linewidths=1.5, zorder=3)
        self.trail, = ax.plot([], [], [], color=th["ink2"], lw=1, ls=(0, (2, 2)), alpha=0.6, zorder=3)
        self.plumb, = ax.plot([], [], [], color=th["ink2"], lw=0.8, alpha=0.5, zorder=3)
        self.prism = Poly3DCollection([np.zeros((3, 3))] * 5, edgecolor=th["edge"], linewidth=th["edge_w"], zorder=5)
        self.contact_pt, = ax.plot([], [], [], marker="*", ms=14, color=th["ink"], mec=th["bg"], mew=1.2,
                                   ls="none", zorder=6)
        for art in (self.floor, self.grid_glow, self.grid, self.shadow, self.ring, self.prism):
            ax.add_collection3d(art)
        stroke = [pe.withStroke(linewidth=2.5, foreground="#1b1b1b")]
        self.labels = [ax.text(0, 0, 0, str(f), color="white", fontsize=14, fontweight="bold",
                               ha="center", va="center", zorder=8, path_effects=stroke) for f in FACES]
        self.h_label = ax.text2D(0.02, 0.04, "", transform=ax.transAxes, color=th["ink2"], fontsize=10)
        self.fx = Fx(fig or ax.figure, ax, th, seed=seed) if fx else None

    def reset_camera(self, target=None):
        self.center = None
        if self.fx is not None:
            self.fx.new_throw(target)

    def _view_dir(self, elev, azim):
        e, a = np.radians(elev), np.radians(azim)
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def update(self, x, q, frame_idx=0, trail=None, contact=None, ring=None, fx_age=None, hud=None):
        model, ax, th = self.model, self.ax, self.th
        d_el, d_az = self.fx.shake(fx_age) if self.fx is not None else (0.0, 0.0)
        elev, azim = self.elev + d_el, self.azim0 + self.spin * frame_idx + d_az
        ax.view_init(elev=elev, azim=azim)
        view = self._view_dir(elev, azim)
        light = view + np.array([0.0, 0.0, 1.2])
        light /= np.linalg.norm(light)

        # --- камера: следит за футляром по горизонтали, пол всегда в кадре ---
        H = max(self.base_size, x[2] + 0.6 * self.base_size)
        target = np.array([x[0], x[1]])
        self.center = target if self.center is None else self.center + 0.25 * (target - self.center)
        cx, cy = self.center
        ax.set_xlim(cx - H / 2, cx + H / 2)
        ax.set_ylim(cy - H / 2, cy + H / 2)
        ax.set_zlim(0, H)

        # --- пол с сеткой 5 см ---
        x0, x1, y0, y1 = cx - H / 2, cx + H / 2, cy - H / 2, cy + H / 2
        self.floor.set_verts([np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])])
        step = 0.05 if H < 0.8 else 0.10
        segs = [[(gx, y0, 0), (gx, y1, 0)] for gx in np.arange(np.ceil(x0 / step), np.floor(x1 / step) + 1) * step]
        segs += [[(x0, gy, 0), (x1, gy, 0)] for gy in np.arange(np.ceil(y0 / step), np.floor(y1 / step) + 1) * step]
        self.grid.set_segments(segs)
        self.grid_glow.set_segments(segs)

        # --- футляр ---
        V = model.world_vertices(x, q)
        R = quat_to_mat(np.asarray(q, float))
        normals = list(model.world_normals(q)) + [-R[:, 0], R[:, 0]]
        polys, colors = [], []
        for k, idx in enumerate(model.body.face_polys):
            polys.append(V[idx])
            base = _rgb(th["faces"][k + 1]) if k < 3 else _rgb(th["cap"])
            lum = 0.55 + 0.45 * max(0.0, float(normals[k] @ light))
            colors.append(np.clip(base * lum + 0.08 * (1 - lum), 0, 1))
        self.prism.set_verts(polys)
        self.prism.set_facecolor(colors)

        fs = float(np.clip(60 * 0.16 / H, 7, 16))
        for k, lab in enumerate(self.labels):
            n = normals[k]
            c = V[model.body.face_polys[k]].mean(0) + 0.004 * n
            lab.set_position_3d(c)
            lab.set_fontsize(fs)
            lab.set_visible(bool(n @ view > 0.3))

        # --- тень (свет сверху) и отвес от центра масс к полу ---
        hull = _hull_2d(V[:, :2])
        self.shadow.set_verts([np.column_stack([hull, np.full(len(hull), 1e-4)])])
        self.shadow.set_alpha(float(np.clip(th["shadow_alpha"] - 0.12 * x[2], 0.08, th["shadow_alpha"])))
        self.plumb.set_data_3d([x[0], x[0]], [x[1], x[1]], [0, max(0.0, V[:, 2].min())])
        self.h_label.set_text("" if self.fx is not None else f"высота центра масс: {x[2]:.2f} м")

        if trail is not None:   # след центра масс — только внутри видимой области
            trail = trail[(trail[:, 2] <= H) & (np.abs(trail[:, 0] - cx) <= H / 2) & (np.abs(trail[:, 1] - cy) <= H / 2)]
        if trail is not None and len(trail) > 1 and self.fx is None:
            self.trail.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        else:
            self.trail.set_data_3d([], [], [])
        if contact is not None:
            self.contact_pt.set_data_3d([contact[0]], [contact[1]], [max(contact[2], 0.0)])
        else:
            self.contact_pt.set_data_3d([], [], [])
        if ring is not None and self.fx is None:   # «волна» от удара: (центр, радиус, прозрачность)
            (rx, ry), rad, alpha = ring
            self.ring.set_segments([_circle((rx, ry), rad)])
            self.ring.set_alpha(alpha)
        else:
            self.ring.set_segments([])

        if self.fx is not None:
            flight = contact is None
            face = hud.get("face", 1) if hud else 1
            self.fx.update(frame_idx, x, H, flight, fx_age, trail, (hud or {}).get("lines", []),
                           th["faces"][face])


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------
def frame_schedule(t_end, t_first, fps=30, speed=0.35, slow=0.06, slow_window=0.05, freeze=0.8):
    """
    Список моментов модельного времени для кадров. speed — скорость воспроизведения
    (1 = реальное время), около первого касания — замедление до slow,
    в сам момент касания — стоп-кадр длительностью freeze секунд.
    """
    times, t, frozen = [], 0.0, False
    while t <= t_end:
        times.append(t)
        if not frozen and t >= t_first:
            times[-1] = t_first
            times += [t_first] * int(freeze * fps)
            frozen = True
        s = slow if abs(t - t_first) < slow_window else speed
        t += s / fps
    times.append(t_end)
    return np.array(times)


def _style_2d(ax, th):
    ax.set_facecolor(th["panel"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["ink2"])
    ax.tick_params(colors=th["ink2"])
    ax.xaxis.label.set_color(th["ink"])
    ax.yaxis.label.set_color(th["ink"])
    ax.title.set_color(th["ink"])
    ax.grid(True, color=th["grid"], lw=0.8)


def _kinematics(traj):
    """Скорость центра масс и модуль угловой скорости по записанной траектории."""
    t, X, Q = traj["t"], traj["x"], traj["q"]
    dt = np.maximum(np.diff(t), 1e-9)
    v = np.linalg.norm(np.diff(X, axis=0), axis=1) / dt
    dots = np.abs(np.sum(Q[1:] * Q[:-1], axis=1)).clip(0, 1)
    w = 2 * np.arccos(dots) / dt
    return np.append(v, 0.0), np.append(w, 0.0)


def _hud_lines(traj, i, speed, omega, stage):
    return [f"▶ {stage}",
            f"  высота   {traj['x'][i, 2]:5.2f} м",
            f"  скорость {speed[i]:5.2f} м/с",
            f"  вращение {omega[i]:5.1f} рад/с"]


# ---------------------------------------------------------------------------
# Один бросок — подробно
# ---------------------------------------------------------------------------
def animate_throw(model: CaseModel, traj=None, seed=11, fps=30, speed=0.4, slow=0.05, fx=True):
    traj = traj or model.trajectory(seed=seed, fps=1000, t_after_rest=0.4)
    th = THEMES["neon" if fx else "light"]
    t, X, Q = traj["t"], traj["x"], traj["q"]
    dt_rec = t[1] - t[0]
    tf, face = traj["t_first"], traj["first_face"]
    i_first = int(round(tf / dt_rec))
    V_first = model.world_vertices(X[i_first], Q[i_first])
    contact = V_first[traj["first_vertex"]]
    v_cm, w_mag = _kinematics(traj)

    # «насколько грань смотрит вниз»: cos угла между внешней нормалью и направлением вниз
    down = np.array([-model.world_normals(q)[:, 2] for q in Q])
    zmin = np.array([model.world_vertices(x, q)[:, 2].min() for x, q in zip(X, Q)])

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(th["bg"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], left=0.01, right=0.97, top=0.86, bottom=0.08,
                          hspace=0.45, wspace=0.12)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axz = fig.add_subplot(gs[0, 1])
    axd = fig.add_subplot(gs[1, 1])
    scene = Scene3D(ax3, model, theme="neon" if fx else "light", fx=fx, fig=fig, seed=seed)
    scene.reset_camera(contact)

    fig.text(0.015, 0.95, "Бросок треугольного футляра", fontsize=17, fontweight="bold", color=th["ink"])
    sub = fig.text(0.015, 0.905, "", fontsize=11, color=th["ink2"])
    banner = fig.text(0.015, 0.875, "", fontsize=14, fontweight="bold", color=th["ink"], va="top", zorder=30)

    # график высоты
    _style_2d(axz, th)
    axz.plot(t, X[:, 2], color=th["ink"], lw=2, label="центр масс")
    axz.plot(t, zmin, color=th["ink2"], lw=1.2, ls="--", label="нижняя точка футляра")
    axz.axvline(tf, color=th["ink2"], lw=1, ls=":")
    axz.text(tf, X[:, 2].max() * 1.02, " первое касание", color=th["ink2"], fontsize=9, va="bottom")
    axz.set_xlim(0, t[-1])
    axz.set_ylim(-0.02, X[:, 2].max() * 1.15)
    axz.set_xlabel("время, с")
    axz.set_ylabel("высота, м")
    axz.set_title("Высота над полом", loc="left", fontsize=12, fontweight="bold")
    axz.legend(frameon=False, fontsize=9, loc="center right", labelcolor=th["ink"])
    dot_z, = axz.plot([], [], "o", color=th["ink"], ms=7, mec=th["panel"], mew=1.5)

    # график «какая грань смотрит вниз»
    _style_2d(axd, th)
    for k, f in enumerate(FACES):
        axd.plot(t, down[:, k], color=th["faces"][f], lw=2, label=f"грань {f}")
    axd.axvline(tf, color=th["ink2"], lw=1, ls=":")
    axd.plot([tf], [down[i_first, face - 1]], "o", color=th["faces"][face], ms=9, mec=th["ink"], mew=1.2)
    axd.set_xlim(0, t[-1])
    axd.set_ylim(-1.05, 1.5)
    axd.set_xlabel("время, с")
    axd.set_ylabel("cos(нормаль, вниз)")
    axd.set_title("Какая грань смотрит вниз (1 — точно вниз)", loc="left", fontsize=12, fontweight="bold")
    axd.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", labelcolor=th["ink"])
    cur_d = axd.axvline(0, color=th["ink"], lw=1)

    times = frame_schedule(t[-1], tf, fps=fps, speed=speed, slow=slow)
    freeze_frames = np.flatnonzero(np.isclose(times, tf))
    j_impact = int(freeze_frames[0])
    n_frames = len(times)
    rest_face = traj["rest_face"]

    def draw(j):
        tt = times[j]
        i = min(int(round(tt / dt_rec)), len(t) - 1)
        after = j >= j_impact
        trail = X[max(0, i - 250):i + 1:4]
        ring = None
        if after and tt - tf < 0.25:
            age = (tt - tf) / 0.25 if j > freeze_frames[-1] else (j - j_impact) / max(1, len(freeze_frames))
            ring = (contact[:2], 0.01 + 0.07 * age, 1 - age)
        stage = "ЛАЗЕРНОЕ СОПРОВОЖДЕНИЕ" if not after else "ЦЕЛЬ ПОРАЖЕНА"
        scene.update(X[i], Q[i], frame_idx=j, trail=trail, contact=contact if after else None, ring=ring,
                     fx_age=(j - j_impact) / fps if after else None,
                     hud={"lines": _hud_lines(traj, i, v_cm, w_mag, stage), "face": face})
        dot_z.set_data([t[i]], [X[i, 2]])
        cur_d.set_xdata([t[i], t[i]])
        mode = "стоп-кадр" if j in freeze_frames else (
            f"замедление ×{1 / slow:.0f}" if abs(tt - tf) < 0.05 else f"замедление ×{1 / speed:.1f}")
        sub.set_text(f"t = {t[i]:.3f} с   ·   {mode}")
        if after:
            txt = f"Первое касание: грань {face}  ← исход испытания"
            if t[i] >= traj["t_rest"] - 1e-9:
                txt += f"\nОстановился на грани {rest_face}" if rest_face > 0 else "\nОстановился на торце"
            banner.set_text(txt)
            banner.set_bbox(dict(boxstyle="round,pad=0.35", fc=th["bg"], ec=th["faces"][face], lw=2.5))
        else:
            banner.set_text("")
            banner.set_bbox(None)
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps, blit=False)
    return fig, anim, traj


# ---------------------------------------------------------------------------
# Серия бросков со «счётчиком»
# ---------------------------------------------------------------------------
def animate_series(model: CaseModel, n_throws=10, seed=3, fps=24, speed=0.7, phys_counts=PHYS_COUNTS, fx=True):
    th = THEMES["neon" if fx else "light"]
    rng = np.random.default_rng(seed)
    trajs = [model.trajectory(rng=rng, fps=1000, t_after_rest=0.25) for _ in range(n_throws)]
    contacts, kin = [], []
    for tr in trajs:
        i_f = int(round(tr["t_first"] / (tr["t"][1] - tr["t"][0])))
        contacts.append(model.world_vertices(tr["x"][i_f], tr["q"][i_f])[tr["first_vertex"]])
        kin.append(_kinematics(tr))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(th["bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=0.01, right=0.96, top=0.84, bottom=0.12, wspace=0.12)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    axb = fig.add_subplot(gs[1])
    scene = Scene3D(ax3, model, spin=0.0, azim=-62, theme="neon" if fx else "light", fx=fx, fig=fig, seed=seed)

    fig.text(0.015, 0.95, "Серия компьютерных бросков", fontsize=17, fontweight="bold", color=th["ink"])
    sub = fig.text(0.015, 0.905, "", fontsize=11, color=th["ink2"])
    banner = fig.text(0.015, 0.865, "", fontsize=14, fontweight="bold", color=th["ink"], va="top", zorder=30)

    n_phys = sum(phys_counts.values())
    _style_2d(axb, th)
    axb.grid(axis="x", visible=False)
    bars = axb.bar(FACES, [0, 0, 0], color=[th["faces"][f] for f in FACES], width=0.6)
    for f in FACES:
        p = phys_counts[f] / n_phys
        axb.plot([f - 0.38, f + 0.38], [p, p], color=th["ink"], lw=2, ls=(0, (1, 1.2)))
    axb.plot([], [], color=th["ink"], lw=2, ls=(0, (1, 1.2)), label=f"реальный эксперимент (N = {n_phys})")
    axb.set_xticks(FACES, [f"грань {f}" for f in FACES])
    axb.set_ylim(0, 1.15)
    axb.set_yticks(np.linspace(0, 1, 6))
    axb.set_ylabel("относительная частота p̂ᵢ = nᵢ / n")
    axb.set_title("Частоты по фактически выпавшим граням", loc="left", fontsize=12, fontweight="bold")
    axb.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=th["ink"])
    bar_txt = [axb.text(f, 0.01, "", ha="center", va="bottom", fontsize=11, color=th["ink"]) for f in FACES]
    plus_one = axb.text(1, 0.5, "", ha="center", va="bottom", fontsize=22, fontweight="bold", color="#ffe14d",
                        path_effects=[pe.withStroke(linewidth=4, foreground="#ff2d2d")])

    # расписание кадров: (номер броска, индекс записи)
    frames = []
    for b, tr in enumerate(trajs):
        dt_rec = tr["t"][1] - tr["t"][0]
        for tt in np.arange(0, tr["t"][-1], speed / fps):
            frames.append((b, min(int(round(tt / dt_rec)), len(tr["t"]) - 1)))
        frames += [(b, len(tr["t"]) - 1)] * int(0.4 * fps)

    counts = {f: 0 for f in FACES}
    state = {"b": -1, "j_hit": None}

    def draw(j):
        b, i = frames[j]
        tr = trajs[b]
        if b != state["b"]:
            state.update(b=b, j_hit=None)
            scene.reset_camera(contacts[b])
        tt = tr["t"][i]
        after = tt >= tr["t_first"] - 1e-9
        face = tr["first_face"]
        if after and state["j_hit"] is None:
            counts[face] += 1
            state["j_hit"] = j
        age = (j - state["j_hit"]) / fps if after else None
        contact = contacts[b]
        ring = (contact[:2], 0.01 + 0.07 * (tt - tr["t_first"]) / 0.3, 1 - (tt - tr["t_first"]) / 0.3) \
            if after and tt - tr["t_first"] < 0.3 else None
        stage = "ЛАЗЕРНОЕ СОПРОВОЖДЕНИЕ" if not after else "ЦЕЛЬ ПОРАЖЕНА"
        scene.update(tr["x"][i], tr["q"][i], frame_idx=j, trail=tr["x"][max(0, i - 250):i + 1:4],
                     contact=contact if after else None, ring=ring, fx_age=age,
                     hud={"lines": _hud_lines(tr, i, *kin[b], stage), "face": face})

        n = sum(counts.values())
        for k, f in enumerate(FACES):
            p = counts[f] / n if n else 0
            bars[k].set_height(p)
            bar_txt[k].set_text(f"{counts[f]}\n{p:.2f}" if n else "0")
            bar_txt[k].set_y(p + 0.01)
            glow = fx and after and age is not None and age < 0.6 and f == face
            bars[k].set_edgecolor("#ffe14d" if glow else "none")
            bars[k].set_linewidth(3 if glow else 0)
        if fx and after and age is not None and age < 0.9:
            p = counts[face] / n
            plus_one.set_text("+1")
            plus_one.set_position((face, min(p + 0.1 + 0.12 * age, 1.02)))
            plus_one.set_alpha(float(np.clip(1.5 - age / 0.6, 0, 1)))
        else:
            plus_one.set_text("")
        sub.set_text(f"бросок {b + 1} из {len(trajs)}   ·   t = {tt:.2f} с   ·   "
                     f"счётчик: 1 → {counts[1]},  2 → {counts[2]},  3 → {counts[3]}   (n = {n})")
        if after:
            banner.set_text(f"Первое касание: грань {face}")
            banner.set_bbox(dict(boxstyle="round,pad=0.35", fc=th["bg"], ec=th["faces"][face], lw=2.5))
        else:
            banner.set_text("")
            banner.set_bbox(None)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    return fig, anim, trajs


# ---------------------------------------------------------------------------
def save(anim, path, fps, dpi=100):
    """Сохраняет анимацию в .mp4 (нужен ffmpeg) или .gif (Pillow)."""
    fc = anim._fig.get_facecolor()
    if str(path).endswith(".gif"):
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": fc})
    else:
        anim.save(path, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
                  dpi=dpi, savefig_kwargs={"facecolor": fc})


def mp4_to_gif(mp4_path, gif_path, fps=15, width=960):
    """Быстро делает .gif из готового .mp4 (без повторной отрисовки кадров)."""
    import subprocess
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4_path), "-vf", vf, str(gif_path)], check=True)


def save_all(anim, base_path, fps):
    """Сохраняет base_path.mp4 и base_path.gif (если ffmpeg недоступен — только .gif через Pillow)."""
    import shutil
    if shutil.which("ffmpeg"):
        save(anim, base_path + ".mp4", fps)
        mp4_to_gif(base_path + ".mp4", base_path + ".gif", fps=min(fps, 15))
    else:
        save(anim, base_path + ".gif", fps=fps, dpi=80)
