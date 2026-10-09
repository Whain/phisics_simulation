# -*- coding: utf-8 -*-
"""
Физическая модель подбрасывания треугольного футляра для очков
(ЛР №1, раздел 4 «Компьютерное моделирование»).

Объект
------
Футляр — прямая призма с равносторонним треугольником в сечении, тонкостенная
оболочка. Боковые грани пронумерованы 1, 2, 3. К грани 1 добавлена небольшая
дополнительная масса (по предположению, грань 1 «чуть тяжелее»), поэтому центр
масс смещён к грани 1, а тензор инерции считается из реального распределения масс.

Движение (твёрдое тело, 6 степеней свободы)
------------------------------------------
* полёт: центр масс движется под действием силы тяжести, вращение свободное —
  момент импульса L сохраняется, угловая скорость w = I(t)^-1 L
  (это и есть уравнения Эйлера для несимметричного волчка);
* удары о пол: импульсный метод (sequential impulses) в вершинах призмы —
  коэффициент восстановления + кулоновское трение;
* после первого касания футляр подпрыгивает, кувыркается и останавливается.

Исход испытания
---------------
Как и в физическом эксперименте — грань ПЕРВОГО касания: номер боковой грани,
внешняя нормаль которой в момент первого касания пола направлена ближе всего
вниз («этой гранью футляр ударился о пол»). Дополнительно записывается грань,
на которой футляр в итоге остановился.

Никакие вероятности в модель заранее НЕ закладываются: исход каждого броска
определяется только начальными условиями и уравнениями движения, а частоты
считаются по фактически выпавшим исходам.
"""
from dataclasses import dataclass, field

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # без numba всё работает так же, но в ~100 раз медленнее
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


FACES = (1, 2, 3)
END_CAP = 0  # код исхода «остановился стоя на торце» (для длинного футляра практически не встречается)


# ----------------------------------------------------------------------------
# Параметры модели
# ----------------------------------------------------------------------------
@dataclass
class CaseParams:
    """Геометрия и массы футляра (подставьте свои измерения)."""
    side: float = 0.055        # сторона треугольного сечения, м
    length: float = 0.160      # длина футляра, м
    mass_shell: float = 0.050  # масса однородного корпуса (все 5 граней), кг
    extra_mass: float = 0.008  # дополнительная масса, равномерно распределённая по грани 1, кг


@dataclass
class TossParams:
    """Процедура броска: распределения случайных начальных условий."""
    h0: tuple = (0.9, 1.1)      # высота центра масс в момент броска, м            ~ U(a, b)
    vz: tuple = (0.5, 2.5)      # начальная вертикальная скорость (вверх), м/с      ~ U(a, b)
    v_xy: float = 0.3           # горизонтальные компоненты скорости, м/с          ~ N(0, σ)
    omega: tuple = (5.0, 25.0)  # модуль угловой скорости, рад/с                   ~ U(a, b)
    # Направление оси вращения — равномерно на сфере.
    # Начальная ориентация — равномерно распределённый случайный поворот (равномерно на SO(3)).


@dataclass
class SurfaceParams:
    """Свойства пола."""
    restitution: float = 0.30  # коэффициент восстановления (0 — абсолютно неупругий удар, 1 — упругий)
    friction: float = 0.40     # коэффициент трения скольжения


@dataclass
class SimParams:
    """Численные параметры."""
    dt: float = 1e-3      # шаг интегрирования, с
    t_max: float = 6.0    # максимальное время моделирования одного броска, с
    n_iter: int = 12      # итераций решателя контактов на шаг
    g: float = 9.81       # ускорение свободного падения, м/с²


# ----------------------------------------------------------------------------
# Геометрия и распределение масс
# ----------------------------------------------------------------------------
@dataclass
class Body:
    verts: np.ndarray        # (6, 3) вершины в связанной системе, относительно центра масс
    normals: np.ndarray      # (3, 3) внешние нормали боковых граней 1, 2, 3
    face_polys: list         # индексы вершин каждой грани (для отрисовки): 3 боковые + 2 торца
    mass: float
    inertia: np.ndarray      # (3, 3) тензор инерции относительно центра масс
    inertia_inv: np.ndarray
    com_shift: np.ndarray    # смещение центра масс от геометрического центра, м
    params: CaseParams = field(repr=False, default=None)


def _sample_rect(p0, p1, p3, n):
    """Равномерная сетка точек на параллелограмме p0, p1, (p1+p3-p0), p3."""
    s = (np.arange(n) + 0.5) / n
    u, w = np.meshgrid(s, s, indexing="ij")
    return p0 + u.reshape(-1, 1) * (p1 - p0) + w.reshape(-1, 1) * (p3 - p0)


def _sample_triangle(a, b, c, n):
    """Равномерная сетка точек на треугольнике."""
    pts = []
    for i in range(n):
        for j in range(n - i):
            u, w = (i + 1 / 3) / n, (j + 1 / 3) / n
            pts.append(a + u * (b - a) + w * (c - a))
            if i + j < n - 1:
                u2, w2 = (i + 2 / 3) / n, (j + 2 / 3) / n
                pts.append(a + u2 * (b - a) + w2 * (c - a))
    return np.array(pts)


def build_body(p: CaseParams = None) -> Body:
    """
    Связанная система координат: ось x — вдоль футляра, сечение — в плоскости yz.
    Грань 1 — «нижняя» в связанной системе (нормаль -z), грани 2 и 3 — наклонные.
    """
    p = p or CaseParams()
    a, Lc = p.side, p.length
    r_in = a / (2 * np.sqrt(3))  # радиус вписанной окружности сечения
    R_c = a / np.sqrt(3)         # радиус описанной окружности сечения
    tri = np.array([[0.0, R_c], [a / 2, -r_in], [-a / 2, -r_in]])  # вершина, правая нижняя, левая нижняя

    verts = np.array([[sx, y, z] for sx in (-Lc / 2, Lc / 2) for (y, z) in tri])
    normals = np.array([
        [0.0, 0.0, -1.0],                     # грань 1 (между правой и левой нижними вершинами)
        [0.0, np.sqrt(3) / 2, 0.5],           # грань 2 (между вершиной и правой нижней)
        [0.0, -np.sqrt(3) / 2, 0.5],          # грань 3 (между вершиной и левой нижней)
    ])
    face_polys = [[1, 2, 5, 4], [0, 1, 4, 3], [0, 3, 5, 2], [0, 2, 1], [3, 4, 5]]

    # --- распределение масс: оболочка постоянной толщины + добавка на грани 1 ---
    n = 40
    rects = [_sample_rect(verts[f[0]], verts[f[1]], verts[f[3]], n) for f in face_polys[:3]]
    caps = [_sample_triangle(verts[f[0]], verts[f[1]], verts[f[2]], n // 2) for f in face_polys[3:]]
    area_rect = a * Lc
    area_cap = np.sqrt(3) / 4 * a ** 2
    sigma = p.mass_shell / (3 * area_rect + 2 * area_cap)  # поверхностная плотность корпуса

    pts, w = [], []
    for i, r in enumerate(rects):
        m_face = sigma * area_rect + (p.extra_mass if i == 0 else 0.0)
        pts.append(r)
        w.append(np.full(len(r), m_face / len(r)))
    for c in caps:
        pts.append(c)
        w.append(np.full(len(c), sigma * area_cap / len(c)))
    pts, w = np.vstack(pts), np.concatenate(w)

    mass = w.sum()
    com = (w[:, None] * pts).sum(0) / mass
    d = pts - com
    inertia = (w[:, None, None] * ((d * d).sum(1)[:, None, None] * np.eye(3)
                                   - d[:, :, None] * d[:, None, :])).sum(0)
    # из-за симметрии (плоскости x=0 и y=0) оси x, y, z — главные оси инерции
    inertia[np.abs(inertia) < 1e-12 * np.abs(inertia).max()] = 0.0
    assert np.allclose(inertia, np.diag(np.diag(inertia))), "оси связанной системы должны быть главными"

    return Body(verts=verts - com, normals=normals, face_polys=face_polys, mass=mass,
                inertia=inertia, inertia_inv=np.linalg.inv(inertia), com_shift=com, params=p)


# ----------------------------------------------------------------------------
# Ядро моделирования (компилируется numba)
# ----------------------------------------------------------------------------
@njit(cache=True)
def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (y * y + z * z); R[0, 1] = 2 * (x * y - w * z);     R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z);     R[1, 1] = 1 - 2 * (x * x + z * z); R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y);     R[2, 1] = 2 * (y * z + w * x);     R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


@njit(cache=True)
def _mat3_vec(M, v, out):
    for i in range(3):
        out[i] = M[i, 0] * v[0] + M[i, 1] * v[1] + M[i, 2] * v[2]


@njit(cache=True)
def _world_inertia_inv(R, Ib_inv, out):
    """I_world^-1 = R · I_body^-1 · R^T."""
    tmp = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += R[i, k] * Ib_inv[k, j]
            tmp[i, j] = s
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += tmp[i, k] * R[j, k]
            out[i, j] = s


@njit(cache=True)
def down_face(R, normals):
    """Номер (1..3) боковой грани, внешняя нормаль которой направлена ближе всего вниз."""
    best, best_val = 0, 1e9
    for f in range(3):
        nz = R[2, 0] * normals[f, 0] + R[2, 1] * normals[f, 1] + R[2, 2] * normals[f, 2]
        if nz < best_val:
            best_val, best = nz, f
    return best + 1


@njit(cache=True)
def _rotate_free(q, L, I_diag, axis, h):
    """
    Точный поток свободного волчка с энергией T_i = Π_i² / (2 I_i):
    поворот тела вокруг собственной главной оси i на угол Π_i h / I_i,
    где Π = R^T L — момент импульса в связанной системе. L (в мировой системе) не меняется.
    """
    R = quat_to_mat(q)
    Pi = R[0, axis] * L[0] + R[1, axis] * L[1] + R[2, axis] * L[2]
    th = Pi * h / I_diag[axis]
    c, s = np.cos(th / 2), np.sin(th / 2)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    # q ← q ⊗ (cos θ/2, sin θ/2 · e_axis)  — поворот в связанной системе
    if axis == 0:
        q[0] = q0 * c - q1 * s; q[1] = q1 * c + q0 * s; q[2] = q2 * c + q3 * s; q[3] = q3 * c - q2 * s
    elif axis == 1:
        q[0] = q0 * c - q2 * s; q[1] = q1 * c - q3 * s; q[2] = q2 * c + q0 * s; q[3] = q3 * c + q1 * s
    else:
        q[0] = q0 * c - q3 * s; q[1] = q1 * c + q2 * s; q[2] = q2 * c - q1 * s; q[3] = q3 * c + q0 * s
    nq = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    for j in range(4):
        q[j] /= nq


@njit(cache=True)
def simulate_one(x, v, q, L, verts, normals, Ib_inv, I_diag, mass, g, e, mu, dt, max_steps,
                 n_iter, stop_at_first, rec_every, rec):
    """
    Моделирует один бросок. x, v, q, L изменяются на месте.
      x — положение центра масс, v — его скорость, q — ориентация (кватернион w,x,y,z),
      L — момент импульса относительно центра масс (мировая система).
    rec — массив (n, 8) для записи траектории (t, x, y, z, qw, qx, qy, qz) каждые
    rec_every шагов (rec_every = 0 — не записывать).
    Возвращает (грань первого касания, шаг первого касания, вершина первого касания,
                грань остановки, шаг остановки, число записанных кадров).
    """
    nv = verts.shape[0]
    r = np.empty((nv, 3))
    z = np.empty(nv)
    idx = np.empty(nv, np.int64)
    Kn = np.empty(nv); Kt1 = np.empty(nv); Kt2 = np.empty(nv); tgt = np.empty(nv)
    lam_n = np.empty(nv); lam_t1 = np.empty(nv); lam_t2 = np.empty(nv)
    Iinv = np.empty((3, 3))
    w = np.empty(3)
    tmp = np.empty(3)
    dL = np.empty(3)

    beta, slop, v_bounce = 0.2, 5e-4, 0.1   # стабилизация проникновения и порог «упругого» удара
    v_rest, w_rest, t_rest = 0.02, 0.3, 0.2  # критерий остановки: |v|, |w| малы в течение t_rest
    inv_m = 1.0 / mass

    first_face, first_step, first_vertex = -1, -1, -1
    rest_face, rest_step = -1, -1
    still = 0
    n_rec = 0
    R = quat_to_mat(q)

    for step in range(max_steps):
        R = quat_to_mat(q)
        _world_inertia_inv(R, Ib_inv, Iinv)

        if rec_every > 0 and step % rec_every == 0 and n_rec < rec.shape[0]:
            rec[n_rec, 0] = step * dt
            rec[n_rec, 1:4] = x
            rec[n_rec, 4:8] = q
            n_rec += 1

        # --- поиск вершин, касающихся пола (z < 0) ---
        nc = 0
        zmin, kmin = 1e9, -1
        for k in range(nv):
            for j in range(3):
                r[k, j] = R[j, 0] * verts[k, 0] + R[j, 1] * verts[k, 1] + R[j, 2] * verts[k, 2]
            z[k] = x[2] + r[k, 2]
            if z[k] < zmin:
                zmin, kmin = z[k], k
            if z[k] < 0.0:
                idx[nc] = k
                nc += 1

        if nc > 0 and first_step < 0:
            first_face = down_face(R, normals)
            first_step = step
            first_vertex = kmin
            if stop_at_first:
                break

        # --- сила тяжести и угловая скорость w = I^-1 L ---
        v[2] -= g * dt
        _mat3_vec(Iinv, L, w)

        # --- удары/контакт: последовательные импульсы ---
        if nc > 0:
            for c in range(nc):
                k = idx[c]
                rx, ry, rz = r[k, 0], r[k, 1], r[k, 2]
                # эффективные массы K = 1/m + (r×n)·I^-1·(r×n) для нормали и двух касательных
                tmp[0], tmp[1], tmp[2] = ry, -rx, 0.0          # r × e_z
                _mat3_vec(Iinv, tmp, dL)
                Kn[c] = inv_m + tmp[0] * dL[0] + tmp[1] * dL[1]
                tmp[0], tmp[1], tmp[2] = 0.0, rz, -ry          # r × e_x
                _mat3_vec(Iinv, tmp, dL)
                Kt1[c] = inv_m + tmp[1] * dL[1] + tmp[2] * dL[2]
                tmp[0], tmp[1], tmp[2] = -rz, 0.0, rx          # r × e_y
                _mat3_vec(Iinv, tmp, dL)
                Kt2[c] = inv_m + tmp[0] * dL[0] + tmp[2] * dL[2]
                # нормальная скорость точки до удара и желаемая скорость после
                vn0 = v[2] + w[0] * ry - w[1] * rx
                target = -e * vn0 if vn0 < -v_bounce else 0.0
                pen = -z[k] - slop
                if pen > 0.0:
                    target = max(target, beta * pen / dt)
                tgt[c] = target
                lam_n[c] = 0.0; lam_t1[c] = 0.0; lam_t2[c] = 0.0

            for it in range(n_iter):
                for c in range(nc):
                    k = idx[c]
                    rx, ry, rz = r[k, 0], r[k, 1], r[k, 2]
                    # нормальный импульс (только отталкивание: накопленный λ ≥ 0)
                    uz = v[2] + w[0] * ry - w[1] * rx
                    dl = (tgt[c] - uz) / Kn[c]
                    new = max(lam_n[c] + dl, 0.0)
                    dl = new - lam_n[c]
                    lam_n[c] = new
                    v[2] += dl * inv_m
                    dL[0], dL[1], dL[2] = ry * dl, -rx * dl, 0.0   # r × (0, 0, dl)
                    L[0] += dL[0]; L[1] += dL[1]
                    _mat3_vec(Iinv, dL, tmp)
                    w[0] += tmp[0]; w[1] += tmp[1]; w[2] += tmp[2]

                    # трение Кулона: |λ_t| ≤ μ λ_n
                    ux = v[0] + w[1] * rz - w[2] * ry
                    uy = v[1] + w[2] * rx - w[0] * rz
                    n1 = lam_t1[c] - ux / Kt1[c]
                    n2 = lam_t2[c] - uy / Kt2[c]
                    lim = mu * lam_n[c]
                    mag = np.sqrt(n1 * n1 + n2 * n2)
                    if mag > lim:
                        s = lim / mag
                        n1 *= s; n2 *= s
                    d1, d2 = n1 - lam_t1[c], n2 - lam_t2[c]
                    lam_t1[c], lam_t2[c] = n1, n2
                    v[0] += d1 * inv_m
                    v[1] += d2 * inv_m
                    dL[0], dL[1], dL[2] = -rz * d2, rz * d1, rx * d2 - ry * d1   # r × (d1, d2, 0)
                    L[0] += dL[0]; L[1] += dL[1]; L[2] += dL[2]
                    _mat3_vec(Iinv, dL, tmp)
                    w[0] += tmp[0]; w[1] += tmp[1]; w[2] += tmp[2]

        # --- остановка ---
        speed = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        wn = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
        if nc > 0 and speed < v_rest and wn < w_rest:
            still += 1
        else:
            still = 0
        if first_step >= 0 and still * dt >= t_rest:
            rest_step = step
            break

        # --- интегрирование ---
        # центр масс — полунеявный Эйлер;
        # вращение — симметричное расщепление свободного волчка по главным осям
        # (оси 1,2,3,2,1): каждый подшаг — точный поворот вокруг одной главной оси,
        # поэтому энергия и |L| не «уплывают» даже за много оборотов.
        x[0] += v[0] * dt; x[1] += v[1] * dt; x[2] += v[2] * dt
        _rotate_free(q, L, I_diag, 0, 0.5 * dt)
        _rotate_free(q, L, I_diag, 1, 0.5 * dt)
        _rotate_free(q, L, I_diag, 2, dt)
        _rotate_free(q, L, I_diag, 1, 0.5 * dt)
        _rotate_free(q, L, I_diag, 0, 0.5 * dt)

    if first_step >= 0 and not stop_at_first:
        R = quat_to_mat(q)
        # стоит на торце, если ось футляра почти вертикальна
        if abs(R[2, 0]) > 0.8:
            rest_face = 0
        else:
            rest_face = down_face(R, normals)
        if rest_step < 0:
            rest_step = max_steps
    return first_face, first_step, first_vertex, rest_face, rest_step, n_rec


@njit(parallel=True, cache=True)
def _simulate_batch(X, V, Q, Lw, verts, normals, Ib_inv, I_diag, mass, g, e, mu, dt, max_steps,
                    n_iter, stop_at_first):
    N = X.shape[0]
    out = np.empty((N, 5), np.int64)
    rec = np.empty((0, 8))
    for i in prange(N):
        res = simulate_one(X[i].copy(), V[i].copy(), Q[i].copy(), Lw[i].copy(), verts, normals,
                           Ib_inv, I_diag, mass, g, e, mu, dt, max_steps, n_iter, stop_at_first, 0, rec)
        out[i, 0] = res[0]; out[i, 1] = res[1]; out[i, 2] = res[2]
        out[i, 3] = res[3]; out[i, 4] = res[4]
    return out


# ----------------------------------------------------------------------------
# Удобный интерфейс
# ----------------------------------------------------------------------------
def random_quaternions(n, rng):
    """Равномерно распределённые повороты: нормированный 4-мерный гауссов вектор."""
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_unit_vectors(n, rng):
    u = rng.normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


class CaseModel:
    """Модель «футляр + бросок + пол»."""

    def __init__(self, case=None, toss=None, surface=None, sim=None):
        self.case = case or CaseParams()
        self.toss = toss or TossParams()
        self.surface = surface or SurfaceParams()
        self.sim = sim or SimParams()
        self.body = build_body(self.case)

    # --- начальные условия ---
    def sample_initial(self, n, rng):
        t = self.toss
        X = np.zeros((n, 3))
        X[:, 2] = rng.uniform(*t.h0, n)
        V = np.column_stack([rng.normal(0, t.v_xy, n), rng.normal(0, t.v_xy, n), rng.uniform(*t.vz, n)])
        Q = random_quaternions(n, rng)
        W = random_unit_vectors(n, rng) * rng.uniform(*t.omega, n)[:, None]
        # L = I_world · w,  I_world = R · I_body · R^T
        Lw = np.empty_like(W)
        for i in range(n):
            R = quat_to_mat(Q[i])
            Lw[i] = R @ self.body.inertia @ R.T @ W[i]
        return dict(X=X, V=V, Q=Q, L=Lw, W=W)

    def _args(self):
        b, s, sim = self.body, self.surface, self.sim
        return (b.verts, b.normals, b.inertia_inv, np.diag(b.inertia).copy(), b.mass, sim.g,
                s.restitution, s.friction,
                sim.dt, int(round(sim.t_max / sim.dt)), sim.n_iter)

    # --- серия бросков ---
    def run(self, n, seed=None, rng=None, until_rest=True):
        """
        n бросков. Возвращает словарь массивов:
          first — грань первого касания (1..3) — ИСХОД испытания,
          rest  — грань, на которой футляр остановился (0 — на торце; -1 если не считали),
          t_first, t_rest — времена, с.
        until_rest=False — моделировать только до первого касания (в несколько раз быстрее).
        """
        rng = rng or np.random.default_rng(seed)
        ic = self.sample_initial(n, rng)
        out = _simulate_batch(ic["X"], ic["V"], ic["Q"], ic["L"], *self._args(), not until_rest)
        dt = self.sim.dt
        return dict(first=out[:, 0], rest=out[:, 3] if until_rest else np.full(n, -1),
                    t_first=out[:, 1] * dt, t_rest=out[:, 4] * dt, first_vertex=out[:, 2], ic=ic)

    # --- траектория одного броска (для анимации) ---
    def trajectory(self, ic=None, rng=None, seed=None, fps=240, t_after_rest=0.3):
        """Записывает траекторию одного броска с частотой fps кадров в секунду."""
        if ic is None:
            rng = rng or np.random.default_rng(seed)
            ic = {k: v[0] for k, v in self.sample_initial(1, rng).items()}
        dt = self.sim.dt
        every = max(1, int(round(1.0 / (fps * dt))))
        max_steps = int(round(self.sim.t_max / dt))
        rec = np.empty((max_steps // every + 2, 8))
        x, v, q, L = (np.array(ic[k], dtype=float) for k in ("X", "V", "Q", "L"))
        args = self._args()
        res = simulate_one(x, v, q, L, *args[:-2], max_steps, args[-1], False, every, rec)
        first_face, first_step, first_vertex, rest_face, rest_step, n_rec = res
        rec = rec[:n_rec]
        # «досчитываем» несколько кадров покоя, чтобы анимация не обрывалась
        n_hold = int(t_after_rest * fps)
        if n_hold > 0 and n_rec > 0:
            hold = np.repeat(rec[-1:], n_hold, axis=0)
            hold[:, 0] = rec[-1, 0] + np.arange(1, n_hold + 1) / fps
            rec = np.vstack([rec, hold])
        return dict(t=rec[:, 0], x=rec[:, 1:4], q=rec[:, 4:8],
                    first_face=int(first_face), t_first=first_step * dt, first_vertex=int(first_vertex),
                    rest_face=int(rest_face), t_rest=rest_step * dt, ic=ic)

    def world_vertices(self, x, q):
        """Мировые координаты 6 вершин для положения x и ориентации q."""
        R = quat_to_mat(np.asarray(q, dtype=float))
        return np.asarray(x) + self.body.verts @ R.T

    def world_normals(self, q):
        R = quat_to_mat(np.asarray(q, dtype=float))
        return self.body.normals @ R.T
