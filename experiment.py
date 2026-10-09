# -*- coding: utf-8 -*-
"""
Компьютерный эксперимент (ЛР №1, разделы 4–5): серии бросков, частоты,
накопленные частоты, сравнение с моделью и с реальным экспериментом.
"""
import numpy as np
import matplotlib.pyplot as plt

from case_sim import CaseModel, CaseParams, FACES

# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
# Реальный эксперимент (счётчик на фото): сколько раз выпала каждая грань.
PHYS_COUNTS = {1: 42, 2: 31, 3: 27}

# Вероятности исходной модели из раздела 2 (заданы ДО эксперимента).
# По умолчанию — симметричная модель. Если в отчёте вы задали другие p_i
# (например, p1 = 0.40, p2 = p3 = 0.30) — впишите их сюда.
P_MODEL = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

# Цвет каждой грани — один и тот же на всех графиках и в анимации.
FACE_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------
def counts_of(outcomes):
    outcomes = np.asarray(outcomes)
    return {f: int((outcomes == f).sum()) for f in FACES}


def freqs_of(outcomes):
    n = len(outcomes)
    return {f: c / n for f, c in counts_of(outcomes).items()}


def wilson_ci(k, n, z=1.96):
    """95%-доверительный интервал Уилсона для вероятности по k успехам из n."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def chi2_test(counts, probs):
    """
    Критерий согласия χ² Пирсона: H0 — исходы имеют вероятности probs.
    Для k = 3 исходов число степеней свободы 2, и p-value = exp(-χ²/2) точно.
    """
    n = sum(counts.values())
    chi2 = sum((counts[f] - n * probs[f]) ** 2 / (n * probs[f]) for f in FACES)
    return chi2, float(np.exp(-chi2 / 2))


def cumulative_freq(outcomes):
    """Накопленная относительная частота p̂_i(n), n = 1..N, для каждой грани."""
    outcomes = np.asarray(outcomes)
    n = np.arange(1, len(outcomes) + 1)
    return {f: np.cumsum(outcomes == f) / n for f in FACES}


# ---------------------------------------------------------------------------
# Серии
# ---------------------------------------------------------------------------
def run_series(model: CaseModel, sizes=(10, 100, 1000, 10_000, 100_000), seed=2026):
    """Независимые компьютерные эксперименты объёма N для каждого N из sizes."""
    rng = np.random.default_rng(seed)
    out = {}
    for N in sizes:
        out[N] = model.run(N, rng=rng)
    return out


def table_rows(series):
    rows = []
    for N, r in series.items():
        pf, pr = freqs_of(r["first"]), freqs_of(r["rest"])
        rows.append((N, pf, pr, int((r["rest"] == 0).sum())))
    return rows


def print_table(series):
    print(f"{'N':>7} | {'первое касание: p̂1   p̂2    p̂3':>32} | {'остановка: p̂1   p̂2    p̂3':>27}")
    print("-" * 75)
    for N, pf, pr, n_end in table_rows(series):
        print(f"{N:>7} | {pf[1]:>18.4f} {pf[2]:.4f} {pf[3]:.4f} | {pr[1]:>15.4f} {pr[2]:.4f} {pr[3]:.4f}"
              + (f"   (на торце: {n_end})" if n_end else ""))


def markdown_table(series):
    lines = ["| N | p̂₁ (1-е касание) | p̂₂ | p̂₃ | p̂₁ (остановка) | p̂₂ | p̂₃ |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for N, pf, pr, _ in table_rows(series):
        lines.append(f"| {N} | {pf[1]:.4f} | {pf[2]:.4f} | {pf[3]:.4f} | {pr[1]:.4f} | {pr[2]:.4f} | {pr[3]:.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def plot_convergence(outcomes, title, p_model=P_MODEL, phys_counts=PHYS_COUNTS, ax=None):
    """Накопленная частота p̂_i(n) для всех граней + p_i модели и частоты реального эксперимента."""
    cum = cumulative_freq(outcomes)
    n = np.arange(1, len(outcomes) + 1)
    n_phys = sum(phys_counts.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.2))
    for f in FACES:
        c = FACE_COLORS[f]
        ax.plot(n, cum[f], color=c, lw=2, label=f"Грань {f}: p̂{f}(n), симуляция")
        ax.axhline(phys_counts[f] / n_phys, color=c, lw=1.5, ls=":", alpha=0.9)
    # подписи финальных значений справа, раздвинутые, чтобы не налезали друг на друга
    ys = sorted((cum[f][-1], f) for f in FACES)
    pos = [ys[0][0]]
    for y, _ in ys[1:]:
        pos.append(max(y, pos[-1] + 0.045))
    for (y, f), yp in zip(ys, pos):
        ax.text(len(n) * 1.12, yp, f"p̂{f} = {y:.3f}", color=INK, va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    for f in FACES:
        ax.axhline(p_model[f], color=INK_2, lw=1.2, ls="--", zorder=1)
    ax.plot([], [], color=INK_2, ls="--", label="p_i исходной модели")
    ax.plot([], [], color=INK_2, ls=":", lw=1.5, label=f"реальный эксперимент (N = {n_phys}), цвет — грань")
    ax.set_xscale("log")
    ax.set_xlim(1, len(n) * 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("число компьютерных бросков n")
    ax.set_ylabel("накопленная относительная частота")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    return ax


def plot_comparison(series_big, p_model=P_MODEL, phys_counts=PHYS_COUNTS, ax=None):
    """p_i модели, p̂_i реального эксперимента (с 95% ДИ) и p̂_i симуляции (оба правила)."""
    n_phys = sum(phys_counts.values())
    groups = [
        ("Исходная\nмодель p_i", {f: p_model[f] for f in FACES}, None),
        (f"Реальный\nэксперимент\n(N = {n_phys})", {f: phys_counts[f] / n_phys for f in FACES},
         {f: wilson_ci(phys_counts[f], n_phys) for f in FACES}),
        (f"Симуляция:\nпервое касание\n(N = {len(series_big['first'])})", freqs_of(series_big["first"]), None),
        (f"Симуляция:\nгрань остановки\n(N = {len(series_big['rest'])})", freqs_of(series_big["rest"]), None),
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.26
    for gi, (name, p, ci) in enumerate(groups):
        for j, f in enumerate(FACES):
            x = gi + (j - 1) * (w + 0.02)
            ax.bar(x, p[f], width=w, color=FACE_COLORS[f], label=f"Грань {f}" if gi == 0 else None)
            if ci is not None:
                lo, hi = ci[f]
                ax.errorbar(x, p[f], yerr=[[p[f] - lo], [hi - p[f]]], color=INK, capsize=4, lw=1.2)
            ax.text(x, p[f] + (0.012 if ci is None else ci[f][1] - p[f] + 0.012), f"{p[f]:.2f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1 / 3, color=INK_2, lw=1, ls="--", zorder=0)
    ax.text(-0.5, 1 / 3 + 0.005, "1/3", va="bottom", ha="left", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, max(0.62, max(max(g[1].values()) for g in groups) + 0.12))
    ax.set_ylabel("вероятность / относительная частота")
    ax.set_title("Сравнение: модель, реальный эксперимент и симуляция", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.grid(axis="x", visible=False)
    return ax


def mass_sweep(base: CaseModel, extra_masses_g=(0, 5, 10, 15, 20, 30, 40), n=20_000, seed=7):
    """Как зависит распределение исходов от «лишней» массы на грани 1."""
    res = {"dm": np.array(extra_masses_g, float), "first": [], "rest": [], "shift_mm": []}
    for dm in extra_masses_g:
        cp = CaseParams(**{**base.case.__dict__, "extra_mass": dm / 1000})
        m = CaseModel(cp, base.toss, base.surface, base.sim)
        r = m.run(n, seed=seed)
        res["first"].append(freqs_of(r["first"]))
        res["rest"].append(freqs_of(r["rest"]))
        res["shift_mm"].append(-m.body.com_shift[2] * 1000)
    return res


def plot_mass_sweep(sw, phys_counts=PHYS_COUNTS, n_sim=None):
    n_phys = sum(phys_counts.values())
    lo, hi = wilson_ci(phys_counts[1], n_phys)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, key, title in ((axes[0], "first", "Исход = грань первого касания"),
                           (axes[1], "rest", "Исход = грань, на которой футляр остановился")):
        ax.axhspan(lo, hi, color=FACE_COLORS[1], alpha=0.10, lw=0)
        ax.axhline(phys_counts[1] / n_phys, color=FACE_COLORS[1], ls=":", lw=1.5)
        for f in FACES:
            y = [p[f] for p in sw[key]]
            ax.plot(sw["dm"], y, "-o", color=FACE_COLORS[f], lw=2, ms=6,
                    mec=SURFACE, mew=1.5, label=f"Грань {f}")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("дополнительная масса на грани 1, г")
        ax.set_ylim(0.15, 0.6)
    axes[0].set_ylabel("относительная частота")
    axes[0].text(sw["dm"][0], hi + 0.01, f"реальный эксперимент: p̂₁ = {phys_counts[1] / n_phys:.2f} (95% ДИ)",
                 color=INK_2, fontsize=9)
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Влияние «тяжёлой» грани на исход" + (f" (N = {n_sim} бросков на точку)" if n_sim else ""),
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
