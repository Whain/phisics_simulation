# -*- coding: utf-8 -*-
"""
Полный компьютерный эксперимент ЛР №1 для треугольного футляра.

    python run_all.py            # всё: серии N = 10…10^5, графики, анимации  (~2–4 мин)
    python run_all.py --quick    # быстро: N до 10^4, без перебора масс
    python run_all.py --no-anim  # без анимаций

Результаты складываются в папку results/.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case_sim import CaseModel, HAVE_NUMBA, FACES
from experiment import (PHYS_COUNTS, P_MODEL, run_series, print_table, markdown_table, freqs_of,
                        counts_of, chi2_test, wilson_ci, binom_test_greater, plot_convergence, plot_comparison,
                        mass_sweep, plot_mass_sweep)


def describe(model):
    c, t, s, b = model.case, model.toss, model.surface, model.body
    return "\n".join([
        "**Футляр** (прямая треугольная призма, тонкостенная оболочка):",
        f"- сторона сечения {c.side * 100:.1f} см, длина {c.length * 100:.1f} см;",
        f"- масса корпуса {c.mass_shell * 1000:.0f} г + {c.extra_mass * 1000:.0f} г на грани 1 "
        f"(всего {b.mass * 1000:.0f} г);",
        f"- центр масс смещён к грани 1 на {-b.com_shift[2] * 1000:.2f} мм;",
        f"- главные моменты инерции: {', '.join(f'{v:.3g}' for v in np.diag(b.inertia))} кг·м².",
        "",
        "**Случайные начальные условия (процедура броска):**",
        f"- высота центра масс h₀ ~ U({t.h0[0]}, {t.h0[1]}) м;",
        f"- вертикальная скорость v_z ~ U({t.vz[0]}, {t.vz[1]}) м/с (вверх), горизонтальные v_x, v_y ~ N(0, {t.v_xy}) м/с;",
        f"- ориентация — равномерно случайный поворот; угловая скорость: ось равномерно на сфере, "
        f"|ω| ~ U({t.omega[0]}, {t.omega[1]}) рад/с.",
        "",
        f"**Пол:** коэффициент восстановления e = {s.restitution}, коэффициент трения μ = {s.friction}.",
        f"**Численно:** шаг {model.sim.dt * 1000:.1f} мс, g = {model.sim.g} м/с².",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)
    if not HAVE_NUMBA:
        print("ВНИМАНИЕ: numba не установлена — будет очень медленно (pip install numba)")

    model = CaseModel()
    print(describe(model).replace("**", ""), "\n")

    # --- 1. Серии разного объёма ---
    sizes = (10, 100, 1000, 10_000) if args.quick else (10, 100, 1000, 10_000, 100_000)
    t0 = time.time()
    series = run_series(model, sizes)
    print(f"Серии N = {sizes} посчитаны за {time.time() - t0:.1f} с\n")
    print_table(series)
    big = series[max(sizes)]
    N_big = len(big["first"])

    # --- 2. Графики ---
    plot_convergence(big["first"], f"Накопленная частота: исход = грань первого касания (N = {N_big})")
    plt.savefig(out("convergence_first_contact.png"), dpi=150, bbox_inches="tight")
    plot_convergence(big["rest"], f"Накопленная частота: исход = грань, на которой футляр остановился (N = {N_big})")
    plt.savefig(out("convergence_rest.png"), dpi=150, bbox_inches="tight")
    plot_comparison(big)
    plt.savefig(out("comparison.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- 3. Проверка согласия (χ², 2 степени свободы) ---
    n_phys = sum(PHYS_COUNTS.values())
    tests = [
        ("исходная модель p_i", P_MODEL),
        ("симуляция, первое касание", freqs_of(big["first"])),
        ("симуляция, грань остановки", freqs_of(big["rest"])),
    ]
    chi_lines = []
    print(f"\nРеальный эксперимент: {PHYS_COUNTS} (N = {n_phys})")
    for name, p in tests:
        chi2, pv = chi2_test(PHYS_COUNTS, p)
        verdict = "не отвергается" if pv > 0.05 else "ОТВЕРГАЕТСЯ"
        line = f"реальные данные vs {name}: χ² = {chi2:.2f}, p-value = {pv:.3f} → гипотеза {verdict} (α = 0.05)"
        print(line)
        chi_lines.append(f"- {line}")
    # направленная гипотеза из раздела 2: грань 1 выпадает чаще, чем в модели
    p_one = binom_test_greater(PHYS_COUNTS[1], n_phys, P_MODEL[1])
    line = (f"грань 1, альтернатива p₁ > {P_MODEL[1]:.3f}: точный односторонний биномиальный критерий, "
            f"P(X ≥ {PHYS_COUNTS[1]}) = {p_one:.3f} → " + ("значимо" if p_one < 0.05 else "не значимо") + " при α = 0.05")
    print(line)
    chi_lines.append(f"- {line}")
    ci_lines = []
    for f in FACES:
        lo, hi = wilson_ci(PHYS_COUNTS[f], n_phys)
        ci_lines.append(f"- грань {f}: p̂ = {PHYS_COUNTS[f] / n_phys:.2f}, 95% ДИ [{lo:.3f}; {hi:.3f}]")

    # --- 4. Перебор «лишней» массы на грани 1 ---
    if not args.quick:
        t0 = time.time()
        sw = mass_sweep(model, n=20_000)
        print(f"\nПеребор массы: {time.time() - t0:.1f} с")
        for dm, pf, pr, sh in zip(sw["dm"], sw["first"], sw["rest"], sw["shift_mm"]):
            print(f"  Δm = {dm:4.0f} г (сдвиг ЦМ {sh:4.2f} мм): первое касание p̂1 = {pf[1]:.3f}, остановка p̂1 = {pr[1]:.3f}")
        plot_mass_sweep(sw, n_sim=20_000)
        plt.savefig(out("mass_sweep.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

    # --- 5. Сводка ---
    pf, pr = freqs_of(big["first"]), freqs_of(big["rest"])
    summary = "\n".join([
        "# Результаты компьютерного эксперимента", "",
        "_Файл создан автоматически скриптом `run_all.py`._", "",
        describe(model), "",
        "## Частоты для разных N (независимые серии)", "",
        markdown_table(series), "",
        "## Сравнение", "",
        "| грань | p_i (модель) | p̂ реальный эксп. | p̂ симуляция (1-е касание) | p̂ симуляция (остановка) |",
        "|---:|---:|---:|---:|---:|",
        *[f"| {f} | {P_MODEL[f]:.3f} | {PHYS_COUNTS[f] / n_phys:.2f} | {pf[f]:.4f} | {pr[f]:.4f} |" for f in FACES],
        "", f"Доверительные интервалы для реального эксперимента (N = {n_phys}, метод Уилсона):", "",
        *ci_lines, "", "Критерий согласия χ² Пирсона (2 степени свободы) и направленный критерий для грани 1:", "",
        *chi_lines, "",
    ])
    with open(out("summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"\nСводка: {out('summary.md')}")

    # --- 6. Анимации ---
    if not args.no_anim:
        from animate import animate_throw, animate_series, save_all
        t0 = time.time()
        fig, anim, tr = animate_throw(model, seed=11, fx=True)      # с лазерами и взрывами
        save_all(anim, out("throw"), fps=30)
        plt.close(fig)
        fig, anim, tr = animate_throw(model, seed=11, fx=False)     # спокойная версия для отчёта
        save_all(anim, out("throw_clean"), fps=30)
        plt.close(fig)
        print(f"Анимация одного броска: грань первого касания {tr['first_face']}, "
              f"остановился на грани {tr['rest_face']} ({time.time() - t0:.0f} с)")
        t0 = time.time()
        fig, anim, trs = animate_series(model, n_throws=10, seed=3, fx=True)
        save_all(anim, out("series"), fps=24)
        plt.close(fig)
        print(f"Анимация серии: {[t['first_face'] for t in trs]} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
