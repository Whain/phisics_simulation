# -*- coding: utf-8 -*-
"""
Собирает самодостаточный ноутбук lab1_simulation.ipynb для Google Colab:
модули case_sim.py, experiment.py, animate.py записываются в ячейки %%writefile,
поэтому ноутбук можно просто загрузить в Colab и выполнить «Среда выполнения → Выполнить всё».

    python tools/build_notebook.py
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# ЛР №1, раздел 4. Компьютерное моделирование: бросок треугольного футляра

**Объект:** футляр для очков — треугольная призма, боковые грани помечены 1, 2, 3.
**Предположение:** грань 1 немного тяжелее остальных, поэтому «1» будет выпадать чаще.
**Реальный эксперимент (N = 100):** 1 → 42, 2 → 31, 3 → 27.

Здесь футляр моделируется как **твёрдое тело** (6 степеней свободы): полёт под действием силы тяжести
со свободным вращением, удары о пол с отскоком и трением, остановка.
Исход каждого компьютерного броска определяется **только физикой** — никакие вероятности заранее не
задаются; частоты считаются по фактически выпавшим граням, как и в реальном эксперименте.

Исход (как в физическом эксперименте) — **грань первого касания**: боковая грань, внешняя нормаль которой
в момент первого касания пола направлена ближе всего вниз. Дополнительно записывается грань, на которой
футляр остановился.

Как пользоваться: «Среда выполнения → Выполнить всё». Параметры футляра и броска можно поменять в разделе 2.
"""),
    code("""# numba ускоряет расчёт примерно в 100 раз (в Colab обычно уже установлена)
import subprocess, sys
try:
    import numba
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"])
QUICK = False   # True — быстрый прогон (N до 10^4, без перебора масс)"""),
    md("## 1. Код модели\nТри модуля записываются в файлы рядом с ноутбуком и импортируются ниже. "
       "Физика — в `case_sim.py`, статистика и графики — в `experiment.py`, анимация — в `animate.py`."),
    code("%%writefile case_sim.py\n" + src("case_sim.py")),
    code("%%writefile experiment.py\n" + src("experiment.py")),
    code("%%writefile animate.py\n" + src("animate.py")),
    md("""## 2. Параметры модели

Подставьте измерения своего футляра (сторона треугольника, длина, масса) и условия броска.
`extra_mass` — насколько грань 1 тяжелее (дополнительная масса, равномерно размазанная по грани 1)."""),
    code("""import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video, display

from case_sim import CaseModel, CaseParams, TossParams, SurfaceParams, SimParams
from experiment import *
from animate import animate_throw, animate_series, save_all

case = CaseParams(side=0.055, length=0.160, mass_shell=0.050, extra_mass=0.008)
toss = TossParams(h0=(0.9, 1.1), vz=(0.5, 2.5), v_xy=0.3, omega=(5.0, 25.0))
surface = SurfaceParams(restitution=0.30, friction=0.40)
model = CaseModel(case, toss, surface, SimParams())

b = model.body
print(f"масса {b.mass*1000:.0f} г, центр масс смещён к грани 1 на {-b.com_shift[2]*1000:.2f} мм")
print("главные моменты инерции, кг·м²:", np.round(np.diag(b.inertia), 7))
print("реальный эксперимент:", PHYS_COUNTS, "  модель p_i:", P_MODEL)"""),
    md("""## 3. Визуализация одного броска

Слева — 3D-вид (камера следит за футляром, пол всегда в кадре), справа — высота над полом и
«какая грань смотрит вниз» (косинус угла между внешней нормалью грани и направлением вниз).
Момент первого касания замедлен; исход — грань с наибольшим косинусом в этот момент.

`fx=True` — с лазерами, взрывом и прочими спецэффектами; `fx=False` — спокойная версия для отчёта.
Отрисовка занимает около минуты."""),
    code("""fig, anim, traj = animate_throw(model, seed=11, fx=True)
plt.close(fig)
save_all(anim, "throw", fps=30)
print("грань первого касания:", traj["first_face"], "  остановился на грани:", traj["rest_face"])
display(Video("throw.mp4", embed=True, width=960))"""),
    code("""# та же анимация без спецэффектов (для отчёта)
fig, anim, _ = animate_throw(model, seed=11, fx=False)
plt.close(fig)
save_all(anim, "throw_clean", fps=30)
display(Video("throw_clean.mp4", embed=True, width=960))"""),
    md("### Серия бросков со «счётчиком»\nЧастоты обновляются по фактически выпавшим граням "
       "(пунктир — реальный эксперимент). Отрисовка — пара минут."),
    code("""fig, anim, trajs = animate_series(model, n_throws=6 if QUICK else 10, seed=3, fx=True)
plt.close(fig)
save_all(anim, "series", fps=24)
display(Video("series.mp4", embed=True, width=960))"""),
    md("## 4. Компьютерный эксперимент для N = 10, 10², 10³, 10⁴, 10⁵"),
    code("""sizes = (10, 100, 1000, 10_000) if QUICK else (10, 100, 1000, 10_000, 100_000)
series = run_series(model, sizes)
print_table(series)
big = series[max(sizes)]"""),
    code("""plot_convergence(big["first"], f"Накопленная частота: исход = грань первого касания (N = {len(big['first'])})")
plt.show()
plot_convergence(big["rest"], f"Накопленная частота: исход = грань остановки (N = {len(big['rest'])})")
plt.show()"""),
    md("## 5. Сравнение: модель p_i, реальный эксперимент, симуляция"),
    code("""plot_comparison(big)
plt.show()
n_phys = sum(PHYS_COUNTS.values())
for f in (1, 2, 3):
    lo, hi = wilson_ci(PHYS_COUNTS[f], n_phys)
    print(f"грань {f}: p̂_phys = {PHYS_COUNTS[f]/n_phys:.2f}, 95% ДИ [{lo:.3f}; {hi:.3f}]")
for name, p in [("модель p_i", P_MODEL), ("симуляция, первое касание", freqs_of(big["first"])),
                ("симуляция, грань остановки", freqs_of(big["rest"]))]:
    chi2, pv = chi2_test(PHYS_COUNTS, p)
    print(f"реальные данные vs {name}: χ² = {chi2:.2f}, p-value = {pv:.3f}")
# направленная гипотеза «грань 1 выпадает чаще» (заявлена до эксперимента)
p_one = binom_test_greater(PHYS_COUNTS[1], n_phys, P_MODEL[1])
print(f"грань 1, альтернатива p₁ > {P_MODEL[1]:.3f}: P(X ≥ {PHYS_COUNTS[1]}) = {p_one:.3f}")"""),
    md("""## 6. Как влияет «тяжёлая» грань

Перебираем дополнительную массу на грани 1. В полёте сила тяжести приложена к центру масс и не создаёт
момента, поэтому ориентация к моменту первого касания от неё почти не зависит. Тяжёлая грань начинает
«работать» только после удара — когда футляр подпрыгивает, кувыркается и укладывается."""),
    code("""if not QUICK:
    sw = mass_sweep(model, n=20_000)
    plot_mass_sweep(sw, n_sim=20_000)
    plt.show()"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
})
path = os.path.join(ROOT, "lab1_simulation.ipynb")
nbf.write(nb, path)
print("записан", path)
