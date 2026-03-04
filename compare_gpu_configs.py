#!/usr/bin/env python3
"""
=============================================================================
  COMPARACIÓN DE CONFIGURACIONES GPU: 1 GPU vs 2 GPU vs 4 GPU
  "Implementación de un Modelo de AutoML en la Nube y su Evaluación
   Utilizando una Base de Datos de Benchmark"
=============================================================================
  Genera:
   - Tablas comparativas (.tex + consola)
   - Figuras (.png)
   - Contexto LLM (.txt) para análisis automático
=============================================================================
"""

import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'comparison_gpu_tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPU_CONFIGS = {
    '1 GPU': [
        os.path.join(BASE_DIR, 'results_gpu_1'),
        os.path.join(BASE_DIR, 'results_gpu_1', 'results_multi'),
    ],
    '2 GPU': os.path.join(BASE_DIR, 'results_gpu_2'),
    '4 GPU': os.path.join(BASE_DIR, 'results_gpu_4', 'results_multi_4gpus'),
}

DATASET_DISPLAY = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10',
    'fashion_mnist': 'Fashion-MNIST',
    'cifar100': 'CIFAR-100',
}

COLORS_GPU = {
    '1 GPU': '#3498db',
    '2 GPU': '#e67e22',
    '4 GPU': '#e74c3c',
}

COLORS_DS = {
    'mnist': '#3498db',
    'cifar10': '#e74c3c',
    'fashion_mnist': '#2ecc71',
    'cifar100': '#9b59b6',
}

# ─── Robust JSON Loader ────────────────────────────────────────────────────

def _parse_json_robust(filepath):
    """Parse JSON files that may have trailing text (old format)."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            raw = open(filepath).read().strip()
            if not raw:
                return None
            dec = json.JSONDecoder()
            data, _ = dec.raw_decode(raw)
            return data
        except Exception:
            return None


def _convert_old_format(data, filename):
    """Convert old per-model format to unified experiment format."""
    mtr = data.get('model_training_request', {})
    perf = data.get('performance', 0)
    perf2 = data.get('performance_2', 0)
    if isinstance(perf, dict):
        perf = perf.get('accuracy', 0)
    if isinstance(perf2, dict):
        perf2 = perf2.get('accuracy', 0)

    arch_info = mtr.get('architecture', {})
    if isinstance(arch_info, dict):
        base_arch = arch_info.get('base_architecture', 'unknown')
    else:
        base_arch = str(arch_info) if arch_info else 'unknown'

    ds_name = mtr.get('dataset', None)
    if ds_name is None or not isinstance(ds_name, str):
        # Infer from filename
        basename = os.path.basename(filename).replace('.json', '')
        parts = basename.split('-')
        ds_name = parts[0] if parts else 'unknown'

    best = max(perf, perf2) if isinstance(perf2, (int, float)) else perf

    return {
        'experiment_id': os.path.basename(filename).replace('.json', ''),
        'dataset': ds_name,
        'leaderboard': [{'performance': best, 'architecture': base_arch, 'phase': 'exploration'}],
        'performance_metrics': {},
        'optimization_stats': {'models_processed': 1},
        '_is_old_format': True,
    }


# ─── Load All Experiments ──────────────────────────────────────────────────

def load_all_configs():
    """Load experiments from all GPU configurations.
    Returns dict: { '1 GPU': [experiments], '2 GPU': [...], '4 GPU': [...] }
    Only includes experiments with meaningful optimization (multiple models processed).
    """
    all_data = {}

    for label, dirpath in GPU_CONFIGS.items():
        experiments = []
        source_dirs = dirpath if isinstance(dirpath, (list, tuple)) else [dirpath]

        for src_dir in source_dirs:
            if not os.path.isdir(src_dir):
                print(f"  Warning: {src_dir} not found, skipping source for {label}")
                continue

            files = sorted(glob.glob(os.path.join(src_dir, '*.json')))
            files = [f for f in files
                     if os.path.isfile(f)
                     and '_state' not in os.path.basename(f)
                     and '_model_' not in os.path.basename(f)
                     and '_summary' not in os.path.basename(f)]

            for f in files:
                data = _parse_json_robust(f)
                if data is None:
                    continue

                # Convert old format
                if 'model_training_request' in data:
                    data = _convert_old_format(data, f)

                data['_filename'] = os.path.basename(f)
                data['_gpu_config'] = label
                experiments.append(data)

        all_data[label] = experiments
        print(f"  {label}: {len(experiments)} experimentos cargados desde {', '.join(source_dirs)}")

    return all_data


def get_dataset(exp):
    ds = exp.get('dataset', None)
    if isinstance(ds, dict):
        ds = ds.get('name', ds.get('tag', None))
    if ds is None or not isinstance(ds, str):
        eid = exp.get('experiment_id', None) or exp.get('_filename', '')
        eid = eid.replace('.json', '')
        parts = eid.split('-')
        ds = parts[0] if parts else 'unknown'
    return ds


def get_exp_id(exp):
    eid = exp.get('experiment_id', None) or exp.get('_filename', '')
    return eid.replace('.json', '') if eid else 'unknown'


def get_total_time(exp):
    pm = exp.get('performance_metrics', {})
    t = pm.get('total_time_seconds', None)
    if t is None:
        t = pm.get('elapsed_seconds', 0)
    return t or 0


def get_best_accuracy(exp):
    lb = exp.get('leaderboard', [])
    return max((m.get('performance', 0) for m in lb), default=0)


def get_models_processed(exp):
    stats = exp.get('optimization_stats', {})
    n = stats.get('models_processed', None)
    if n is None:
        pm = exp.get('performance_metrics', {})
        n = pm.get('models_processed', None)
    if n is None:
        n = len(exp.get('leaderboard', []))
    return n or 0


def filter_low_accuracy_outliers(all_data, min_ratio_to_median=0.90):
    """Filter out experiments with very low accuracy compared to median.

    The filter is applied per (dataset, GPU config). Any experiment with
    best_accuracy < median_accuracy * min_ratio_to_median is excluded.
    """
    medians = {}

    for label, exps in all_data.items():
        grouped_accs = defaultdict(list)
        for exp in exps:
            ds = get_dataset(exp)
            if ds in ('unknown', '?', 'grietas_baches'):
                continue
            best = get_best_accuracy(exp)
            t = get_total_time(exp)
            if best <= 0 and t <= 0:
                continue
            grouped_accs[ds].append(best)

        for ds, accs in grouped_accs.items():
            if accs:
                medians[(label, ds)] = float(np.median(accs))

    filtered = {}
    removed = defaultdict(int)

    for label, exps in all_data.items():
        keep = []
        for exp in exps:
            ds = get_dataset(exp)
            best = get_best_accuracy(exp)
            t = get_total_time(exp)

            # Keep unknown/empty entries for compatibility with existing logic;
            # they are filtered later in table generators.
            if ds in ('unknown', '?', 'grietas_baches') or (best <= 0 and t <= 0):
                keep.append(exp)
                continue

            median_acc = medians.get((label, ds), 0)
            threshold = median_acc * min_ratio_to_median

            if median_acc > 0 and best < threshold:
                removed[(label, ds)] += 1
                continue

            keep.append(exp)

        filtered[label] = keep

    total_removed = sum(removed.values())
    if total_removed > 0:
        print()
        print("  Filtro robusto aplicado (outliers de accuracy):")
        print(f"    Criterio: best_acc < mediana × {min_ratio_to_median:.2f}")
        print(f"    Experimentos excluidos: {total_removed}")
        for (label, ds), n in sorted(removed.items()):
            ds_name = DATASET_DISPLAY.get(ds, ds)
            print(f"      - {label} / {ds_name}: {n}")

    return filtered


def fmt_time(secs):
    if secs <= 0:
        return "N/A"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_latex(filename, content):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  LaTeX → {path}")


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C1: Resumen por Configuración GPU y Dataset
# ═══════════════════════════════════════════════════════════════════════════

def table_c1_resumen(all_data):
    """Best accuracy and time per GPU config per dataset."""
    print("=" * 90)
    print("  TABLA C1: Resumen por Configuración GPU y Dataset")
    print("=" * 90)

    # Aggregate: { dataset: { gpu_label: { 'accs': [...], 'times': [...], 'models': [...] } } }
    agg = defaultdict(lambda: defaultdict(lambda: {'accs': [], 'times': [], 'models': [], 'is_automl': []}))

    for label, exps in all_data.items():
        for exp in exps:
            ds = get_dataset(exp)
            if ds in ('unknown', '?', 'grietas_baches'):
                continue
            best = get_best_accuracy(exp)
            t = get_total_time(exp)
            n = get_models_processed(exp)
            # Filter out experiments with 0 accuracy and time (old broken ones)
            if best <= 0 and t <= 0:
                continue
            is_automl = not exp.get('_is_old_format', False)
            agg[ds][label]['accs'].append(best)
            agg[ds][label]['times'].append(t)
            agg[ds][label]['models'].append(n)
            agg[ds][label]['is_automl'].append(is_automl)

    header = f"  {'Dataset':<16} {'Config':<8} {'Runs':>5} {'Best Acc':>10} {'Mean Acc':>10} {'Mean Time':>12} {'Mod/h':>7}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
             "Dataset & Config & Runs & Best Acc & Mean Acc & Mean Time & Mod/h \\\\\n\\midrule\n")

    for ds in sorted(agg.keys()):
        for gpu_label in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(gpu_label)
            if not info or not info['accs']:
                continue
            accs = info['accs']
            times = info['times']
            models = info['models']
            best = max(accs)
            mean_acc = np.mean(accs)
            mean_time = np.mean([t for t in times if t > 0]) if any(t > 0 for t in times) else 0
            total_models = sum(models)
            total_hours = sum(t for t in times if t > 0) / 3600.0
            throughput = total_models / total_hours if total_hours > 0 else 0
            ds_name = DATASET_DISPLAY.get(ds, ds)
            print(f"  {ds_name:<16} {gpu_label:<8} {len(accs):>5} {best:>10.4f} {mean_acc:>10.4f} {fmt_time(mean_time):>12} {throughput:>7.1f}")
            latex += f"{ds_name} & {gpu_label} & {len(accs)} & {best:.4f} & {mean_acc:.4f} & {fmt_time(mean_time)} & {throughput:.1f} \\\\\n"
        print()  # separator between datasets

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac1_resumen_gpu_dataset.tex', latex)
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C2: Mejor Accuracy por Dataset × GPU Config (Tabla Cruzada)
# ═══════════════════════════════════════════════════════════════════════════

def table_c2_accuracy_cruzada(agg):
    print("=" * 90)
    print("  TABLA C2: Mejor Accuracy — Tabla Cruzada Dataset × GPUs")
    print("=" * 90)

    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    header = f"  {'Dataset':<16}" + "".join(f" {g:>10}" for g in gpu_labels) + f" {'Δ(4-1)':>10}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = "\\begin{tabular}{l" + "r" * (len(gpu_labels) + 1) + "}\n\\toprule\n"
    latex += "Dataset & " + " & ".join(gpu_labels) + " & $\\Delta$(4-1) \\\\\n\\midrule\n"

    for ds in datasets:
        row = f"  {DATASET_DISPLAY.get(ds, ds):<16}"
        vals = {}
        for g in gpu_labels:
            info = agg[ds].get(g)
            if info and info['accs']:
                best = max(info['accs'])
                vals[g] = best
                row += f" {best:>10.4f}"
            else:
                row += f" {'—':>10}"
        # Delta 4GPU - 1GPU
        if '4 GPU' in vals and '1 GPU' in vals:
            delta = vals['4 GPU'] - vals['1 GPU']
            sign = '+' if delta >= 0 else ''
            row += f" {sign}{delta:>9.4f}"
            delta_str = f"{sign}{delta:.4f}"
        else:
            row += f" {'—':>10}"
            delta_str = "—"
        print(row)

        latex_row = DATASET_DISPLAY.get(ds, ds)
        for g in gpu_labels:
            if g in vals:
                latex_row += f" & {vals[g]:.4f}"
            else:
                latex_row += " & —"
        latex_row += f" & {delta_str} \\\\\n"
        latex += latex_row

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac2_accuracy_cruzada.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C3: Tiempo Promedio por Dataset × GPU Config
# ═══════════════════════════════════════════════════════════════════════════

def table_c3_tiempo_cruzada(agg):
    print("=" * 90)
    print("  TABLA C3: Tiempo Promedio — Tabla Cruzada Dataset × GPUs")
    print("=" * 90)

    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    header = f"  {'Dataset':<16}" + "".join(f" {g:>12}" for g in gpu_labels) + f" {'Speedup':>10}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = "\\begin{tabular}{l" + "r" * (len(gpu_labels) + 1) + "}\n\\toprule\n"
    latex += "Dataset & " + " & ".join(gpu_labels) + " & Speedup \\\\\n\\midrule\n"

    for ds in datasets:
        row = f"  {DATASET_DISPLAY.get(ds, ds):<16}"
        vals = {}
        for g in gpu_labels:
            info = agg[ds].get(g)
            if info and info['times']:
                valid_times = [t for t in info['times'] if t > 0]
                if valid_times:
                    mean_t = np.mean(valid_times)
                    vals[g] = mean_t
                    row += f" {fmt_time(mean_t):>12}"
                else:
                    row += f" {'—':>12}"
            else:
                row += f" {'—':>12}"

        # Speedup = time_1gpu / time_4gpu
        if '1 GPU' in vals and '4 GPU' in vals and vals['4 GPU'] > 0:
            speedup = vals['1 GPU'] / vals['4 GPU']
            row += f" {speedup:>9.2f}×"
            speedup_str = f"{speedup:.2f}$\\times$"
        elif '1 GPU' in vals and '2 GPU' in vals and vals['2 GPU'] > 0:
            speedup = vals['1 GPU'] / vals['2 GPU']
            row += f" {speedup:>9.2f}× (vs 2)"
            speedup_str = f"{speedup:.2f}$\\times$ (vs 2)"
        else:
            row += f" {'—':>10}"
            speedup_str = "—"
        print(row)

        latex_row = DATASET_DISPLAY.get(ds, ds)
        for g in gpu_labels:
            if g in vals:
                latex_row += f" & {fmt_time(vals[g])}"
            else:
                latex_row += " & —"
        latex_row += f" & {speedup_str} \\\\\n"
        latex += latex_row

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac3_tiempo_cruzada.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C4: Throughput (modelos/hora) por configuración
# ═══════════════════════════════════════════════════════════════════════════

def table_c4_throughput(agg):
    print("=" * 90)
    print("  TABLA C4: Throughput (modelos/hora) por Configuración GPU")
    print("=" * 90)

    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    header = f"  {'Dataset':<16}" + "".join(f" {g:>10}" for g in gpu_labels) + f" {'Δ(4/1)':>10}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = "\\begin{tabular}{l" + "r" * (len(gpu_labels) + 1) + "}\n\\toprule\n"
    latex += "Dataset & " + " & ".join(gpu_labels) + " & $\\Delta$(4/1) \\\\\n\\midrule\n"

    for ds in datasets:
        row = f"  {DATASET_DISPLAY.get(ds, ds):<16}"
        vals = {}
        for g in gpu_labels:
            info = agg[ds].get(g)
            if info and info['times'] and info['models']:
                valid_times = [t for t in info['times'] if t > 0]
                total_h = sum(valid_times) / 3600.0
                total_m = sum(info['models'])
                if total_h > 0:
                    tp = total_m / total_h
                    vals[g] = tp
                    row += f" {tp:>10.1f}"
                else:
                    row += f" {'—':>10}"
            else:
                row += f" {'—':>10}"

        if '4 GPU' in vals and '1 GPU' in vals and vals['1 GPU'] > 0:
            ratio = vals['4 GPU'] / vals['1 GPU']
            row += f" {ratio:>9.2f}×"
            ratio_str = f"{ratio:.2f}$\\times$"
        else:
            row += f" {'—':>10}"
            ratio_str = "—"
        print(row)

        latex_row = DATASET_DISPLAY.get(ds, ds)
        for g in gpu_labels:
            if g in vals:
                latex_row += f" & {vals[g]:.1f}"
            else:
                latex_row += " & —"
        latex_row += f" & {ratio_str} \\\\\n"
        latex += latex_row

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac4_throughput.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C5: Eficiencia de Escalabilidad (Scaling Efficiency)
# ═══════════════════════════════════════════════════════════════════════════

def table_c5_scaling_efficiency(agg):
    print("=" * 90)
    print("  TABLA C5: Eficiencia de Escalabilidad GPU")
    print("=" * 90)
    print("  (Eficiencia = Speedup / N_GPUs × 100%)")
    print("  (Ideal = 100%, >80% = buena, >60% = aceptable)")
    print()

    datasets = sorted(agg.keys())
    gpu_labels = ['2 GPU', '4 GPU']

    header = f"  {'Dataset':<16} {'Ref 1GPU':>10}" + "".join(f" {'Speedup '+g:>14} {'Efic. '+g:>10}" for g in gpu_labels)
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = ("\\begin{tabular}{lrrrrr}\n\\toprule\n"
             "Dataset & T(1 GPU) & Speedup(2) & Efic.(2) & Speedup(4) & Efic.(4) \\\\\n\\midrule\n")

    for ds in datasets:
        info_1 = agg[ds].get('1 GPU')
        if not info_1 or not info_1['times']:
            continue
        valid_1 = [t for t in info_1['times'] if t > 0]
        if not valid_1:
            continue
        t1 = np.mean(valid_1)

        row = f"  {DATASET_DISPLAY.get(ds, ds):<16} {fmt_time(t1):>10}"
        latex_row = f"{DATASET_DISPLAY.get(ds, ds)} & {fmt_time(t1)}"

        for g, n_gpus in [('2 GPU', 2), ('4 GPU', 4)]:
            info_g = agg[ds].get(g)
            if info_g and info_g['times']:
                valid_g = [t for t in info_g['times'] if t > 0]
                if valid_g:
                    tg = np.mean(valid_g)
                    speedup = t1 / tg
                    efficiency = (speedup / n_gpus) * 100
                    row += f" {speedup:>13.2f}× {efficiency:>9.1f}%"
                    latex_row += f" & {speedup:.2f}$\\times$ & {efficiency:.1f}\\%"
                else:
                    row += f" {'—':>14} {'—':>10}"
                    latex_row += " & — & —"
            else:
                row += f" {'—':>14} {'—':>10}"
                latex_row += " & — & —"

        print(row)
        latex += latex_row + " \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac5_scaling_efficiency.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C5b: Comparación Justa 2 GPU vs 4 GPU (misma config AutoML)
# ═══════════════════════════════════════════════════════════════════════════

def table_c5b_fair_comparison(agg):
    """Only compare 2 GPU vs 4 GPU (both are full AutoML runs with same config)."""
    print("=" * 90)
    print("  TABLA C5b: Comparación Directa 2 GPU vs 4 GPU (misma configuración AutoML)")
    print("=" * 90)
    print("  Nota: Solo runs AutoML completos (expl + deep training, 8 modelos)")
    print()

    datasets = sorted(agg.keys())
    gpu_labels = ['2 GPU', '4 GPU']

    header = f"  {'Dataset':<16} {'Métrica':<20}" + "".join(f" {g:>12}" for g in gpu_labels) + f" {'Δ':>12}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = ("\\begin{tabular}{llrrr}\n\\toprule\n"
             "Dataset & Métrica & 2 GPU & 4 GPU & $\\Delta$ \\\\\n\\midrule\n")

    for ds in datasets:
        info_2 = agg[ds].get('2 GPU')
        info_4 = agg[ds].get('4 GPU')
        if not info_2 or not info_2['accs'] or not info_4 or not info_4['accs']:
            continue

        ds_name = DATASET_DISPLAY.get(ds, ds)
        best_2 = max(info_2['accs'])
        best_4 = max(info_4['accs'])
        mean_2 = np.mean(info_2['accs'])
        mean_4 = np.mean(info_4['accs'])

        valid_t2 = [t for t in info_2['times'] if t > 0]
        valid_t4 = [t for t in info_4['times'] if t > 0]
        mt2 = np.mean(valid_t2) if valid_t2 else 0
        mt4 = np.mean(valid_t4) if valid_t4 else 0

        tp2 = sum(info_2['models']) / (sum(valid_t2) / 3600) if valid_t2 and sum(valid_t2) > 0 else 0
        tp4 = sum(info_4['models']) / (sum(valid_t4) / 3600) if valid_t4 and sum(valid_t4) > 0 else 0

        # Best accuracy
        d_acc = best_4 - best_2
        sign = '+' if d_acc >= 0 else ''
        print(f"  {ds_name:<16} {'Best Accuracy':<20} {best_2:>12.4f} {best_4:>12.4f} {sign}{d_acc:>11.4f}")
        latex += f"{ds_name} & Best Acc & {best_2:.4f} & {best_4:.4f} & {sign}{d_acc:.4f} \\\\\n"

        # Mean accuracy
        d_mean = mean_4 - mean_2
        sign = '+' if d_mean >= 0 else ''
        print(f"  {'':<16} {'Mean Accuracy':<20} {mean_2:>12.4f} {mean_4:>12.4f} {sign}{d_mean:>11.4f}")
        latex += f" & Mean Acc & {mean_2:.4f} & {mean_4:.4f} & {sign}{d_mean:.4f} \\\\\n"

        # Time
        if mt2 > 0 and mt4 > 0:
            speedup = mt2 / mt4
            print(f"  {'':<16} {'Tiempo Promedio':<20} {fmt_time(mt2):>12} {fmt_time(mt4):>12} {speedup:>10.2f}× faster")
            latex += f" & Tiempo & {fmt_time(mt2)} & {fmt_time(mt4)} & {speedup:.2f}$\\times$ \\\\\n"

        # Throughput
        if tp2 > 0 and tp4 > 0:
            ratio = tp4 / tp2
            print(f"  {'':<16} {'Throughput (mod/h)':<20} {tp2:>12.1f} {tp4:>12.1f} {ratio:>10.2f}×")
            latex += f" & Throughput & {tp2:.1f} & {tp4:.1f} & {ratio:.2f}$\\times$ \\\\\n"

        latex += "\\midrule\n"
        print()

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac5b_fair_2vs4.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# TABLA C6: Detalle Individual de Todos los Experimentos
# ═══════════════════════════════════════════════════════════════════════════

def table_c6_detalle(all_data):
    print("=" * 100)
    print("  TABLA C6: Detalle Individual — Todos los Experimentos")
    print("=" * 100)

    header = f"  {'Experimento':<35} {'Config':<7} {'Dataset':<16} {'Best Acc':>10} {'Time':>10} {'Models':>7} {'T/Mod':>8}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    latex = ("\\begin{tabular}{llllrrrr}\n\\toprule\n"
             "Experimento & Config & Dataset & Best Acc & Tiempo & Modelos & T/Modelo \\\\\n\\midrule\n")

    for label in ['1 GPU', '2 GPU', '4 GPU']:
        exps = all_data.get(label, [])
        for exp in exps:
            ds = get_dataset(exp)
            if ds in ('unknown', '?', 'grietas_baches'):
                continue
            best = get_best_accuracy(exp)
            t = get_total_time(exp)
            n = get_models_processed(exp)
            if best <= 0 and t <= 0:
                continue
            eid = get_exp_id(exp)
            t_per_model = t / n if n > 0 and t > 0 else 0
            ds_name = DATASET_DISPLAY.get(ds, ds)
            print(f"  {eid:<35} {label:<7} {ds_name:<16} {best:>10.4f} {fmt_time(t):>10} {n:>7} {t_per_model:>7.0f}s")
            latex += f"{eid} & {label} & {ds_name} & {best:.4f} & {fmt_time(t)} & {n} & {t_per_model:.0f}s \\\\\n"
        if exps:
            latex += "\\midrule\n"

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tablac6_detalle_todos.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════════════════

def fig_c1_accuracy_comparison(agg):
    """Bar chart: Best accuracy per dataset, grouped by GPU config."""
    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25

    for i, g in enumerate(gpu_labels):
        vals = []
        for ds in datasets:
            info = agg[ds].get(g)
            if info and info['accs']:
                vals.append(max(info['accs']))
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=g, color=COLORS_GPU[g], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Best Accuracy', fontsize=12)
    ax.set_title('Mejor Accuracy por Dataset y Configuración GPU', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.9, 1.005)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    fig.tight_layout()
    save_figure(fig, 'figc1_accuracy_comparison.png')


def fig_c2_time_comparison(agg):
    """Bar chart: Mean time per dataset, grouped by GPU config."""
    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25

    for i, g in enumerate(gpu_labels):
        vals = []
        for ds in datasets:
            info = agg[ds].get(g)
            if info and info['times']:
                valid = [t for t in info['times'] if t > 0]
                vals.append(np.mean(valid) / 60 if valid else 0)  # minutes
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=g, color=COLORS_GPU[g], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{v:.0f}m', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Tiempo Promedio (minutos)', fontsize=12)
    ax.set_title('Tiempo Promedio por Dataset y Configuración GPU', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'figc2_time_comparison.png')


def fig_c3_throughput_comparison(agg):
    """Bar chart: Throughput (models/hour) per dataset, grouped by GPU config."""
    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25

    for i, g in enumerate(gpu_labels):
        vals = []
        for ds in datasets:
            info = agg[ds].get(g)
            if info and info['times'] and info['models']:
                valid_t = [t for t in info['times'] if t > 0]
                total_h = sum(valid_t) / 3600.0
                total_m = sum(info['models'])
                vals.append(total_m / total_h if total_h > 0 else 0)
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=g, color=COLORS_GPU[g], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Modelos / Hora', fontsize=12)
    ax.set_title('Throughput por Dataset y Configuración GPU', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'figc3_throughput_comparison.png')


def fig_c4_accuracy_vs_time_scatter(agg):
    """Scatter plot: Best accuracy vs mean time, colored by GPU config, shaped by dataset."""
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {'mnist': 'o', 'cifar10': 's', 'fashion_mnist': '^', 'cifar100': 'D'}

    for ds in sorted(agg.keys()):
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if not info or not info['accs']:
                continue
            for acc, t in zip(info['accs'], info['times']):
                if acc <= 0 or t <= 0:
                    continue
                ax.scatter(t / 60, acc, c=COLORS_GPU[g], marker=markers.get(ds, 'o'),
                           s=120, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Legend for GPU configs (color)
    for g in ['1 GPU', '2 GPU', '4 GPU']:
        ax.scatter([], [], c=COLORS_GPU[g], label=g, s=80, edgecolors='black')
    # Legend for datasets (shape)
    for ds, mk in markers.items():
        if ds in agg:
            ax.scatter([], [], c='gray', marker=mk, label=DATASET_DISPLAY.get(ds, ds), s=80, edgecolors='black')

    ax.set_xlabel('Tiempo (minutos)', fontsize=12)
    ax.set_ylabel('Best Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Tiempo: Comparación de Configuraciones GPU', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'figc4_accuracy_vs_time.png')


def fig_c5_speedup_bars(agg):
    """Horizontal bar chart showing speedup ratio per dataset relative to 1 GPU."""
    datasets_with_both = [ds for ds in sorted(agg.keys())
                          if agg[ds].get('1 GPU') and agg[ds]['1 GPU']['times']
                          and any(t > 0 for t in agg[ds]['1 GPU']['times'])]

    if not datasets_with_both:
        print("  No speedup data available (need 1 GPU baseline).")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(datasets_with_both) * 1.5)))
    y_pos = np.arange(len(datasets_with_both))
    bar_height = 0.35

    for i, (g, n_gpus) in enumerate([('2 GPU', 2), ('4 GPU', 4)]):
        speedups = []
        for ds in datasets_with_both:
            t1 = np.mean([t for t in agg[ds]['1 GPU']['times'] if t > 0])
            info_g = agg[ds].get(g)
            if info_g and info_g['times']:
                valid_g = [t for t in info_g['times'] if t > 0]
                if valid_g:
                    tg = np.mean(valid_g)
                    speedups.append(t1 / tg)
                else:
                    speedups.append(0)
            else:
                speedups.append(0)

        bars = ax.barh(y_pos + i * bar_height, speedups, bar_height,
                       label=f'{g} (ideal={n_gpus}×)', color=COLORS_GPU[g], alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, speedups):
            if v > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f'{v:.2f}×', ha='left', va='center', fontsize=9, fontweight='bold')

    # Ideal lines
    ax.axvline(x=2, color=COLORS_GPU['2 GPU'], linestyle='--', alpha=0.4, label='Ideal 2×')
    ax.axvline(x=4, color=COLORS_GPU['4 GPU'], linestyle='--', alpha=0.4, label='Ideal 4×')

    ax.set_yticks(y_pos + bar_height / 2)
    ax.set_yticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets_with_both], fontsize=11)
    ax.set_xlabel('Speedup (×)', fontsize=12)
    ax.set_title('Speedup vs 1 GPU por Dataset', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'figc5_speedup.png')


def fig_c6_cost_efficiency(agg):
    """Time per model comparison across GPU configs (efficiency)."""
    datasets = sorted(agg.keys())
    gpu_labels = ['1 GPU', '2 GPU', '4 GPU']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25

    for i, g in enumerate(gpu_labels):
        vals = []
        for ds in datasets:
            info = agg[ds].get(g)
            if info and info['times'] and info['models']:
                valid_t = [t for t in info['times'] if t > 0]
                total_t = sum(valid_t)
                total_m = sum(info['models'])
                vals.append(total_t / total_m / 60 if total_m > 0 else 0)  # min per model
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=g, color=COLORS_GPU[g], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{v:.1f}m', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Minutos / Modelo', fontsize=12)
    ax.set_title('Tiempo por Modelo por Dataset y Configuración GPU', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'figc6_cost_per_model.png')


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXTO LLM COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_context(all_data, agg):
    """Generate plain-text context file for LLM analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("  CONTEXTO PARA ANÁLISIS LLM")
    lines.append("  Comparación de Configuraciones GPU: 1 GPU vs 2 GPU vs 4 GPU")
    lines.append("  AutoML en la Nube — Tesis")
    lines.append("=" * 80)
    lines.append("")

    # ─── Section 1: Overview ────────────────────────────────────────────
    lines.append("─── SECCIÓN 1: INVENTARIO DE EXPERIMENTOS ───")
    lines.append("")
    for label in ['1 GPU', '2 GPU', '4 GPU']:
        exps = all_data.get(label, [])
        valid = [e for e in exps if get_best_accuracy(e) > 0 or get_total_time(e) > 0]
        datasets_used = set(get_dataset(e) for e in valid if get_dataset(e) not in ('unknown', '?'))
        lines.append(f"  {label}:")
        lines.append(f"    Total experimentos válidos: {len(valid)}")
        lines.append(f"    Datasets: {', '.join(sorted(datasets_used))}")
        for exp in valid:
            ds = get_dataset(exp)
            best = get_best_accuracy(exp)
            t = get_total_time(exp)
            n = get_models_processed(exp)
            eid = get_exp_id(exp)
            lines.append(f"    {eid:40s} ds={ds:15s} acc={best:.4f} time={fmt_time(t):>10} models={n}")
        lines.append("")

    # ─── Section 2: Accuracy Comparison ─────────────────────────────────
    lines.append("─── SECCIÓN 2: COMPARACIÓN DE ACCURACY ───")
    lines.append("")
    for ds in sorted(agg.keys()):
        ds_name = DATASET_DISPLAY.get(ds, ds)
        lines.append(f"  Dataset: {ds_name}")
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if info and info['accs']:
                best = max(info['accs'])
                mean = np.mean(info['accs'])
                std = np.std(info['accs'])
                lines.append(f"    {g}: best={best:.4f}  mean={mean:.4f}  std={std:.4f}  runs={len(info['accs'])}")
            else:
                lines.append(f"    {g}: sin datos")
        lines.append("")

    # ─── Section 3: Time Comparison ─────────────────────────────────────
    lines.append("─── SECCIÓN 3: COMPARACIÓN DE TIEMPOS ───")
    lines.append("")
    for ds in sorted(agg.keys()):
        ds_name = DATASET_DISPLAY.get(ds, ds)
        lines.append(f"  Dataset: {ds_name}")
        t_ref = None
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if info and info['times']:
                valid = [t for t in info['times'] if t > 0]
                if valid:
                    mean_t = np.mean(valid)
                    if g == '1 GPU':
                        t_ref = mean_t
                    speedup_str = ""
                    if t_ref and mean_t > 0:
                        speedup = t_ref / mean_t
                        eff_n = int(g.split()[0])
                        eff = (speedup / eff_n) * 100 if eff_n > 1 else 100
                        speedup_str = f"  speedup={speedup:.2f}× efficiency={eff:.1f}%"
                    lines.append(f"    {g}: mean={fmt_time(mean_t)}  ({mean_t:.0f}s){speedup_str}")
                else:
                    lines.append(f"    {g}: sin tiempos válidos")
            else:
                lines.append(f"    {g}: sin datos")
        lines.append("")

    # ─── Section 4: Throughput ──────────────────────────────────────────
    lines.append("─── SECCIÓN 4: THROUGHPUT (MODELOS/HORA) ───")
    lines.append("")
    for ds in sorted(agg.keys()):
        ds_name = DATASET_DISPLAY.get(ds, ds)
        lines.append(f"  Dataset: {ds_name}")
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if info and info['times'] and info['models']:
                valid_t = [t for t in info['times'] if t > 0]
                total_h = sum(valid_t) / 3600.0
                total_m = sum(info['models'])
                tp = total_m / total_h if total_h > 0 else 0
                t_per_model = sum(valid_t) / total_m / 60 if total_m > 0 else 0
                lines.append(f"    {g}: {tp:.1f} mod/h  ({t_per_model:.1f} min/modelo)  total_modelos={total_m}")
            else:
                lines.append(f"    {g}: sin datos")
        lines.append("")

    # ─── Section 5: Key Findings ────────────────────────────────────────
    lines.append("─── SECCIÓN 5: HALLAZGOS CLAVE PARA LA TESIS ───")
    lines.append("")

    # Find best accuracy across all configs
    overall_best = {}
    for ds in agg:
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if info and info['accs']:
                best = max(info['accs'])
                if ds not in overall_best or best > overall_best[ds][1]:
                    overall_best[ds] = (g, best)

    lines.append("  Mejor accuracy absoluta por dataset:")
    for ds in sorted(overall_best.keys()):
        g, best = overall_best[ds]
        lines.append(f"    {DATASET_DISPLAY.get(ds, ds)}: {best:.4f} ({g})")
    lines.append("")

    # Scalability summary
    lines.append("  Análisis de escalabilidad:")
    for ds in sorted(agg.keys()):
        info_1 = agg[ds].get('1 GPU')
        if not info_1 or not info_1['times']:
            continue
        valid_1 = [t for t in info_1['times'] if t > 0]
        if not valid_1:
            continue
        t1 = np.mean(valid_1)
        for g, n_gpus in [('2 GPU', 2), ('4 GPU', 4)]:
            info_g = agg[ds].get(g)
            if info_g and info_g['times']:
                valid_g = [t for t in info_g['times'] if t > 0]
                if valid_g:
                    tg = np.mean(valid_g)
                    speedup = t1 / tg
                    eff = (speedup / n_gpus) * 100
                    qualifier = "excelente" if eff > 80 else "buena" if eff > 60 else "moderada" if eff > 40 else "baja"
                    lines.append(f"    {DATASET_DISPLAY.get(ds, ds)} {g}: speedup={speedup:.2f}× efficiency={eff:.1f}% ({qualifier})")
    lines.append("")

    # Accuracy vs GPUs
    lines.append("  ¿Más GPUs = mejor accuracy?")
    for ds in sorted(agg.keys()):
        accs_by_g = {}
        for g in ['1 GPU', '2 GPU', '4 GPU']:
            info = agg[ds].get(g)
            if info and info['accs']:
                accs_by_g[g] = max(info['accs'])
        if len(accs_by_g) >= 2:
            sorted_g = sorted(accs_by_g.items(), key=lambda x: x[1], reverse=True)
            lines.append(f"    {DATASET_DISPLAY.get(ds, ds)}: " +
                         " > ".join(f"{g}({a:.4f})" for g, a in sorted_g))
    lines.append("")

    lines.append("  Observaciones importantes para tesis:")
    lines.append("    - NOTA: Los experimentos de 1 GPU (old format) son modelos individuales, NO corridas AutoML completas")
    lines.append("      Los de 2 y 4 GPU son corridas AutoML con exploración+deep training (8 modelos cada una)")
    lines.append("      Para comparar accuracy de forma justa, usar solo los runs con misma configuración AutoML")
    lines.append("    - La comparación de TIEMPO es más justa entre 2 GPU y 4 GPU (misma config AutoML)")
    lines.append("    - La accuracy NO necesariamente mejora con más GPUs (misma exploración, diferente paralelismo)")
    lines.append("    - El throughput (modelos/hora) es la métrica principal de escalabilidad")
    lines.append("    - El speedup sub-lineal es esperado por overhead de comunicación RabbitMQ")
    lines.append("    - El tiempo total se reduce, permitiendo explorar más arquitecturas en menos tiempo")
    lines.append("")

    # ─── Section 6: Suggested Analyses ──────────────────────────────────
    lines.append("─── SECCIÓN 6: ANÁLISIS SUGERIDOS ───")
    lines.append("")
    lines.append("  1. Gráfico de scaling efficiency: qué tan cerca del speedup ideal (lineal)")
    lines.append("  2. Costo-beneficio: si se renta N GPUs en la nube, ¿cuánto se ahorra en tiempo?")
    lines.append("  3. Impacto en convergencia: ¿la paralelización afecta la calidad de la búsqueda NAS?")
    lines.append("  4. Overhead de comunicación: diferencia entre speedup ideal y real")
    lines.append("  5. Reproducibilidad: varianza entre runs con misma configuración")
    lines.append("")

    content = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, 'llm_context_comparison.txt')
    with open(path, 'w') as f:
        f.write(content)
    size = os.path.getsize(path)
    print(f"  CONTEXTO LLM guardado en: {path}")
    print(f"  Tamaño: {size:,} bytes")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  COMPARACIÓN DE CONFIGURACIONES GPU: 1 GPU vs 2 GPU vs 4 GPU")
    print("  AutoML en la Nube — Tablas para Tesis")
    print("=" * 80)
    print()

    all_data = load_all_configs()
    all_data = filter_low_accuracy_outliers(all_data, min_ratio_to_median=0.90)
    print()

    agg = table_c1_resumen(all_data)
    print()
    table_c2_accuracy_cruzada(agg)
    print()
    table_c3_tiempo_cruzada(agg)
    print()
    table_c4_throughput(agg)
    print()
    table_c5_scaling_efficiency(agg)
    print()
    table_c5b_fair_comparison(agg)
    print()
    table_c6_detalle(all_data)
    print()

    # Figures
    print("=" * 80)
    print("  GENERANDO FIGURAS")
    print("=" * 80)
    fig_c1_accuracy_comparison(agg)
    fig_c2_time_comparison(agg)
    fig_c3_throughput_comparison(agg)
    fig_c4_accuracy_vs_time_scatter(agg)
    fig_c5_speedup_bars(agg)
    fig_c6_cost_efficiency(agg)

    # LLM Context
    print()
    print("=" * 80)
    print("  GENERANDO CONTEXTO LLM")
    print("=" * 80)
    generate_llm_context(all_data, agg)

    # Summary
    print()
    print("=" * 80)
    print(f"  COMPLETADO — Archivos generados en: {OUTPUT_DIR}")
    print("=" * 80)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        print(f"  {f:<55} {os.path.getsize(fpath):>8} bytes")


if __name__ == '__main__':
    main()
