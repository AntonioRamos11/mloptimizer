#!/usr/bin/env python3
"""
=============================================================================
 GENERADOR COMPLETO DE TABLAS PARA TESIS (con Hardware Metrics)
 "Implementación de un Modelo de AutoML en la Nube y su Evaluación
  Utilizando una Base de Datos de Benchmark"
=============================================================================
 Fuentes de datos:
   - *.json              → Resultados de experimentos AutoML
   - hardware_metrics/   → GPU util, temp, power, memory, idle time, latencia
   - hardware_performance_logs/ → Tiempos de entrenamiento, build, epoch details
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
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'thesis_tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try both with and without .json suffix for directory names
_hm1 = os.path.join(RESULTS_DIR, 'hardware_metrics')
_hm2 = os.path.join(RESULTS_DIR, 'hardware_metrics.json')
HW_METRICS_DIR = _hm1 if os.path.isdir(_hm1) else _hm2

_hp1 = os.path.join(RESULTS_DIR, 'hardware_performance_logs')
_hp2 = os.path.join(RESULTS_DIR, 'hardware_performance_logs.json')
HW_PERF_DIR = _hp1 if os.path.isdir(_hp1) else _hp2

DATASET_DISPLAY = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10',
    'fashion_mnist': 'Fashion-MNIST',
    'grietas_baches': 'Grietas/Baches',
}

COLORS = {
    'exploration': '#3498db',
    'deep_training': '#e74c3c',
    'cnn': '#2ecc71',
    'inception': '#9b59b6',
    'gpu_util': '#f39c12',
    'gpu_mem': '#1abc9c',
    'power': '#e67e22',
    'temp': '#c0392b',
    'idle': '#95a5a6',
}

# ─── Loaders ────────────────────────────────────────────────────────────────

def _parse_json_robust(filepath):
    """Parse a JSON file that may have trailing text or concatenated objects."""
    with open(filepath) as fh:
        raw = fh.read().strip()
    if not raw:
        return None
    # First try normal parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Use raw_decode to parse just the first JSON object
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        return None


def _convert_old_format_to_leaderboard(model_results, exp_id, filename):
    """Convert old per-model result dicts into a unified experiment dict.
    
    Old format: {model_training_request: {id, architecture, ...}, performance, performance_2}
    """
    leaderboard = []
    for mr in model_results:
        req = mr.get('model_training_request', {})
        arch = req.get('architecture', {})
        base_arch = arch.get('base_architecture', '?')
        # Determine classifier type
        clf = arch.get('classifier_layer_type', '?')
        phase = 'deep_training' if req.get('is_partial_training') is False else 'exploration'
        training_type = req.get('training_type', 1)
        if training_type == 2:
            phase = 'deep_training'
        elif training_type == 1:
            phase = 'exploration'
        leaderboard.append({
            'id': req.get('id', 0),
            'phase': phase,
            'performance': mr.get('performance', 0),
            'performance_2': mr.get('performance_2', 0),
            'base_architecture': base_arch,
            'classifier_type': clf,
        })
    
    # Build a unified experiment dict
    dataset_tag = None
    if model_results:
        req0 = model_results[0].get('model_training_request', {})
        dataset_tag = req0.get('dataset_tag', None)
    
    n_models = len(leaderboard)
    return {
        'experiment_id': exp_id,
        'dataset': dataset_tag,
        'leaderboard': leaderboard,
        'optimization_stats': {
            'models_generated': n_models,
            'models_processed': n_models,
            'exploration_models': sum(1 for m in leaderboard if m['phase'] == 'exploration'),
            'deep_training_models': sum(1 for m in leaderboard if m['phase'] == 'deep_training'),
        },
        'performance_metrics': {},
        '_filename': filename,
    }


def load_experiments():
    """Load all experiment JSON files, handling multiple formats:
    - New format: {experiment_id, leaderboard, optimization_stats, ...}
    - Summary format: {model_info, dataset_ranges, performance_metrics, optimization_stats}
    - Old format: {model_training_request, performance} with trailing text
    - Summary-only (_summary.json): skip these, they're metadata
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')))
    # Group old-format per-model results by experiment_id
    old_format_models = defaultdict(list)  # exp_id -> [model_result, ...]
    experiments = []
    skipped_summaries = []

    for f in files:
        basename = os.path.basename(f)
        # Skip directories that end in .json
        if os.path.isdir(f):
            continue
        # Skip _summary.json files (we'll merge their data later)
        if '_summary.json' in basename:
            skipped_summaries.append(f)
            continue
        # Skip _model_N.json files — they belong to a specific experiment
        if '_model_' in basename:
            # These are per-model files from new experiment format
            continue

        data = _parse_json_robust(f)
        if data is None:
            print(f"  Warning: Could not parse {f}")
            continue

        # Detect format
        if 'model_training_request' in data:
            # Old per-model format — group by experiment_id
            req = data.get('model_training_request', {})
            exp_id = req.get('experiment_id', basename.replace('.json', ''))
            old_format_models[exp_id].append(data)
        else:
            # New or summary format
            data['_filename'] = basename
            experiments.append(data)

    # Convert old-format grouped models into experiment dicts
    for exp_id, model_list in sorted(old_format_models.items()):
        exp = _convert_old_format_to_leaderboard(model_list, exp_id, exp_id + '.json')
        experiments.append(exp)

    # Try to enrich experiments that have no leaderboard with _model_N.json files
    model_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_model_*.json')))
    model_by_exp = defaultdict(list)
    for mf in model_files:
        data = _parse_json_robust(mf)
        if data and 'model_training_request' in data:
            req = data['model_training_request']
            exp_id = req.get('experiment_id', '')
            model_by_exp[exp_id].append(data)

    # For experiments that have no leaderboard, check if we have _model_ files
    for exp in experiments:
        eid = get_exp_id(exp)
        if not exp.get('leaderboard') and eid in model_by_exp:
            converted = _convert_old_format_to_leaderboard(model_by_exp[eid], eid, exp.get('_filename', ''))
            exp['leaderboard'] = converted['leaderboard']
            if not exp.get('optimization_stats'):
                exp['optimization_stats'] = converted['optimization_stats']

    # Also check _summary.json files to enrich experiments
    for sf in skipped_summaries:
        sdata = _parse_json_robust(sf)
        if not sdata:
            continue
        s_eid = sdata.get('experiment_id', '')
        # Find matching experiment
        for exp in experiments:
            if get_exp_id(exp) == s_eid:
                # Merge top_5_models into leaderboard if leaderboard is empty
                if not exp.get('leaderboard') and sdata.get('top_5_models'):
                    exp['leaderboard'] = [{
                        'id': m.get('id', 0),
                        'phase': m.get('type', '?'),
                        'performance': m.get('performance', 0),
                        'performance_2': m.get('performance_2', 0),
                        'base_architecture': m.get('architecture', {}).get('base_architecture', '?'),
                        'classifier_type': m.get('architecture', {}).get('classifier_layer_type', '?'),
                    } for m in sdata['top_5_models']]
                break

    return experiments


def load_hardware_metrics():
    """Load all hardware_metrics/*/summary.json + gpu_metrics.json files."""
    all_hw = {}
    if not os.path.isdir(HW_METRICS_DIR):
        return all_hw
    for exp_dir in sorted(os.listdir(HW_METRICS_DIR)):
        exp_path = os.path.join(HW_METRICS_DIR, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        models = {}
        for model_dir in sorted(os.listdir(exp_path)):
            model_path = os.path.join(exp_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            # There's a timestamp subfolder inside
            for ts_dir in os.listdir(model_path):
                ts_path = os.path.join(model_path, ts_dir)
                if not os.path.isdir(ts_path):
                    continue
                entry = {'model_id': model_dir}
                for fname in ['summary.json', 'gpu_metrics.json', 'cpu_metrics.json',
                              'latency_metrics.json', 'metadata.json']:
                    fpath = os.path.join(ts_path, fname)
                    if os.path.isfile(fpath):
                        with open(fpath) as fh:
                            entry[fname.replace('.json', '')] = json.load(fh)
                models[model_dir] = entry
        all_hw[exp_dir] = models
    return all_hw


def load_hardware_perf_logs():
    """Load all hardware_performance_logs/*/*.json files."""
    all_perf = {}
    if not os.path.isdir(HW_PERF_DIR):
        return all_perf
    for exp_dir in sorted(os.listdir(HW_PERF_DIR)):
        exp_path = os.path.join(HW_PERF_DIR, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        models = []
        for jf in sorted(glob.glob(os.path.join(exp_path, '*.json'))):
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                    data['_perf_file'] = os.path.basename(jf)
                    models.append(data)
            except:
                pass
        all_perf[exp_dir] = models
    return all_perf


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_dataset(exp):
    """Safely extract dataset name from experiment."""
    ds = exp.get('dataset', None)
    if isinstance(ds, dict):
        ds = ds.get('name', ds.get('tag', None))
    if ds is None or not isinstance(ds, str):
        eid = exp.get('experiment_id', None) or exp.get('_filename', '')
        # Remove .json extension for filename-based IDs
        eid = eid.replace('.json', '')
        parts = eid.split('-')
        ds = parts[0] if parts else 'unknown'
    return ds


def get_exp_id(exp):
    """Safely extract experiment ID."""
    eid = exp.get('experiment_id', None) or exp.get('_filename', '')
    return eid.replace('.json', '') if eid else 'unknown'


def get_total_time(exp):
    """Extract total time in seconds from experiment."""
    perf = exp.get('performance_metrics', {})
    t = perf.get('total_time_seconds', None)
    if t is None:
        t = perf.get('elapsed_seconds', 0)
    return t or 0


def save_latex(filename, content):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  LaTeX → {path}")


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura → {path}")


# ─── TABLE 1: Resumen de Resultados ────────────────────────────────────────

def table1_resumen(experiments):
    print("=" * 80)
    print("  TABLA 1: Resumen General de Resultados por Experimento")
    print("=" * 80)

    rows = []
    for exp in experiments:
        eid = get_exp_id(exp)
        ds = get_dataset(exp)
        ds_name = DATASET_DISPLAY.get(ds, ds)
        lb = exp.get('leaderboard', [])
        ostats = exp.get('optimization_stats', {})
        best = max((m.get('performance', 0) for m in lb), default=0)
        gen = ostats.get('models_generated', len(lb))
        proc = ostats.get('models_processed', len(lb))
        expl = ostats.get('exploration_models', 0)
        deep = ostats.get('deep_training_models', 0)
        total_sec = get_total_time(exp)

        rows.append({
            'exp': eid, 'ds': ds_name, 'best': best,
            'gen': gen, 'proc': proc, 'expl': expl, 'deep': deep,
            'time': total_sec,
        })

    header = f"  {'Experimento':<36} {'Dataset':<16} {'Best Acc':>9} {'Gen':>5} {'Proc':>5} {'Expl':>5} {'Deep':>5} {'Tiempo':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<36} {r['ds']:<16} {r['best']:>9.4f} {r['gen']:>5} {r['proc']:>5} {r['expl']:>5} {r['deep']:>5} {fmt_time(r['time']):>10}")

    # LaTeX
    latex = "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
    latex += "Experimento & Dataset & Best Acc & Gen & Proc & Expl & Deep & Tiempo \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['best']:.4f} & {r['gen']} & {r['proc']} & {r['expl']} & {r['deep']} & {fmt_time(r['time'])} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla1_resumen_resultados.tex', latex)

    return rows


# ─── TABLE 2: Reproducibilidad ─────────────────────────────────────────────

def table2_reproducibilidad(experiments):
    print("=" * 80)
    print("  TABLA 2: Reproducibilidad — Mejores Accuracies por Dataset")
    print("=" * 80)

    by_ds = defaultdict(list)
    for exp in experiments:
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        best = max((m.get('performance', 0) for m in lb), default=0)
        by_ds[ds].append(best)

    header = f"  {'Dataset':<18} {'Runs':>5} {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))

    latex = "\\begin{tabular}{lrrrrr}\n\\toprule\nDataset & Runs & Media & Std & Min & Max \\\\\n\\midrule\n"

    for ds in sorted(by_ds.keys()):
        vals = by_ds[ds]
        ds_name = DATASET_DISPLAY.get(ds, ds)
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {ds_name:<18} {len(vals):>5} {mean:>10.4f} {std:>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")
        latex += f"{ds_name} & {len(vals)} & {mean:.4f} & {std:.4f} & {min(vals):.4f} & {max(vals):.4f} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla2_reproducibilidad.tex', latex)

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    data_plot = []
    labels = []
    for ds in sorted(by_ds.keys()):
        data_plot.append(by_ds[ds])
        labels.append(DATASET_DISPLAY.get(ds, ds))
    bp = ax.boxplot(data_plot, labels=labels, patch_artist=True)
    for patch, c in zip(bp['boxes'], ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Best Accuracy')
    ax.set_title('Reproducibilidad: Distribución de Mejores Accuracies')
    ax.grid(axis='y', alpha=0.3)
    save_figure(fig, 'fig2_reproducibilidad_boxplot.png')


# ─── TABLE 3: Exploración vs Deep Training ─────────────────────────────────

def table3_exploracion_vs_deep(experiments):
    print("=" * 80)
    print("  TABLA 3: Exploración vs Deep Training")
    print("=" * 80)

    rows = []
    for exp in experiments:
        eid = get_exp_id(exp)
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        expl_accs = [m['performance'] for m in lb if m.get('phase') == 'exploration']
        deep_accs = [m['performance'] for m in lb if m.get('phase') == 'deep_training']
        if expl_accs and deep_accs:
            best_expl = max(expl_accs)
            best_deep = max(deep_accs)
            improvement = best_deep - best_expl
            rows.append({
                'exp': eid, 'ds': DATASET_DISPLAY.get(ds, ds),
                'best_expl': best_expl, 'mean_expl': np.mean(expl_accs),
                'best_deep': best_deep, 'mean_deep': np.mean(deep_accs),
                'improvement': improvement,
            })

    header = f"  {'Experimento':<36} {'Dataset':<16} {'BestExpl':>9} {'BestDeep':>9} {'Mejora':>9}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        sign = '+' if r['improvement'] >= 0 else ''
        print(f"  {r['exp']:<36} {r['ds']:<16} {r['best_expl']:>9.4f} {r['best_deep']:>9.4f} {sign}{r['improvement']:>8.4f}")

    latex = "\\begin{tabular}{llrrrr}\n\\toprule\nExperimento & Dataset & Best Expl & Best Deep & Mejora \\\\\n\\midrule\n"
    for r in rows:
        sign = '+' if r['improvement'] >= 0 else ''
        latex += f"{r['exp']} & {r['ds']} & {r['best_expl']:.4f} & {r['best_deep']:.4f} & {sign}{r['improvement']:.4f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla3_mejora_fases.tex', latex)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w/2, [r['best_expl'] for r in rows], w, label='Exploración', color=COLORS['exploration'])
    ax.bar(x + w/2, [r['best_deep'] for r in rows], w, label='Deep Training', color=COLORS['deep_training'])
    ax.set_xticks(x)
    ax.set_xticklabels([r['ds'] + '\n' + (r['exp'].split('-')[1] if len(r['exp'].split('-')) > 1 else r['exp'][:10]) for r in rows], fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Exploración vs Deep Training')
    ax.legend()
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis='y', alpha=0.3)
    save_figure(fig, 'fig3_exploracion_vs_deep.png')


# ─── TABLE 4: Mejores Arquitecturas ────────────────────────────────────────

def table4_arquitecturas(experiments):
    print("=" * 80)
    print("  TABLA 4: Mejores Arquitecturas Descubiertas")
    print("=" * 80)

    rows = []
    for exp in experiments:
        eid = get_exp_id(exp)
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        if not lb:
            continue
        best = max(lb, key=lambda m: m.get('performance', 0))
        rows.append({
            'exp': eid, 'ds': DATASET_DISPLAY.get(ds, ds),
            'acc': best.get('performance', 0),
            'arch': best.get('base_architecture', '?'),
            'clf': best.get('classifier_type', '?'),
            'phase': best.get('phase', '?'),
        })

    header = f"  {'Experimento':<36} {'Dataset':<16} {'Acc':>8} {'Arquitectura':<15} {'Clasificador':<10} {'Fase':<15}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<36} {r['ds']:<16} {r['acc']:>8.4f} {r['arch']:<15} {r['clf']:<10} {r['phase']:<15}")

    latex = "\\begin{tabular}{llrlll}\n\\toprule\nExperimento & Dataset & Acc & Arquitectura & Clasificador & Fase \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['acc']:.4f} & {r['arch']} & {r['clf']} & {r['phase']} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla4_arquitecturas_optimas.tex', latex)

    # Distribution pie
    arch_count = defaultdict(int)
    for r in rows:
        arch_count[r['arch']] += 1
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(arch_count.keys())
    sizes = list(arch_count.values())
    colors = [COLORS.get(l.lower(), '#95a5a6') for l in labels]
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.set_title('Distribución de Arquitecturas Ganadoras')
    save_figure(fig, 'fig4_arquitecturas_distribucion.png')


# ─── TABLE 5: Infraestructura ──────────────────────────────────────────────

def table5_infraestructura(experiments):
    print("=" * 80)
    print("  TABLA 5: Infraestructura del Cluster en la Nube")
    print("=" * 80)

    workers = {}
    for exp in experiments:
        for w in exp.get('all_workers_hardware', []):
            hostname = w.get('hostname', 'unknown')
            if hostname not in workers:
                workers[hostname] = w

    print(f"  Total de nodos worker únicos detectados: {len(workers)}")
    print()

    header = f"  {'Worker':<12} {'CPU Cores':>10} {'RAM':>12} {'GPU':<30} {'VRAM':>10} {'Driver':>14}"
    print(header)
    print("  " + "-" * len(header.strip()))

    latex = "\\begin{tabular}{lrrlrl}\n\\toprule\nWorker & CPU Cores & RAM & GPU & VRAM & Driver \\\\\n\\midrule\n"

    total_cores = 0
    total_vram = 0
    for i, (hostname, w) in enumerate(sorted(workers.items()), 1):
        cores = w.get('cpu_threads', w.get('cpu_cores', '?'))
        ram = w.get('ram_total_gb', 0)
        gpus = w.get('gpu_models', ['?'])
        gpu_str = f"{len(gpus)}× {gpus[0]}" if gpus else '?'
        vram_list = w.get('gpu_memory_gb', [0])
        vram = int(sum(vram_list) * 1024) if vram_list else 0
        driver = w.get('nvidia_driver', w.get('cuda_version', '?'))
        total_cores += int(cores) if str(cores).isdigit() else 0
        total_vram += vram

        name = f"Worker {i}"
        print(f"  {name:<12} {cores:>10} {ram:>10.2f} GB  {gpu_str:<30} {vram:>7} MiB  {driver:>14}")
        latex += f"{name} & {cores} & {ram:.1f} GB & {gpu_str} & {vram} MiB & {driver} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla5_infraestructura.tex', latex)

    print(f"\n  RESUMEN CLUSTER:")
    print(f"    Nodos totales:     {len(workers)}")
    print(f"    GPUs totales:      {len(workers)}")
    print(f"    CPU cores totales: {total_cores}")
    print(f"    VRAM total:        {total_vram} MiB")


# ─── TABLE 6: Eficiencia del Pipeline ──────────────────────────────────────

def table6_eficiencia(experiments):
    print("=" * 80)
    print("  TABLA 6: Eficiencia del Pipeline de Optimización")
    print("=" * 80)

    rows = []
    for exp in experiments:
        eid = get_exp_id(exp)
        ds = get_dataset(exp)
        ostats = exp.get('optimization_stats', {})
        gen = ostats.get('models_generated', 0)
        proc = ostats.get('models_processed', 0)
        total_sec = get_total_time(exp)
        expl = ostats.get('exploration_models', 0)
        deep = ostats.get('deep_training_models', 0)

        t_per_model = total_sec / proc if proc > 0 else 0
        throughput = (proc / total_sec * 3600) if total_sec > 0 else 0

        rows.append({
            'exp': eid, 'ds': DATASET_DISPLAY.get(ds, ds),
            'gen': gen, 'proc': proc, 'expl': expl, 'deep': deep,
            'total_sec': total_sec, 't_per_model': t_per_model,
            'throughput': throughput,
        })

    header = f"  {'Experimento':<36} {'Dataset':<16} {'Gen':>5} {'Proc':>5} {'T/Modelo':>9} {'Mod/h':>8} {'Tiempo':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<36} {r['ds']:<16} {r['gen']:>5} {r['proc']:>5} {r['t_per_model']:>7.0f}s {r['throughput']:>8.1f} {fmt_time(r['total_sec']):>10}")

    total_models = sum(r['proc'] for r in rows)
    total_time = sum(r['total_sec'] for r in rows)
    avg_throughput = np.mean([r['throughput'] for r in rows]) if rows else 0
    avg_tpm = np.mean([r['t_per_model'] for r in rows]) if rows else 0

    print(f"\n  ESTADÍSTICAS AGREGADAS:")
    print(f"    Total modelos entrenados: {total_models}")
    print(f"    Tiempo total:             {total_time/3600:.1f} horas")
    print(f"    Throughput promedio:       {avg_throughput:.1f} modelos/hora")
    print(f"    Tiempo promedio/modelo:    {avg_tpm:.0f} segundos")

    latex = "\\begin{tabular}{llrrrrr}\n\\toprule\nExperimento & Dataset & Gen & Proc & T/Modelo & Mod/h & Tiempo \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['gen']} & {r['proc']} & {r['t_per_model']:.0f}s & {r['throughput']:.1f} & {fmt_time(r['total_sec'])} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla6_eficiencia_pipeline.tex', latex)

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    def short_label(r):
        parts = r['exp'].split('-')
        suffix = parts[1][:6] if len(parts) > 1 else r['exp'][:10]
        return r['ds'] + '\n' + suffix
    names = [short_label(r) for r in rows]
    x = np.arange(len(rows))
    ax1.bar(x, [r['throughput'] for r in rows], color=COLORS['exploration'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('Modelos/hora')
    ax1.set_title('Throughput del Pipeline')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, [r['t_per_model'] for r in rows], color=COLORS['deep_training'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Segundos/modelo')
    ax2.set_title('Tiempo por Modelo')
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'fig6_eficiencia_pipeline.png')

    return rows


# ─── TABLE 7: CNN vs Inception ──────────────────────────────────────────────

def table7_cnn_vs_inception(experiments):
    print("=" * 80)
    print("  TABLA 7: Comparación CNN vs Inception por Dataset")
    print("=" * 80)

    by_ds_arch = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        ds = get_dataset(exp)
        for m in exp.get('leaderboard', []):
            arch = m.get('base_architecture', '').upper()
            if arch in ('CNN', 'INCEPTION'):
                by_ds_arch[ds][arch].append(m.get('performance', 0))

    header = f"  {'Dataset':<18} {'Arquitectura':<14} {'Modelos':>9} {'Mejor':>10} {'Media':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))

    latex = "\\begin{tabular}{llrrrr}\n\\toprule\nDataset & Arquitectura & Modelos & Mejor Acc & Media Acc \\\\\n\\midrule\n"

    for ds in sorted(by_ds_arch.keys()):
        for arch in ['CNN', 'INCEPTION']:
            vals = by_ds_arch[ds].get(arch, [])
            if not vals:
                continue
            ds_name = DATASET_DISPLAY.get(ds, ds)
            print(f"  {ds_name:<18} {arch:<14} {len(vals):>9} {max(vals):>10.4f} {np.mean(vals):>10.4f}")
            latex += f"{ds_name} & {arch} & {len(vals)} & {max(vals):.4f} & {np.mean(vals):.4f} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla7_cnn_vs_inception.tex', latex)

    # Grouped bar chart
    datasets = sorted(by_ds_arch.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    w = 0.35
    cnn_best = [max(by_ds_arch[ds].get('CNN', [0])) for ds in datasets]
    inc_best = [max(by_ds_arch[ds].get('INCEPTION', [0])) for ds in datasets]
    ax.bar(x - w/2, cnn_best, w, label='CNN', color=COLORS['cnn'])
    ax.bar(x + w/2, inc_best, w, label='Inception', color=COLORS['inception'])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in datasets])
    ax.set_ylabel('Best Accuracy')
    ax.set_title('CNN vs Inception: Mejor Accuracy por Dataset')
    ax.legend()
    ax.set_ylim(0.9, 1.01)
    ax.grid(axis='y', alpha=0.3)
    save_figure(fig, 'fig7_cnn_vs_inception.png')


# ─── TABLE 8: Benchmarks ───────────────────────────────────────────────────

def table8_benchmarks(experiments):
    print("=" * 80)
    print("  TABLA 8: Comparación con Benchmarks Publicados")
    print("=" * 80)

    BENCHMARKS = {
        'mnist': [
            ('LeNet-5 (LeCun 1998)', 0.9920, 'Manual'),
            ('VGG-16 (adaptado)', 0.9950, 'Manual'),
            ('ResNet-18 (adaptado)', 0.9960, 'Manual'),
            ('Auto-Keras (Jin 2019)', 0.9940, 'AutoML'),
            ('Google AutoML Vision', 0.9950, 'AutoML (Cloud)'),
            ('DARTS (Liu 2019)', 0.9960, 'NAS'),
        ],
        'cifar10': [
            ('VGG-16 (Simonyan 2014)', 0.9340, 'Manual'),
            ('ResNet-110 (He 2015)', 0.9370, 'Manual'),
            ('DenseNet-BC (Huang 2017)', 0.9520, 'Manual'),
            ('Auto-Keras (Jin 2019)', 0.9530, 'AutoML'),
            ('NASNet-A (Zoph 2018)', 0.9720, 'NAS'),
            ('DARTS (Liu 2019)', 0.9706, 'NAS'),
        ],
        'fashion_mnist': [
            ('CNN simple (2 conv)', 0.9200, 'Manual'),
            ('VGG-16 (adaptado)', 0.9350, 'Manual'),
            ('ResNet-18 (adaptado)', 0.9460, 'Manual'),
            ('Auto-Keras (Jin 2019)', 0.9460, 'AutoML'),
            ('Google AutoML Tables', 0.9500, 'AutoML (Cloud)'),
        ],
    }

    our_best = {}
    for exp in experiments:
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        best = max((m.get('performance', 0) for m in lb), default=0)
        if ds not in our_best or best > our_best[ds]:
            our_best[ds] = best

    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        if ds_key not in our_best:
            continue
        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        print(f"\n  ─── {ds_name} {'─' * (60 - len(ds_name))}")

        benchmarks = BENCHMARKS.get(ds_key, [])
        our = our_best[ds_key]
        beats = 0
        for name, acc, tipo in benchmarks:
            diff = our - acc
            sign = '+' if diff >= 0 else ''
            print(f"  {name:<36} {acc:.4f}  {tipo:<16} {sign}{diff:.4f}")
            if diff > 0:
                beats += 1
        print(f"  {'─' * 70}")
        print(f"  {'NUESTRO AUTOML':<36} {our:.4f}  {'AutoML (Cloud)':<16} ---")
        print(f"  Supera a: {beats}/{len(benchmarks)} benchmarks")

        latex = f"\\begin{{tabular}}{{lrll}}\n\\toprule\nMétodo & Accuracy & Tipo & vs Nuestro \\\\\n\\midrule\n"
        for name, acc, tipo in benchmarks:
            diff = our - acc
            sign = '+' if diff >= 0 else ''
            latex += f"{name} & {acc:.4f} & {tipo} & {sign}{diff:.4f} \\\\\n"
        latex += f"\\midrule\n\\textbf{{Nuestro AutoML}} & \\textbf{{{our:.4f}}} & AutoML (Cloud) & --- \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}"
        save_latex(f'tabla8_benchmark_{ds_key}.tex', latex)

    # Combined bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, ds_key in enumerate(['mnist', 'cifar10', 'fashion_mnist']):
        if ds_key not in our_best:
            continue
        ax = axes[idx]
        benchmarks = BENCHMARKS.get(ds_key, [])
        names = [b[0].split('(')[0].strip() for b in benchmarks] + ['Nuestro\nAutoML']
        accs = [b[1] for b in benchmarks] + [our_best[ds_key]]
        colors = ['#bdc3c7'] * len(benchmarks) + ['#e74c3c']
        ax.barh(range(len(names)), accs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Accuracy')
        ax.set_title(DATASET_DISPLAY.get(ds_key, ds_key))
        ax.set_xlim(min(accs) - 0.02, 1.01)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Comparación con Benchmarks Publicados', fontsize=14)
    fig.tight_layout()
    save_figure(fig, 'fig8_comparacion_benchmarks.png')


# ─── TABLE 9: Configuración AutoML ─────────────────────────────────────────

def table9_config(experiments):
    print("=" * 80)
    print("  TABLA 9: Configuración y Espacio de Búsqueda del AutoML")
    print("=" * 80)

    configs = defaultdict(list)
    for exp in experiments:
        tc = exp.get('training_config', {})
        key = json.dumps(tc, sort_keys=True)
        configs[key].append(exp.get('experiment_id', ''))

    for i, (key, exps) in enumerate(configs.items(), 1):
        tc = json.loads(key)
        print(f"\n  ─── Configuración {i} (usada en {len(exps)} experimentos) ───")
        for k, v in tc.items():
            print(f"    {k:<30} {v}")

    latex = "\\begin{tabular}{lr}\n\\toprule\nParámetro & Valor \\\\\n\\midrule\n"
    # Use first config as representative
    if configs:
        first_key = list(configs.keys())[0]
        tc = json.loads(first_key)
        for k, v in tc.items():
            latex += f"{k} & {v} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla9_configuracion_automl.tex', latex)


# ═══════════════════════════════════════════════════════════════════════════
#  NUEVAS TABLAS: Hardware Metrics & Performance Logs
# ═══════════════════════════════════════════════════════════════════════════

# ─── TABLE 10: GPU Utilization & Memory per Model ──────────────────────────

def table10_gpu_utilization(hw_metrics):
    print("=" * 80)
    print("  TABLA 10: Utilización GPU por Modelo (hardware_metrics)")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        for mid, mdata in sorted(models.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            s = mdata.get('summary', {})
            avg_gpu = s.get('avg_gpu_utilization', {})
            avg_mem = s.get('avg_gpu_memory', {})
            idle = s.get('idle_time', {})

            gpu_util = list(avg_gpu.values())[0] if avg_gpu else 0
            gpu_mem_pct = list(avg_mem.values())[0] if avg_mem else 0
            idle_total = idle.get('total_seconds', 0)
            idle_avg = idle.get('average_seconds', 0)
            n_records = s.get('num_records', 0)

            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'model_id': mid,
                'gpu_util': gpu_util, 'gpu_mem_pct': gpu_mem_pct,
                'idle_total': idle_total, 'idle_avg': idle_avg,
                'n_records': n_records,
            })

    if not rows:
        print("  No data found.")
        return

    header = f"  {'Experimento':<32} {'Model':>6} {'GPU Util%':>10} {'GPU Mem%':>10} {'Idle Total':>12} {'Idle Avg':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['model_id']:>6} {r['gpu_util']:>9.1f}% {r['gpu_mem_pct']:>9.1f}% {r['idle_total']:>10.0f}s {r['idle_avg']:>9.0f}s")

    # Aggregate by experiment
    print(f"\n  RESUMEN POR EXPERIMENTO:")
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r['exp']].append(r)

    agg_rows = []
    header2 = f"  {'Experimento':<32} {'Dataset':<16} {'Models':>7} {'Avg GPU%':>10} {'Avg Mem%':>10} {'Avg Idle':>10}"
    print(header2)
    print("  " + "-" * len(header2.strip()))
    for exp_id in sorted(by_exp.keys()):
        rs = by_exp[exp_id]
        ds = rs[0]['ds']
        avg_gpu = np.mean([r['gpu_util'] for r in rs])
        avg_mem = np.mean([r['gpu_mem_pct'] for r in rs])
        avg_idle = np.mean([r['idle_avg'] for r in rs])
        print(f"  {exp_id:<32} {ds:<16} {len(rs):>7} {avg_gpu:>9.1f}% {avg_mem:>9.1f}% {avg_idle:>9.0f}s")
        agg_rows.append({'exp': exp_id, 'ds': ds, 'n': len(rs), 'gpu': avg_gpu, 'mem': avg_mem, 'idle': avg_idle})

    latex = "\\begin{tabular}{llrrrr}\n\\toprule\nExperimento & Dataset & Modelos & GPU Util (\\%) & GPU Mem (\\%) & Idle Prom (s) \\\\\n\\midrule\n"
    for r in agg_rows:
        latex += f"{r['exp']} & {r['ds']} & {r['n']} & {r['gpu']:.1f} & {r['mem']:.1f} & {r['idle']:.0f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla10_gpu_utilization.tex', latex)

    # Figure: GPU util + mem as grouped bars per experiment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [r['exp'].replace('-2026', '\n2026').replace('-2025', '\n2025')[:25] for r in agg_rows]
    x = np.arange(len(agg_rows))

    ax1.bar(x, [r['gpu'] for r in agg_rows], color=COLORS['gpu_util'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['ds'] for r in agg_rows], fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('GPU Utilización (%)')
    ax1.set_title('Utilización Promedio GPU por Experimento')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, [r['mem'] for r in agg_rows], color=COLORS['gpu_mem'], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['ds'] for r in agg_rows], fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Memoria GPU (%)')
    ax2.set_title('Uso Promedio de Memoria GPU')
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'fig10_gpu_utilization.png')

    return rows


# ─── TABLE 11: GPU Temperature & Power ─────────────────────────────────────

def table11_gpu_power_temp(hw_metrics):
    print("=" * 80)
    print("  TABLA 11: Temperatura y Consumo Energético GPU")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        exp_temps = []
        exp_powers = []
        exp_power_pcts = []

        for mid, mdata in sorted(models.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            gpu_data = mdata.get('gpu_metrics', [])
            if not isinstance(gpu_data, list):
                continue
            for record in gpu_data:
                metrics = record.get('metrics', {})
                temp = metrics.get('temperature', {})
                power = metrics.get('power', {})
                for gpu_id in temp:
                    if isinstance(temp[gpu_id], (int, float)):
                        exp_temps.append(temp[gpu_id])
                for gpu_id in power:
                    if isinstance(power[gpu_id], dict):
                        exp_powers.append(power[gpu_id].get('draw_watts', 0))
                        exp_power_pcts.append(power[gpu_id].get('percentage', 0))

        if exp_temps:
            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'temp_avg': np.mean(exp_temps), 'temp_max': max(exp_temps), 'temp_min': min(exp_temps),
                'power_avg': np.mean(exp_powers), 'power_max': max(exp_powers),
                'power_pct_avg': np.mean(exp_power_pcts),
                'n_samples': len(exp_temps),
            })

    if not rows:
        print("  No temperature/power data found.")
        return

    header = f"  {'Experimento':<32} {'Dataset':<16} {'Temp Avg':>9} {'Temp Max':>9} {'W Avg':>8} {'W Max':>8} {'%TDP':>7} {'Samples':>8}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['ds']:<16} {r['temp_avg']:>7.1f}°C {r['temp_max']:>7.0f}°C {r['power_avg']:>7.1f}W {r['power_max']:>7.1f}W {r['power_pct_avg']:>6.1f}% {r['n_samples']:>8}")

    global_temp_avg = np.mean([r['temp_avg'] for r in rows])
    global_power_avg = np.mean([r['power_avg'] for r in rows])
    global_power_pct = np.mean([r['power_pct_avg'] for r in rows])
    print(f"\n  PROMEDIOS GLOBALES:")
    print(f"    Temperatura media:         {global_temp_avg:.1f}°C")
    print(f"    Consumo medio:             {global_power_avg:.1f}W")
    print(f"    Uso TDP medio:             {global_power_pct:.1f}%")

    latex = "\\begin{tabular}{llrrrrr}\n\\toprule\nExperimento & Dataset & Temp Avg (°C) & Temp Max (°C) & Power Avg (W) & Power Max (W) & \\%TDP \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['temp_avg']:.1f} & {r['temp_max']:.0f} & {r['power_avg']:.1f} & {r['power_max']:.1f} & {r['power_pct_avg']:.1f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla11_gpu_power_temp.tex', latex)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(rows))
    ds_labels = [r['ds'] for r in rows]

    ax1.bar(x, [r['temp_avg'] for r in rows], color=COLORS['temp'], alpha=0.7, label='Avg')
    ax1.scatter(x, [r['temp_max'] for r in rows], color='black', zorder=5, marker='^', label='Max')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Temperatura GPU por Experimento')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, [r['power_avg'] for r in rows], color=COLORS['power'], alpha=0.7, label='Avg Draw')
    ax2.scatter(x, [r['power_max'] for r in rows], color='black', zorder=5, marker='^', label='Max Draw')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_labels, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Watts')
    ax2.set_title('Consumo Energético GPU')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, 'fig11_gpu_power_temp.png')

    return rows


# ─── TABLE 12: Inference Latency ───────────────────────────────────────────

def table12_latency(hw_metrics):
    print("=" * 80)
    print("  TABLA 12: Latencia de Inferencia por Modelo")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        for mid, mdata in sorted(models.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            s = mdata.get('summary', {})
            lat = s.get('avg_inference_latency_ms', 0)

            # Get architecture from summary
            arch_info = s.get('model_architecture', {})
            total_params = arch_info.get('total_params', 0)
            layers = arch_info.get('layers', [])
            layer_types = [l.get('type', '') for l in layers]
            has_inception = any('Inception' in t for t in layer_types)
            arch_type = 'INCEPTION' if has_inception else 'CNN'

            # Get latency details from latency_metrics
            lat_data = mdata.get('latency_metrics', [])
            latencies = []
            throughputs = []
            if isinstance(lat_data, list):
                for record in lat_data:
                    inf = record.get('metrics', {}).get('inference', {})
                    if 'avg_latency_ms' in inf:
                        latencies.append(inf['avg_latency_ms'])
                    if 'throughput_samples_per_sec' in inf:
                        throughputs.append(inf['throughput_samples_per_sec'])

            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'model_id': mid, 'arch': arch_type,
                'latency_ms': lat if lat else (np.mean(latencies) if latencies else 0),
                'throughput': np.mean(throughputs) if throughputs else 0,
                'total_params': total_params,
            })

    if not rows:
        print("  No latency data found.")
        return

    header = f"  {'Experimento':<32} {'Model':>6} {'Arch':<10} {'Latencia':>10} {'Throughput':>12} {'Params':>12}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['model_id']:>6} {r['arch']:<10} {r['latency_ms']:>8.2f}ms {r['throughput']:>8.1f} s/s {r['total_params']:>12,}")

    # By architecture
    by_arch = defaultdict(list)
    for r in rows:
        if r['latency_ms'] > 0:
            by_arch[r['arch']].append(r)

    print(f"\n  RESUMEN POR ARQUITECTURA:")
    for arch in sorted(by_arch.keys()):
        rs = by_arch[arch]
        avg_lat = np.mean([r['latency_ms'] for r in rs])
        avg_thr = np.mean([r['throughput'] for r in rs if r['throughput'] > 0])
        avg_par = np.mean([r['total_params'] for r in rs])
        print(f"    {arch:<12} {len(rs)} modelos  Lat: {avg_lat:.2f}ms  Throughput: {avg_thr:.1f} s/s  Params: {avg_par:,.0f}")

    latex = "\\begin{tabular}{llrlrrr}\n\\toprule\nExperimento & Model & Arch & Latencia (ms) & Throughput (s/s) & Params \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['model_id']} & {r['arch']} & {r['latency_ms']:.2f} & {r['throughput']:.1f} & {r['total_params']:,} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla12_latencia_inferencia.tex', latex)

    # Figure: Latency vs Params scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for arch in sorted(by_arch.keys()):
        rs = by_arch[arch]
        params = [r['total_params'] for r in rs]
        lats = [r['latency_ms'] for r in rs]
        color = COLORS['cnn'] if arch == 'CNN' else COLORS['inception']
        ax.scatter(params, lats, label=arch, color=color, alpha=0.7, s=60)
    ax.set_xlabel('Total Parámetros')
    ax.set_ylabel('Latencia de Inferencia (ms)')
    ax.set_title('Latencia vs Complejidad del Modelo')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    save_figure(fig, 'fig12_latencia_vs_params.png')

    return rows


# ─── TABLE 13: Training Time per Epoch (hardware_performance_logs) ─────────

def table13_training_epochs(hw_perf):
    print("=" * 80)
    print("  TABLA 13: Métricas de Entrenamiento por Modelo (Perf Logs)")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_perf.items()):
        ds = exp_id.split('-')[0]
        for mdata in models:
            mm = mdata.get('model_metrics', {})
            tm = mdata.get('training_metrics', {})
            if not tm:
                continue

            model_id = mm.get('model_id', '?')
            total_params = mm.get('total_parameters', 0)
            trainable = mm.get('trainable_parameters', 0)
            layers = mm.get('layer_count', 0)
            build_ms = tm.get('build_time_ms', 0)
            train_ms = tm.get('train_time_ms', 0)
            epochs = tm.get('epochs_completed', 0)
            epoch_times = tm.get('train_time_per_epoch_ms', [])
            batch_size = tm.get('batch_size', 0)
            final_acc = tm.get('final_accuracy', 0)
            final_loss = tm.get('final_loss', 0)

            avg_epoch_ms = np.mean(epoch_times) if epoch_times else 0

            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'model_id': model_id,
                'total_params': total_params, 'trainable': trainable,
                'layers': layers,
                'build_ms': build_ms, 'train_s': train_ms / 1000,
                'epochs': epochs, 'avg_epoch_s': avg_epoch_ms / 1000,
                'final_acc': final_acc, 'final_loss': final_loss,
                'batch_size': batch_size,
            })

    if not rows:
        print("  No training performance data found.")
        return

    header = f"  {'Experimento':<32} {'Mod':>4} {'Params':>10} {'Layers':>7} {'Build':>7} {'Train':>8} {'Epochs':>7} {'Ep Avg':>8} {'Acc':>8}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['model_id']:>4} {r['total_params']:>10,} {r['layers']:>7} {r['build_ms']/1000:>6.1f}s {r['train_s']:>7.0f}s {r['epochs']:>7} {r['avg_epoch_s']:>7.1f}s {r['final_acc']:>8.4f}")

    # Aggregate by dataset
    print(f"\n  RESUMEN POR DATASET:")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r['ds']].append(r)

    agg_rows = []
    for ds_name in sorted(by_ds.keys()):
        rs = by_ds[ds_name]
        avg_train = np.mean([r['train_s'] for r in rs])
        avg_epoch = np.mean([r['avg_epoch_s'] for r in rs])
        avg_params = np.mean([r['total_params'] for r in rs])
        avg_acc = np.mean([r['final_acc'] for r in rs])
        best_acc = max(r['final_acc'] for r in rs)
        print(f"    {ds_name:<18} {len(rs):>3} modelos  Avg Train: {avg_train:.0f}s  Avg Epoch: {avg_epoch:.1f}s  Avg Params: {avg_params:,.0f}  Best Acc: {best_acc:.4f}")
        agg_rows.append({'ds': ds_name, 'n': len(rs), 'avg_train': avg_train, 'avg_epoch': avg_epoch, 'avg_params': avg_params, 'best_acc': best_acc, 'avg_acc': avg_acc})

    latex = "\\begin{tabular}{llrrrrrrr}\n\\toprule\nExperimento & Modelo & Params & Capas & Build (s) & Train (s) & Epochs & Avg Epoch (s) & Accuracy \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['model_id']} & {r['total_params']:,} & {r['layers']} & {r['build_ms']/1000:.1f} & {r['train_s']:.0f} & {r['epochs']} & {r['avg_epoch_s']:.1f} & {r['final_acc']:.4f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla13_training_epochs.tex', latex)

    # Compact summary table
    latex2 = "\\begin{tabular}{lrrrrrr}\n\\toprule\nDataset & Modelos & Avg Train (s) & Avg Epoch (s) & Avg Params & Best Acc & Avg Acc \\\\\n\\midrule\n"
    for r in agg_rows:
        latex2 += f"{r['ds']} & {r['n']} & {r['avg_train']:.0f} & {r['avg_epoch']:.1f} & {r['avg_params']:,.0f} & {r['best_acc']:.4f} & {r['avg_acc']:.4f} \\\\\n"
    latex2 += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla13b_training_summary_by_dataset.tex', latex2)

    # Figure: Training time vs accuracy scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ds_name in sorted(by_ds.keys()):
        rs = by_ds[ds_name]
        ax1.scatter([r['train_s'] for r in rs], [r['final_acc'] for r in rs], label=ds_name, alpha=0.7, s=50)
    ax1.set_xlabel('Tiempo de Entrenamiento (s)')
    ax1.set_ylabel('Accuracy Final')
    ax1.set_title('Tiempo de Entrenamiento vs Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    for ds_name in sorted(by_ds.keys()):
        rs = by_ds[ds_name]
        ax2.scatter([r['total_params'] for r in rs], [r['final_acc'] for r in rs], label=ds_name, alpha=0.7, s=50)
    ax2.set_xlabel('Total Parámetros')
    ax2.set_ylabel('Accuracy Final')
    ax2.set_title('Complejidad del Modelo vs Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    fig.tight_layout()
    save_figure(fig, 'fig13_training_vs_accuracy.png')

    return rows


# ─── TABLE 14: Idle Time Analysis ──────────────────────────────────────────

def table14_idle_time(hw_metrics):
    print("=" * 80)
    print("  TABLA 14: Análisis de Tiempo Ocioso de GPU")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        total_idle = 0
        count = 0
        max_idle = 0
        for mid, mdata in models.items():
            s = mdata.get('summary', {})
            idle = s.get('idle_time', {})
            total_idle += idle.get('total_seconds', 0)
            max_idle = max(max_idle, idle.get('max_seconds', 0))
            count += idle.get('idle_records_count', 0)

        if count > 0:
            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'total_idle': total_idle,
                'max_idle': max_idle,
                'models': len(models),
                'avg_idle_per_model': total_idle / len(models) if models else 0,
            })

    if not rows:
        print("  No idle time data found.")
        return

    header = f"  {'Experimento':<32} {'Dataset':<16} {'Models':>7} {'Total Idle':>12} {'Max Idle':>10} {'Avg/Model':>12}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['ds']:<16} {r['models']:>7} {fmt_time(r['total_idle']):>12} {r['max_idle']:>9.0f}s {r['avg_idle_per_model']:>10.0f}s")

    total_idle_all = sum(r['total_idle'] for r in rows)
    avg_idle = np.mean([r['avg_idle_per_model'] for r in rows])
    print(f"\n  RESUMEN:")
    print(f"    Tiempo ocioso total acumulado: {fmt_time(total_idle_all)} ({total_idle_all/3600:.1f} horas)")
    print(f"    Promedio idle por modelo:      {avg_idle:.0f}s")
    print(f"    El tiempo ocioso corresponde a la transferencia de parámetros vía RabbitMQ,")
    print(f"    la compilación del modelo, y la espera de nuevos trabajos entre entrenamientos.")

    latex = "\\begin{tabular}{llrrrr}\n\\toprule\nExperimento & Dataset & Modelos & Total Idle (s) & Max Idle (s) & Avg/Modelo (s) \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['models']} & {r['total_idle']:.0f} & {r['max_idle']:.0f} & {r['avg_idle_per_model']:.0f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla14_idle_time.tex', latex)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    ax.bar(x, [r['avg_idle_per_model'] for r in rows], color=COLORS['idle'], alpha=0.8)
    ax.scatter(x, [r['max_idle'] for r in rows], color='red', zorder=5, marker='^', label='Max Idle', s=50)
    ax.set_xticks(x)
    ax.set_xticklabels([r['ds'] for r in rows], fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Segundos')
    ax.set_title('Tiempo Ocioso de GPU por Experimento')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save_figure(fig, 'fig14_idle_time.png')

    return rows


# ─── TABLE 15: PCIe & Clock Speeds ─────────────────────────────────────────

def table15_pcie_clocks(hw_metrics):
    print("=" * 80)
    print("  TABLA 15: Velocidades PCIe y Reloj GPU")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        graphics_clocks = []
        mem_clocks = []
        pcie_gens = []
        pcie_widths = []

        for mid, mdata in models.items():
            gpu_data = mdata.get('gpu_metrics', [])
            if not isinstance(gpu_data, list):
                continue
            for record in gpu_data:
                metrics = record.get('metrics', {})
                clocks = metrics.get('clock_speeds', {})
                pcie = metrics.get('pcie_throughput', {})
                for gpu_id in clocks:
                    if not isinstance(clocks[gpu_id], dict):
                        continue
                    graphics_clocks.append(clocks[gpu_id].get('graphics_mhz', 0))
                    mem_clocks.append(clocks[gpu_id].get('memory_mhz', 0))
                for gpu_id in pcie:
                    if not isinstance(pcie[gpu_id], dict):
                        continue
                    gen = pcie[gpu_id].get('link_gen', '')
                    width = pcie[gpu_id].get('link_width', '')
                    if gen:
                        pcie_gens.append(str(gen))
                    if width:
                        pcie_widths.append(str(width))

        if graphics_clocks:
            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'gfx_avg': np.mean(graphics_clocks), 'gfx_max': max(graphics_clocks),
                'mem_avg': np.mean(mem_clocks), 'mem_max': max(mem_clocks),
                'pcie_gen': pcie_gens[0] if pcie_gens else '?',
                'pcie_width': pcie_widths[0] if pcie_widths else '?',
            })

    if not rows:
        print("  No clock/PCIe data found.")
        return

    header = f"  {'Experimento':<32} {'GFX Avg':>9} {'GFX Max':>9} {'MEM Avg':>9} {'MEM Max':>9} {'PCIe Gen':>9} {'Width':>6}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows:
        print(f"  {r['exp']:<32} {r['gfx_avg']:>7.0f}MHz {r['gfx_max']:>7.0f}MHz {r['mem_avg']:>7.0f}MHz {r['mem_max']:>7.0f}MHz {r['pcie_gen']:>9} {r['pcie_width']:>5}x")

    latex = "\\begin{tabular}{llrrrrl}\n\\toprule\nExperimento & Dataset & GFX Avg (MHz) & GFX Max (MHz) & MEM Avg (MHz) & MEM Max (MHz) & PCIe \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['ds']} & {r['gfx_avg']:.0f} & {r['gfx_max']:.0f} & {r['mem_avg']:.0f} & {r['mem_max']:.0f} & Gen{r['pcie_gen']} x{r['pcie_width']} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla15_pcie_clocks.tex', latex)


# ─── TABLE 16: Model Complexity vs Accuracy (from perf logs) ───────────────

def table16_model_complexity(hw_perf):
    print("=" * 80)
    print("  TABLA 16: Complejidad del Modelo vs Rendimiento (Perf Logs)")
    print("=" * 80)

    rows = []
    for exp_id, models in sorted(hw_perf.items()):
        ds = exp_id.split('-')[0]
        for mdata in models:
            mm = mdata.get('model_metrics', {})
            tm = mdata.get('training_metrics', {})
            hi = mdata.get('hardware_info', {})
            if not tm:
                continue

            # Detect arch from layers
            layers = mm.get('layer_details', [])
            layer_classes = [l.get('class_name', '') for l in layers]
            has_inception = any('Inception' in c for c in layer_classes)
            arch = 'INCEPTION' if has_inception else 'CNN'

            rows.append({
                'exp': exp_id, 'ds': DATASET_DISPLAY.get(ds, ds),
                'model_id': mm.get('model_id', '?'),
                'arch': arch,
                'total_params': mm.get('total_parameters', 0),
                'trainable': mm.get('trainable_parameters', 0),
                'layers': mm.get('layer_count', 0),
                'accuracy': tm.get('final_accuracy', 0),
                'loss': tm.get('final_loss', 0),
                'train_s': tm.get('train_time_ms', 0) / 1000,
                'epochs': tm.get('epochs_completed', 0),
                'build_s': tm.get('build_time_ms', 0) / 1000,
            })

    if not rows:
        print("  No model complexity data found.")
        return

    # Sort by accuracy descending
    rows.sort(key=lambda r: r['accuracy'], reverse=True)

    header = f"  {'Exp':<28} {'Mod':>4} {'Arch':<10} {'Params':>10} {'Layers':>7} {'Epochs':>7} {'Train':>8} {'Acc':>8} {'Loss':>8}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for r in rows[:30]:  # Show top 30
        print(f"  {r['exp']:<28} {r['model_id']:>4} {r['arch']:<10} {r['total_params']:>10,} {r['layers']:>7} {r['epochs']:>7} {r['train_s']:>7.0f}s {r['accuracy']:>8.4f} {r['loss']:>8.4f}")

    if len(rows) > 30:
        print(f"  ... (mostrando top 30 de {len(rows)} modelos)")

    # Summary: efficiency ratio (accuracy per million parameters)
    print(f"\n  TOP 5 MODELOS MÁS EFICIENTES (Acc/Million Params):")
    for r in rows:
        r['efficiency'] = r['accuracy'] / (r['total_params'] / 1e6) if r['total_params'] > 0 else 0
    eff_sorted = sorted(rows, key=lambda r: r['efficiency'], reverse=True)
    for r in eff_sorted[:5]:
        print(f"    {r['exp']} Model {r['model_id']}: {r['arch']} {r['total_params']:,} params → {r['accuracy']:.4f} acc  (eff: {r['efficiency']:.2f})")

    latex = "\\begin{tabular}{llrlrrrr}\n\\toprule\nExperimento & Modelo & Arch & Params & Capas & Train (s) & Accuracy & Loss \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['exp']} & {r['model_id']} & {r['arch']} & {r['total_params']:,} & {r['layers']} & {r['train_s']:.0f} & {r['accuracy']:.4f} & {r['loss']:.4f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    save_latex('tabla16_model_complexity.tex', latex)

    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  LLM CONTEXT OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_context(experiments, hw_metrics, hw_perf):
    """Generate a single text file with ALL data for LLM context injection."""
    print("=" * 80)
    print("  GENERANDO CONTEXTO LLM (archivo de texto plano)")
    print("=" * 80)

    lines = []
    lines.append("=" * 80)
    lines.append("CONTEXTO COMPLETO PARA LLM — TESIS: AutoML en la Nube")
    lines.append("Datos extraídos de experimentos con 2 GPUs (NVIDIA RTX 3080)")
    lines.append("=" * 80)
    lines.append("")

    # ─── Section 1: Experiment Results Summary ───
    lines.append("─── SECCIÓN 1: RESUMEN DE RESULTADOS ───")
    lines.append("")
    for exp in experiments:
        eid = get_exp_id(exp)
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        ostats = exp.get('optimization_stats', {})
        best = max((m.get('performance', 0) for m in lb), default=0)
        best_model = max(lb, key=lambda m: m.get('performance', 0)) if lb else {}

        lines.append(f"Experimento: {eid}")
        lines.append(f"  Dataset: {DATASET_DISPLAY.get(ds, ds)}")
        lines.append(f"  Mejor Accuracy: {best:.4f}")
        lines.append(f"  Arquitectura ganadora: {best_model.get('base_architecture', '?')} + {best_model.get('classifier_type', '?')}")
        lines.append(f"  Fase ganadora: {best_model.get('phase', '?')}")
        lines.append(f"  Modelos generados: {ostats.get('models_generated', 0)} (Exploración: {ostats.get('exploration_models', 0)}, Deep: {ostats.get('deep_training_models', 0)})")
        lines.append(f"  Tiempo total: {fmt_time(get_total_time(exp))}")

        lines.append(f"  Leaderboard completo:")
        for m in lb:
            lines.append(f"    Model {m.get('id','?')}: {m.get('performance',0):.4f} ({m.get('base_architecture','?')}+{m.get('classifier_type','?')}) [{m.get('phase','?')}]")
        lines.append("")

    # ─── Section 2: Reproducibility ───
    lines.append("─── SECCIÓN 2: REPRODUCIBILIDAD ───")
    by_ds = defaultdict(list)
    for exp in experiments:
        ds = get_dataset(exp)
        lb = exp.get('leaderboard', [])
        best = max((m.get('performance', 0) for m in lb), default=0)
        by_ds[ds].append(best)
    for ds in sorted(by_ds.keys()):
        vals = by_ds[ds]
        lines.append(f"  {DATASET_DISPLAY.get(ds,ds)}: {len(vals)} runs, Media={np.mean(vals):.4f}, Std={np.std(vals):.4f}, Min={min(vals):.4f}, Max={max(vals):.4f}")
    lines.append("")

    # ─── Section 3: Infrastructure ───
    lines.append("─── SECCIÓN 3: INFRAESTRUCTURA DEL CLUSTER ───")
    workers = {}
    for exp in experiments:
        for w in exp.get('all_workers_hardware', []):
            hostname = w.get('hostname', 'unknown')
            if hostname not in workers:
                workers[hostname] = w
    for i, (hostname, w) in enumerate(sorted(workers.items()), 1):
        lines.append(f"  Worker {i} ({hostname}):")
        lines.append(f"    CPU: {w.get('cpu_threads', '?')} threads, RAM: {w.get('ram_total_gb', 0):.1f} GB")
        gpus = w.get('gpu_models', [])
        lines.append(f"    GPU: {len(gpus)}× {gpus[0] if gpus else '?'}")
        lines.append(f"    VRAM: {sum(w.get('gpu_memory_gb', [0]))*1024:.0f} MiB")
        lines.append(f"    TF: {w.get('tensorflow_version', '?')}, CUDA: {w.get('cuda_version', '?')}")
        lines.append(f"    Python: {w.get('python_version', '?')}, OS: {w.get('os_type', '?')}")
    lines.append("")

    # ─── Section 4: GPU Hardware Metrics ───
    lines.append("─── SECCIÓN 4: MÉTRICAS DE HARDWARE GPU (hardware_metrics) ───")
    lines.append("  Incluye: Utilización GPU, Memoria, Temperatura, Power draw, Idle time, Latencia")
    lines.append("")

    for exp_id, models in sorted(hw_metrics.items()):
        ds = exp_id.split('-')[0]
        lines.append(f"  Experimento: {exp_id} ({DATASET_DISPLAY.get(ds, ds)})")
        for mid, mdata in sorted(models.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            s = mdata.get('summary', {})
            avg_gpu = s.get('avg_gpu_utilization', {})
            avg_mem = s.get('avg_gpu_memory', {})
            idle = s.get('idle_time', {})
            lat = s.get('avg_inference_latency_ms', 0)
            gpu_util = list(avg_gpu.values())[0] if avg_gpu else 0
            gpu_mem_pct = list(avg_mem.values())[0] if avg_mem else 0

            lines.append(f"    Model {mid}: GPU Util={gpu_util:.1f}%, Mem={gpu_mem_pct:.1f}%, Idle Total={idle.get('total_seconds',0):.0f}s, Idle Avg={idle.get('average_seconds',0):.0f}s, Latencia={lat:.2f}ms")

            # Temperature & Power from gpu_metrics
            gpu_data = mdata.get('gpu_metrics', [])
            if isinstance(gpu_data, list) and gpu_data:
                temps = []
                powers = []
                for record in gpu_data:
                    metrics = record.get('metrics', {})
                    for gid in metrics.get('temperature', {}):
                        t = metrics['temperature'][gid]
                        if isinstance(t, (int, float)):
                            temps.append(t)
                    for gid in metrics.get('power', {}):
                        p = metrics['power'][gid]
                        if isinstance(p, dict):
                            powers.append(p.get('draw_watts', 0))
                if temps:
                    lines.append(f"      Temp: avg={np.mean(temps):.1f}°C, max={max(temps)}°C")
                if powers:
                    lines.append(f"      Power: avg={np.mean(powers):.1f}W, max={max(powers):.1f}W")

            # Architecture from summary
            arch_info = s.get('model_architecture', {})
            if arch_info:
                lines.append(f"      Params: {arch_info.get('total_params', 0):,}, Input: {arch_info.get('input_shape','?')}, Output: {arch_info.get('output_shape','?')}")
        lines.append("")

    # ─── Section 5: Training Performance (hardware_performance_logs) ───
    lines.append("─── SECCIÓN 5: RENDIMIENTO DE ENTRENAMIENTO (hardware_performance_logs) ───")
    lines.append("  Incluye: Build time, Train time, Epochs, Time per epoch, Accuracy final, Loss")
    lines.append("")

    for exp_id, models in sorted(hw_perf.items()):
        ds = exp_id.split('-')[0]
        lines.append(f"  Experimento: {exp_id} ({DATASET_DISPLAY.get(ds, ds)})")
        for mdata in models:
            mm = mdata.get('model_metrics', {})
            tm = mdata.get('training_metrics', {})
            hi = mdata.get('hardware_info', {})
            if not tm:
                continue
            model_id = mm.get('model_id', '?')
            layers_det = mm.get('layer_details', [])
            layer_classes = [l.get('class_name', '') for l in layers_det]
            has_inception = any('Inception' in c for c in layer_classes)
            arch = 'INCEPTION' if has_inception else 'CNN'

            epoch_times = tm.get('train_time_per_epoch_ms', [])
            lines.append(f"    Model {model_id} ({arch}):")
            lines.append(f"      Params: {mm.get('total_parameters', 0):,} (trainable: {mm.get('trainable_parameters', 0):,}), Layers: {mm.get('layer_count', 0)}")
            lines.append(f"      Build: {tm.get('build_time_ms',0)/1000:.1f}s, Train: {tm.get('train_time_ms',0)/1000:.0f}s, Epochs: {tm.get('epochs_completed',0)}")
            if epoch_times:
                lines.append(f"      Time/Epoch: [{', '.join(f'{t/1000:.1f}s' for t in epoch_times)}]")
            lines.append(f"      Final Accuracy: {tm.get('final_accuracy',0):.4f}, Final Loss: {tm.get('final_loss',0):.4f}")
            lines.append(f"      Batch size: {tm.get('batch_size',0)}, Optimizer: {tm.get('optimizer','?')}, LR: {tm.get('learning_rate',0)}")

            # Layer breakdown
            if layers_det:
                layer_summary = defaultdict(int)
                for l in layers_det:
                    layer_summary[l.get('class_name', '?')] += 1
                lines.append(f"      Capas: {dict(layer_summary)}")
        lines.append("")

    # ─── Section 6: Aggregated Statistics ───
    lines.append("─── SECCIÓN 6: ESTADÍSTICAS AGREGADAS ───")
    lines.append("")

    all_accs = []
    all_params = []
    all_train_times = []
    for exp_id, models in hw_perf.items():
        for mdata in models:
            tm = mdata.get('training_metrics', {})
            mm = mdata.get('model_metrics', {})
            if tm.get('final_accuracy'):
                all_accs.append(tm['final_accuracy'])
            if mm.get('total_parameters'):
                all_params.append(mm['total_parameters'])
            if tm.get('train_time_ms'):
                all_train_times.append(tm['train_time_ms'] / 1000)

    if all_accs:
        lines.append(f"  Total modelos con métricas de entrenamiento: {len(all_accs)}")
        lines.append(f"  Accuracy: media={np.mean(all_accs):.4f}, std={np.std(all_accs):.4f}, min={min(all_accs):.4f}, max={max(all_accs):.4f}")
    if all_params:
        lines.append(f"  Parámetros: media={np.mean(all_params):,.0f}, min={min(all_params):,}, max={max(all_params):,}")
    if all_train_times:
        lines.append(f"  Tiempo entrenamiento: media={np.mean(all_train_times):.0f}s, total={sum(all_train_times)/3600:.1f}h")

    # GPU stats
    all_gpu_utils = []
    all_gpu_mems = []
    all_temps = []
    all_powers = []
    all_idles = []
    for exp_id, models in hw_metrics.items():
        for mid, mdata in models.items():
            s = mdata.get('summary', {})
            avg_gpu = s.get('avg_gpu_utilization', {})
            avg_mem = s.get('avg_gpu_memory', {})
            idle = s.get('idle_time', {})
            if avg_gpu:
                all_gpu_utils.append(list(avg_gpu.values())[0])
            if avg_mem:
                all_gpu_mems.append(list(avg_mem.values())[0])
            if idle.get('average_seconds'):
                all_idles.append(idle['average_seconds'])
            gpu_data = mdata.get('gpu_metrics', [])
            if isinstance(gpu_data, list):
                for record in gpu_data:
                    metrics = record.get('metrics', {})
                    for gid in metrics.get('temperature', {}):
                        t = metrics['temperature'][gid]
                        if isinstance(t, (int, float)):
                            all_temps.append(t)
                    for gid in metrics.get('power', {}):
                        p = metrics['power'][gid]
                        if isinstance(p, dict):
                            all_powers.append(p.get('draw_watts', 0))

    lines.append("")
    if all_gpu_utils:
        lines.append(f"  GPU Utilización promedio: {np.mean(all_gpu_utils):.1f}% (max: {max(all_gpu_utils):.1f}%)")
    if all_gpu_mems:
        lines.append(f"  GPU Memoria promedio: {np.mean(all_gpu_mems):.1f}% (max: {max(all_gpu_mems):.1f}%)")
    if all_temps:
        lines.append(f"  GPU Temperatura: media={np.mean(all_temps):.1f}°C, max={max(all_temps)}°C")
    if all_powers:
        lines.append(f"  GPU Power draw: media={np.mean(all_powers):.1f}W, max={max(all_powers):.1f}W")
    if all_idles:
        lines.append(f"  Idle time promedio: {np.mean(all_idles):.0f}s")

    lines.append("")
    lines.append("=" * 80)
    lines.append("FIN DEL CONTEXTO")
    lines.append("=" * 80)

    output_path = os.path.join(OUTPUT_DIR, 'llm_context_complete.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  CONTEXTO LLM guardado en: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path):,} bytes")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("  GENERADOR COMPLETO DE TABLAS PARA TESIS")
    print("  + Hardware Metrics + Performance Logs")
    print("=" * 80)

    # Load all data
    experiments = load_experiments()
    hw_metrics = load_hardware_metrics()
    hw_perf = load_hardware_perf_logs()

    print(f"  Experimentos JSON cargados: {len(experiments)}")
    print(f"  Hardware Metrics (experimentos): {len(hw_metrics)}")
    print(f"  Hardware Perf Logs (experimentos): {len(hw_perf)}")
    ds_set = set()
    for e in experiments:
        ds = e.get('dataset', None)
        if ds is None or not isinstance(ds, str):
            ds = e.get('experiment_id', e.get('_filename', '')).split('-')[0]
        ds_set.add(ds)
    print(f"  Datasets: {', '.join(sorted(ds_set))}")
    print("=" * 80)

    # ─── Tables 1-9: From Experiment JSONs ───
    table1_resumen(experiments)
    table2_reproducibilidad(experiments)
    table3_exploracion_vs_deep(experiments)
    table4_arquitecturas(experiments)
    table5_infraestructura(experiments)
    table6_eficiencia(experiments)
    table7_cnn_vs_inception(experiments)
    table8_benchmarks(experiments)
    table9_config(experiments)

    # ─── Tables 10-16: From Hardware Metrics & Performance Logs ───
    table10_gpu_utilization(hw_metrics)
    table11_gpu_power_temp(hw_metrics)
    table12_latency(hw_metrics)
    table13_training_epochs(hw_perf)
    table14_idle_time(hw_metrics)
    table15_pcie_clocks(hw_metrics)
    table16_model_complexity(hw_perf)

    # ─── LLM Context ───
    generate_llm_context(experiments, hw_metrics, hw_perf)

    # ─── Summary ───
    print("=" * 80)
    print(f"  COMPLETADO — Archivos generados en: {OUTPUT_DIR}")
    print("=" * 80)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:<55} {size:>8} bytes")
