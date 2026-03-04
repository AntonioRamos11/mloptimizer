#!/usr/bin/env python3
"""
=============================================================================
 GENERADOR DE TABLAS PARA TESIS (Parte 2: Tablas 5-9)
 "Implementación de un Modelo de AutoML en la Nube y su Evaluación
  Utilizando una Base de Datos de Benchmark"
=============================================================================
 Experimentos: 2 GPUs (2 workers × 1× NVIDIA RTX 3080 cada uno)
 Datasets: MNIST (×3), CIFAR-10 (×2), Fashion-MNIST (×2)
=============================================================================
 Salida: LaTeX + PNG + texto plano formateado para contexto LLM
=============================================================================
"""

import json
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'thesis_tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_DISPLAY = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10',
    'fashion_mnist': 'Fashion-MNIST',
}

COLORS = {
    'exploration': '#3498db',
    'deep_training': '#e74c3c',
    'cnn': '#2ecc71',
    'inception': '#9b59b6',
    'improve': '#f39c12',
    'bar1': '#1abc9c',
    'bar2': '#e67e22',
    'bar3': '#3498db',
}

# Known benchmark accuracies for comparison (published results)
BENCHMARKS = {
    'mnist': [
        {'name': 'LeNet-5 (LeCun 1998)',      'accuracy': 0.9920, 'params': '60K',   'type': 'Manual'},
        {'name': 'VGG-16 (adaptado)',          'accuracy': 0.9950, 'params': '15M',   'type': 'Manual'},
        {'name': 'ResNet-18 (adaptado)',       'accuracy': 0.9960, 'params': '11M',   'type': 'Manual'},
        {'name': 'Auto-Keras (Jin 2019)',      'accuracy': 0.9940, 'params': '~2M',   'type': 'AutoML'},
        {'name': 'Google AutoML Vision',       'accuracy': 0.9950, 'params': 'N/A',   'type': 'AutoML (Cloud)'},
        {'name': 'DARTS (Liu 2019)',           'accuracy': 0.9960, 'params': '~3M',   'type': 'NAS'},
    ],
    'cifar10': [
        {'name': 'VGG-16 (Simonyan 2014)',     'accuracy': 0.9340, 'params': '15M',   'type': 'Manual'},
        {'name': 'ResNet-110 (He 2015)',       'accuracy': 0.9370, 'params': '1.7M',  'type': 'Manual'},
        {'name': 'DenseNet-BC (Huang 2017)',   'accuracy': 0.9520, 'params': '0.8M',  'type': 'Manual'},
        {'name': 'Auto-Keras (Jin 2019)',      'accuracy': 0.9530, 'params': '~5M',   'type': 'AutoML'},
        {'name': 'NASNet-A (Zoph 2018)',       'accuracy': 0.9720, 'params': '3.3M',  'type': 'NAS'},
        {'name': 'DARTS (Liu 2019)',           'accuracy': 0.9706, 'params': '3.3M',  'type': 'NAS'},
    ],
    'fashion_mnist': [
        {'name': 'CNN simple (2 conv)',        'accuracy': 0.9200, 'params': '~100K', 'type': 'Manual'},
        {'name': 'VGG-16 (adaptado)',          'accuracy': 0.9350, 'params': '15M',   'type': 'Manual'},
        {'name': 'ResNet-18 (adaptado)',       'accuracy': 0.9460, 'params': '11M',   'type': 'Manual'},
        {'name': 'Auto-Keras (Jin 2019)',      'accuracy': 0.9460, 'params': '~2M',   'type': 'AutoML'},
        {'name': 'Google AutoML Tables',       'accuracy': 0.9500, 'params': 'N/A',   'type': 'AutoML (Cloud)'},
    ],
}

# Collector for LLM context text
llm_context_lines = []

def ctx(text):
    """Print and collect for LLM context."""
    print(text)
    llm_context_lines.append(text)


def load_experiments():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')))
    experiments = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            data['_filename'] = os.path.basename(f)
            experiments.append(data)
    return experiments


def group_by_dataset(experiments):
    groups = defaultdict(list)
    for exp in experiments:
        ds = exp.get('dataset', {}).get('name', 'unknown')
        groups[ds].append(exp)
    return dict(groups)


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 5: Infraestructura Cloud (2 GPUs)
# ═════════════════════════════════════════════════════════════════════════════
def table5_infrastructure(experiments):
    ctx("\n" + "=" * 80)
    ctx("  TABLA 5: Infraestructura del Cluster en la Nube")
    ctx("=" * 80)

    # Collect unique workers across all experiments
    all_workers = []
    seen_keys = set()
    for exp in experiments:
        for wh in exp.get('all_workers_hardware', []):
            gpu_models = tuple(sorted([g.get('model', '') for g in wh.get('gpus', [])]))
            key = (wh.get('cpu_cores', 0), wh.get('gpu_count', 0), gpu_models, wh.get('python_version', ''))
            if key not in seen_keys:
                seen_keys.add(key)
                all_workers.append(wh)

    ctx(f"\n  Total de nodos worker únicos detectados: {len(all_workers)}")
    ctx(f"  Configuración: Master-Slave distribuido vía RabbitMQ (ngrok tunnel)")
    ctx(f"  Protocolo: AMQP con colas 'parameters' (jobs) y 'results' (responses)")
    ctx("")

    # ── Console table ───────────────────────────────────────────────────────
    ctx(f"  {'Worker':<10} {'CPU Cores':<12} {'RAM Total':<14} {'GPU':<30} {'VRAM':<12} {'Driver':<14} {'Python':<10} {'OS':<8}")
    ctx(f"  {'-'*10:<10} {'-'*12:<12} {'-'*14:<14} {'-'*30:<30} {'-'*12:<12} {'-'*14:<14} {'-'*10:<10} {'-'*8:<8}")

    total_gpus = 0
    total_cores = 0
    for i, wh in enumerate(all_workers):
        gpu_list = wh.get('gpus', [])
        gpu_name = gpu_list[0].get('model', 'N/A') if gpu_list else 'CPU-only'
        gpu_mem = gpu_list[0].get('memory', 'N/A') if gpu_list else 'N/A'
        gpu_driver = gpu_list[0].get('driver', 'N/A') if gpu_list else 'N/A'
        n_gpus = wh.get('gpu_count', 0)
        cores = wh.get('cpu_cores', 0)
        total_gpus += n_gpus
        total_cores += cores
        gpu_str = f"{n_gpus}× {gpu_name}" if n_gpus > 0 else "CPU-only"
        ctx(f"  Worker {i+1:<4} {cores:<12} {wh.get('ram_total', 'N/A'):<14} {gpu_str:<30} {gpu_mem:<12} {gpu_driver:<14} {wh.get('python_version', '?'):<10} {wh.get('system', '?'):<8}")

    ctx(f"\n  RESUMEN CLUSTER:")
    ctx(f"    Nodos totales:     {len(all_workers)}")
    ctx(f"    GPUs totales:      {total_gpus}")
    ctx(f"    CPU cores totales: {total_cores}")
    ctx(f"    VRAM total:        {total_gpus * 10240} MiB ({total_gpus * 10:.0f} GB)")
    ctx(f"    Comunicación:      RabbitMQ vía ngrok (AMQP sobre TCP tunnel)")
    ctx(f"    Estrategia:        Master genera arquitecturas (Optuna), Slaves entrenan en paralelo")

    # ── LaTeX ───────────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Especificaciones del cluster de cómputo en la nube}")
    latex.append(r"\label{tab:infraestructura}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{clcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Nodo} & \textbf{GPU} & \textbf{VRAM} & \textbf{CPU Cores} & \textbf{RAM} & \textbf{Driver} \\")
    latex.append(r"\midrule")

    for i, wh in enumerate(all_workers):
        gpu_list = wh.get('gpus', [])
        gpu_name = gpu_list[0].get('model', 'N/A') if gpu_list else 'CPU-only'
        gpu_mem = gpu_list[0].get('memory', 'N/A') if gpu_list else 'N/A'
        gpu_driver = gpu_list[0].get('driver', 'N/A') if gpu_list else 'N/A'
        cores = wh.get('cpu_cores', 0)
        ram = wh.get('ram_total', 'N/A')
        latex.append(f"  Worker {i+1} & {gpu_name} & {gpu_mem} & {cores} & {ram} & {gpu_driver} \\\\")

    latex.append(r"\midrule")
    latex.append(f"  \\textbf{{Total}} & \\textbf{{{total_gpus} GPUs}} & \\textbf{{{total_gpus * 10:.0f} GB}} & \\textbf{{{total_cores}}} & -- & -- \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla5_infraestructura.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    ctx(f"\n  LaTeX → {latex_path}")

    return all_workers


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 6: Eficiencia del Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def table6_efficiency(experiments):
    ctx("\n" + "=" * 80)
    ctx("  TABLA 6: Eficiencia del Pipeline de Optimización")
    ctx("=" * 80)

    rows = []
    for exp in experiments:
        pm = exp['performance_metrics']
        os_stats = exp.get('optimization_stats', {})
        tc = exp.get('training_config', {})
        elapsed = pm.get('elapsed_seconds', 0)
        models_gen = os_stats.get('models_generated', 0)
        models_proc = os_stats.get('models_processed', 0)
        expl_models = os_stats.get('exploration_models', 0)
        deep_models = os_stats.get('deep_training_models', 0)

        time_per_model = elapsed / models_proc if models_proc > 0 else 0
        throughput = (models_proc / elapsed) * 3600 if elapsed > 0 else 0  # models/hour

        rows.append({
            'experiment_id': exp['experiment_id'],
            'dataset': DATASET_DISPLAY.get(exp['dataset']['name'], exp['dataset']['name']),
            'ds_key': exp['dataset']['name'],
            'elapsed_s': elapsed,
            'elapsed_fmt': pm.get('elapsed_time', '?'),
            'models_gen': models_gen,
            'models_proc': models_proc,
            'expl_models': expl_models,
            'deep_models': deep_models,
            'time_per_model': time_per_model,
            'throughput': throughput,
            'exploration_size': tc.get('exploration_size', '?'),
            'hof_size': tc.get('hall_of_fame_size', '?'),
            'expl_epochs': tc.get('exploration_epochs', '?'),
            'deep_epochs': tc.get('deep_training_epochs', '?'),
        })

    # Console
    ctx(f"\n  {'Experimento':<35} {'Dataset':<16} {'Generados':<10} {'Procesados':<12} {'Expl.':<7} {'Deep':<6} {'T/Modelo':<10} {'Modelos/h':<10} {'Tiempo':<10}")
    ctx(f"  {'-'*35} {'-'*16} {'-'*10} {'-'*12} {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for r in rows:
        t_model_fmt = f"{r['time_per_model']:.0f}s"
        ctx(f"  {r['experiment_id']:<35} {r['dataset']:<16} {r['models_gen']:<10} {r['models_proc']:<12} {r['expl_models']:<7} {r['deep_models']:<6} {t_model_fmt:<10} {r['throughput']:<10.1f} {r['elapsed_fmt']:<10}")

    # Aggregated stats
    ctx(f"\n  ESTADÍSTICAS AGREGADAS:")
    all_throughput = [r['throughput'] for r in rows]
    all_tpm = [r['time_per_model'] for r in rows]
    all_elapsed = [r['elapsed_s'] for r in rows]
    total_models = sum(r['models_proc'] for r in rows)
    total_time = sum(r['elapsed_s'] for r in rows)
    ctx(f"    Total de modelos entrenados:         {total_models}")
    ctx(f"    Tiempo total acumulado:              {total_time/3600:.1f} horas")
    ctx(f"    Throughput promedio:                  {np.mean(all_throughput):.1f} modelos/hora")
    ctx(f"    Tiempo promedio por modelo:           {np.mean(all_tpm):.0f} segundos")
    ctx(f"    Tiempo promedio por experimento:      {np.mean(all_elapsed)/60:.1f} minutos")
    ctx(f"    Eficiencia pipeline (proc/gen):       {total_models/sum(r['models_gen'] for r in rows)*100:.0f}%")

    # Per-dataset aggregated
    ctx(f"\n  EFICIENCIA POR DATASET:")
    ctx(f"  {'Dataset':<18} {'Runs':<6} {'Media Tiempo':<14} {'Media T/Modelo':<16} {'Media Throughput':<16}")
    ctx(f"  {'-'*18} {'-'*6} {'-'*14} {'-'*16} {'-'*16}")
    groups = group_by_dataset(experiments)
    ds_stats = {}
    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        ds_rows = [r for r in rows if r['ds_key'] == ds_key]
        if not ds_rows:
            continue
        mean_elapsed = np.mean([r['elapsed_s'] for r in ds_rows])
        mean_tpm = np.mean([r['time_per_model'] for r in ds_rows])
        mean_tp = np.mean([r['throughput'] for r in ds_rows])
        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        elapsed_fmt = f"{int(mean_elapsed//3600):02d}:{int((mean_elapsed%3600)//60):02d}:{int(mean_elapsed%60):02d}"
        ctx(f"  {ds_name:<18} {len(ds_rows):<6} {elapsed_fmt:<14} {mean_tpm:<16.0f}s {mean_tp:<16.1f} mod/h")
        ds_stats[ds_key] = {'mean_elapsed': mean_elapsed, 'mean_tpm': mean_tpm, 'mean_tp': mean_tp, 'n': len(ds_rows)}

    # ── LaTeX ───────────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Eficiencia del pipeline de optimización distribuido (2 GPUs)}")
    latex.append(r"\label{tab:eficiencia_pipeline}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Experimento} & \textbf{Modelos} & \textbf{Exploración} & \textbf{Deep Training} & \textbf{T/Modelo (s)} & \textbf{Throughput (mod/h)} \\")
    latex.append(r"\midrule")

    prev_ds = None
    for r in rows:
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex.append(r"\midrule")
        prev_ds = r['dataset']
        short_id = r['experiment_id'].split('-', 1)[-1]
        latex.append(f"  {ds_display} & {short_id} & {r['models_proc']} & {r['expl_models']} & {r['deep_models']} & {r['time_per_model']:.0f} & {r['throughput']:.1f} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla6_eficiencia_pipeline.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    ctx(f"\n  LaTeX → {latex_path}")

    # ── Figure: throughput + time per model ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = [f"{r['dataset']}\n#{i+1}" for i, r in enumerate(rows)]
    x = np.arange(len(rows))

    # Left: time per model
    bars1 = ax1.bar(x, [r['time_per_model'] / 60 for r in rows], color=COLORS['bar1'], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Minutos por modelo', fontsize=12)
    ax1.set_title('Tiempo Promedio por Modelo', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{bar.get_height():.1f}m', ha='center', va='bottom', fontsize=9)

    # Right: throughput
    bars2 = ax2.bar(x, [r['throughput'] for r in rows], color=COLORS['bar2'], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Modelos por hora', fontsize=12)
    ax2.set_title('Throughput del Pipeline', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Eficiencia del Pipeline Distribuido (2 GPUs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig6_eficiencia_pipeline.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    ctx(f"  Figura → {fig_path}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 7: CNN vs Inception por Dataset
# ═════════════════════════════════════════════════════════════════════════════
def table7_cnn_vs_inception(experiments):
    ctx("\n" + "=" * 80)
    ctx("  TABLA 7: Comparación CNN vs Inception por Dataset")
    ctx("=" * 80)

    groups = group_by_dataset(experiments)

    # Collect per-dataset per-architecture stats from leaderboard
    ctx(f"\n  {'Dataset':<18} {'Arquitectura':<14} {'Modelos':<9} {'Mejor Acc':<12} {'Media Acc':<12} {'Ganó Best?':<12}")
    ctx(f"  {'-'*18} {'-'*14} {'-'*9} {'-'*12} {'-'*12} {'-'*12}")

    all_rows = []
    chart_data = {}  # ds_key -> {arch -> [accs]}

    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        exps = groups.get(ds_key, [])
        if not exps:
            continue

        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        arch_accs = defaultdict(list)  # arch -> list of best_performance per leaderboard entry (deep training only)

        # Count how many times each arch won (was best model)
        winner_counts = defaultdict(int)
        for exp in exps:
            best_arch = exp.get('architecture', {}).get('base_architecture', '?')
            winner_counts[best_arch] += 1

        # Gather all leaderboard performances
        for exp in exps:
            for entry in exp.get('leaderboard', []):
                arch = entry.get('base_architecture', '?')
                if entry.get('phase') == 'deep_training':
                    perf = entry.get('performance_2', entry.get('performance', 0))
                    if perf and perf > 0:
                        arch_accs[arch].append(perf)
                else:
                    arch_accs[arch].append(entry.get('performance', 0))

        chart_data[ds_key] = dict(arch_accs)

        for arch in ['cnn', 'inception']:
            accs = arch_accs.get(arch, [])
            if not accs:
                continue
            best = max(accs)
            mean = np.mean(accs)
            wins = winner_counts.get(arch, 0)
            won_str = f"Sí ({wins}×)" if wins > 0 else "No"
            row = {
                'dataset': ds_name,
                'ds_key': ds_key,
                'arch': arch.upper(),
                'count': len(accs),
                'best': best,
                'mean': mean,
                'won': won_str,
                'wins': wins,
            }
            all_rows.append(row)
            ctx(f"  {ds_name:<18} {arch.upper():<14} {len(accs):<9} {best:<12.4f} {mean:<12.4f} {won_str:<12}")
        ctx("")

    # Summary
    ctx(f"  RESUMEN CNN vs Inception:")
    cnn_wins = sum(1 for r in all_rows if r['arch'] == 'CNN' and r['wins'] > 0)
    inc_wins = sum(1 for r in all_rows if r['arch'] == 'INCEPTION' and r['wins'] > 0)
    ctx(f"    CNN ganó en:       {cnn_wins} datasets")
    ctx(f"    Inception ganó en: {inc_wins} datasets")
    ctx(f"    El AutoML explora ambas arquitecturas y selecciona la mejor para cada dataset.")

    # ── LaTeX ───────────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Comparación de rendimiento entre arquitecturas CNN e Inception}")
    latex.append(r"\label{tab:cnn_vs_inception}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Arquitectura} & \textbf{Modelos} & \textbf{Mejor Acc.} & \textbf{Media Acc.} & \textbf{Ganadora} \\")
    latex.append(r"\midrule")

    prev_ds = None
    for r in all_rows:
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex.append(r"\midrule")
        prev_ds = r['dataset']
        won_latex = r"\textbf{Sí}" if r['wins'] > 0 else "No"
        acc_best = f"\\textbf{{{r['best']:.4f}}}" if r['wins'] > 0 else f"{r['best']:.4f}"
        latex.append(f"  {ds_display} & {r['arch']} & {r['count']} & {acc_best} & {r['mean']:.4f} & {won_latex} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla7_cnn_vs_inception.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    ctx(f"  LaTeX → {latex_path}")

    # ── Figure: grouped bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets_in_chart = [k for k in ['mnist', 'cifar10', 'fashion_mnist'] if k in chart_data]
    x = np.arange(len(datasets_in_chart))
    width = 0.35

    cnn_best = []
    inc_best = []
    for ds_key in datasets_in_chart:
        cnn_accs = chart_data[ds_key].get('cnn', [0])
        inc_accs = chart_data[ds_key].get('inception', [0])
        cnn_best.append(max(cnn_accs) if cnn_accs else 0)
        inc_best.append(max(inc_accs) if inc_accs else 0)

    bars_cnn = ax.bar(x - width/2, cnn_best, width, label='CNN', color=COLORS['cnn'], alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_inc = ax.bar(x + width/2, inc_best, width, label='Inception', color=COLORS['inception'], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Mejor Accuracy', fontsize=12)
    ax.set_title('Mejor Accuracy: CNN vs Inception por Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY[k] for k in datasets_in_chart], fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(bottom=0.95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.grid(axis='y', alpha=0.3)

    # Labels on bars
    for bar in bars_cnn:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_inc:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Star the winner
    for i, ds_key in enumerate(datasets_in_chart):
        if cnn_best[i] > inc_best[i]:
            ax.annotate('★', (i - width/2, cnn_best[i] + 0.002), ha='center', fontsize=14, color='gold')
        elif inc_best[i] > cnn_best[i]:
            ax.annotate('★', (i + width/2, inc_best[i] + 0.002), ha='center', fontsize=14, color='gold')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig7_cnn_vs_inception.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    ctx(f"  Figura → {fig_path}")

    return all_rows


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 8: Comparación con Benchmarks Publicados
# ═════════════════════════════════════════════════════════════════════════════
def table8_benchmarks(experiments):
    ctx("\n" + "=" * 80)
    ctx("  TABLA 8: Comparación con Benchmarks Publicados")
    ctx("=" * 80)

    groups = group_by_dataset(experiments)

    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        exps = groups.get(ds_key, [])
        if not exps:
            continue

        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        # Best accuracy from our system
        our_best = max(e['performance_metrics'].get('best_deep_training_accuracy', 0) for e in exps)
        our_arch = None
        for e in exps:
            if e['performance_metrics'].get('best_deep_training_accuracy', 0) == our_best:
                a = e.get('architecture', {})
                our_arch = f"{a.get('base_architecture', '?').upper()} + {a.get('classifier_layer_type', '?').upper()}"
                break

        ctx(f"\n  ─── {ds_name} ───{'─' * (60 - len(ds_name))}")
        ctx(f"  {'Método':<35} {'Accuracy':<12} {'Tipo':<16} {'vs Nuestro':<12}")
        ctx(f"  {'-'*35} {'-'*12} {'-'*16} {'-'*12}")

        benchmarks = BENCHMARKS.get(ds_key, [])
        for bm in benchmarks:
            diff = our_best - bm['accuracy']
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            ctx(f"  {bm['name']:<35} {bm['accuracy']:<12.4f} {bm['type']:<16} {diff_str:<12}")

        # Our result
        ctx(f"  {'─' * 75}")
        ctx(f"  {'NUESTRO AUTOML':<35} {our_best:<12.4f} {'AutoML (Cloud)':<16} {'---':<12}")
        ctx(f"  Arquitectura descubierta: {our_arch}")

        # Analysis
        better_than = [bm for bm in benchmarks if our_best >= bm['accuracy']]
        worse_than = [bm for bm in benchmarks if our_best < bm['accuracy']]
        ctx(f"  Supera a: {len(better_than)}/{len(benchmarks)} benchmarks conocidos")
        if worse_than:
            ctx(f"  Por debajo de: {', '.join(bm['name'] for bm in worse_than)}")

    # ── LaTeX (one table per dataset for clarity) ───────────────────────────
    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        exps = groups.get(ds_key, [])
        if not exps:
            continue

        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        our_best = max(e['performance_metrics'].get('best_deep_training_accuracy', 0) for e in exps)
        our_arch = "AutoML propuesto"

        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{Comparación con benchmarks publicados — {ds_name}}}")
        latex.append(f"\\label{{tab:benchmark_{ds_key}}}")
        latex.append(r"\begin{tabular}{llcc}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Método} & \textbf{Tipo} & \textbf{Accuracy} & \textbf{$\Delta$ vs Nuestro} \\")
        latex.append(r"\midrule")

        benchmarks = BENCHMARKS.get(ds_key, [])
        for bm in benchmarks:
            diff = our_best - bm['accuracy']
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            latex.append(f"  {bm['name']} & {bm['type']} & {bm['accuracy']:.4f} & {diff_str} \\\\")

        latex.append(r"\midrule")
        latex.append(f"  \\textbf{{Nuestro AutoML}} & \\textbf{{AutoML (Cloud)}} & \\textbf{{{our_best:.4f}}} & --- \\\\")
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        latex_path = os.path.join(OUTPUT_DIR, f'tabla8_benchmark_{ds_key}.tex')
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex))
        ctx(f"\n  LaTeX → {latex_path}")

    # ── Combined figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, ds_key in zip(axes, ['mnist', 'cifar10', 'fashion_mnist']):
        exps = groups.get(ds_key, [])
        if not exps:
            continue

        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        our_best = max(e['performance_metrics'].get('best_deep_training_accuracy', 0) for e in exps)
        benchmarks = BENCHMARKS.get(ds_key, [])

        names = [bm['name'].split('(')[0].strip() for bm in benchmarks] + ['Nuestro\nAutoML']
        accs = [bm['accuracy'] for bm in benchmarks] + [our_best]
        colors = []
        for bm in benchmarks:
            if 'AutoML' in bm['type'] or 'NAS' in bm['type']:
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')
        colors.append('#2ecc71')  # ours

        y = np.arange(len(names))
        bars = ax.barh(y, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5, height=0.6)

        # Highlight ours
        bars[-1].set_edgecolor('#e74c3c')
        bars[-1].set_linewidth(2)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Accuracy', fontsize=11)
        ax.set_title(f'{ds_name}', fontsize=13, fontweight='bold')

        # Set xlim to zoom in
        min_acc = min(accs) - 0.01
        ax.set_xlim(left=min_acc, right=1.005)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.grid(axis='x', alpha=0.3)

        # Value labels
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{acc:.4f}', va='center', fontsize=8, fontweight='bold')

        # Vertical line for our accuracy
        ax.axvline(x=our_best, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Diseño manual'),
        Patch(facecolor='#e74c3c', label='AutoML/NAS (otros)'),
        Patch(facecolor='#2ecc71', edgecolor='#e74c3c', linewidth=2, label='Nuestro AutoML'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Comparación con Benchmarks Publicados por Dataset', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig_path = os.path.join(OUTPUT_DIR, 'fig8_comparacion_benchmarks.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    ctx(f"  Figura → {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 9: Configuración del Espacio de Búsqueda
# ═════════════════════════════════════════════════════════════════════════════
def table9_search_space(experiments):
    ctx("\n" + "=" * 80)
    ctx("  TABLA 9: Configuración del Espacio de Búsqueda y Parámetros del AutoML")
    ctx("=" * 80)

    # Get unique configs
    configs = {}
    for exp in experiments:
        tc = exp.get('training_config', {})
        key = json.dumps(tc, sort_keys=True)
        if key not in configs:
            configs[key] = {
                'config': tc,
                'experiments': [exp['experiment_id']],
                'datasets': [exp['dataset']['name']],
            }
        else:
            configs[key]['experiments'].append(exp['experiment_id'])
            configs[key]['datasets'].append(exp['dataset']['name'])

    ctx(f"\n  Configuraciones únicas encontradas: {len(configs)}")

    for i, (key, cfg_info) in enumerate(configs.items()):
        tc = cfg_info['config']
        ctx(f"\n  ─── Configuración {i+1} (usada en {len(cfg_info['experiments'])} experimentos) ───")
        ctx(f"    Experimentos: {', '.join(cfg_info['experiments'])}")
        ctx(f"    Datasets:     {', '.join(set(cfg_info['datasets']))}")
        ctx(f"    ┌─────────────────────────────────────────────┐")
        ctx(f"    │ Parámetro                      │ Valor      │")
        ctx(f"    ├─────────────────────────────────────────────┤")
        ctx(f"    │ Trials (Optuna)                 │ {tc.get('trials', '?'):<10} │")
        ctx(f"    │ Exploration Size                │ {tc.get('exploration_size', '?'):<10} │")
        ctx(f"    │ Exploration Epochs              │ {tc.get('exploration_epochs', '?'):<10} │")
        ctx(f"    │ Hall of Fame Size               │ {tc.get('hall_of_fame_size', '?'):<10} │")
        ctx(f"    │ Deep Training Epochs            │ {tc.get('deep_training_epochs', '?'):<10} │")
        ctx(f"    │ HoF Early Stopping Patience     │ {tc.get('hof_early_stopping_patience', '?'):<10} │")
        ctx(f"    └─────────────────────────────────────────────┘")

    # Fixed search space parameters
    ctx(f"\n  ─── Espacio de Búsqueda de Arquitecturas (fijo para todos los experimentos) ───")
    ctx(f"    Tipo de modelo:         Clasificación de imágenes")
    ctx(f"    Arquitecturas base:     CNN, Inception")
    ctx(f"    Clasificadores:         GAP (Global Average Pooling), MLP")
    ctx(f"    Optimizador:            Adam")
    ctx(f"    Función de pérdida:     Sparse Categorical Crossentropy")
    ctx(f"    Métrica:                Accuracy")
    ctx(f"    Activación oculta:      ReLU")
    ctx(f"    Activación salida:      Softmax")
    ctx(f"    Inicialización pesos:   He Uniform")
    ctx(f"    Padding:                Same")
    ctx(f"    Tipo de datos:          float32")
    ctx(f"    Batch size:             4")
    ctx(f"    Validation split:       0.2")

    ctx(f"\n  ─── Rangos del Espacio de Búsqueda (Optuna) ───")
    ctx(f"    CNN:")
    ctx(f"      Bloques conv:         1-5")
    ctx(f"      Capas conv/bloque:    1-5")
    ctx(f"      Filtros:              16-256")
    ctx(f"      Tamaño kernel:        3, 5")
    ctx(f"      Max pooling:          2, 3")
    ctx(f"      Dropout:              0.0-0.5")
    ctx(f"    Inception:")
    ctx(f"      Bloques Inception:    1-5")
    ctx(f"      Módulos/bloque:       1-3")
    ctx(f"      Stem filtros:         8-128")
    ctx(f"      Conv 1×1 filtros:     4-64")
    ctx(f"      Conv 3×3 filtros:     16-128")
    ctx(f"      Conv 5×5 filtros:     8-64")
    ctx(f"      Pool conv filtros:    4-32")
    ctx(f"    Clasificador:")
    ctx(f"      Tipo:                 GAP o MLP")
    ctx(f"      Capas densas MLP:     0-3")
    ctx(f"      Unidades/capa:        16-512")
    ctx(f"      Dropout MLP:          0.0-0.5")

    # ── LaTeX: training config ──────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Configuración del proceso de optimización AutoML}")
    latex.append(r"\label{tab:config_automl}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Parámetro} & \textbf{Config. 1 (inicial)} & \textbf{Config. 2 (optimizada)} \\")
    latex.append(r"\midrule")

    sorted_configs = sorted(configs.values(), key=lambda c: c['config'].get('trials', 0), reverse=True)
    if len(sorted_configs) == 1:
        tc = sorted_configs[0]['config']
        latex[-3] = r"\begin{tabular}{lc}"
        latex[-2] = r"\toprule"
        latex[-1] = r"\textbf{Parámetro} & \textbf{Valor} \\"
        latex.append(r"\midrule")
        latex.append(f"  Trials (Optuna) & {tc.get('trials', '?')} \\\\")
        latex.append(f"  Tamaño de exploración & {tc.get('exploration_size', '?')} modelos \\\\")
        latex.append(f"  Épocas de exploración & {tc.get('exploration_epochs', '?')} \\\\")
        latex.append(f"  Hall of Fame (mejores modelos) & {tc.get('hall_of_fame_size', '?')} \\\\")
        latex.append(f"  Épocas de deep training & {tc.get('deep_training_epochs', '?')} \\\\")
        latex.append(f"  Early stopping patience & {tc.get('hof_early_stopping_patience', '?')} \\\\")
    else:
        tc1 = sorted_configs[0]['config']
        tc2 = sorted_configs[-1]['config']
        latex.append(f"  Trials (Optuna) & {tc1.get('trials', '?')} & {tc2.get('trials', '?')} \\\\")
        latex.append(f"  Tamaño de exploración & {tc1.get('exploration_size', '?')} & {tc2.get('exploration_size', '?')} \\\\")
        latex.append(f"  Épocas de exploración & {tc1.get('exploration_epochs', '?')} & {tc2.get('exploration_epochs', '?')} \\\\")
        latex.append(f"  Hall of Fame & {tc1.get('hall_of_fame_size', '?')} & {tc2.get('hall_of_fame_size', '?')} \\\\")
        latex.append(f"  Épocas deep training & {tc1.get('deep_training_epochs', '?')} & {tc2.get('deep_training_epochs', '?')} \\\\")
        latex.append(f"  Early stopping patience & {tc1.get('hof_early_stopping_patience', '?')} & {tc2.get('hof_early_stopping_patience', '?')} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla9_configuracion_automl.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    ctx(f"\n  LaTeX → {latex_path}")

    # ── LaTeX: search space ranges ──────────────────────────────────────────
    latex2 = []
    latex2.append(r"\begin{table}[htbp]")
    latex2.append(r"\centering")
    latex2.append(r"\caption{Espacio de búsqueda de arquitecturas para el AutoML}")
    latex2.append(r"\label{tab:search_space}")
    latex2.append(r"\small")
    latex2.append(r"\begin{tabular}{llc}")
    latex2.append(r"\toprule")
    latex2.append(r"\textbf{Componente} & \textbf{Parámetro} & \textbf{Rango} \\")
    latex2.append(r"\midrule")
    latex2.append(r"CNN & Bloques convolucionales & 1--5 \\")
    latex2.append(r"    & Capas conv/bloque & 1--5 \\")
    latex2.append(r"    & Filtros por capa & 16--256 \\")
    latex2.append(r"    & Tamaño de kernel & \{3, 5\} \\")
    latex2.append(r"    & Dropout & 0.0--0.5 \\")
    latex2.append(r"\midrule")
    latex2.append(r"Inception & Bloques Inception & 1--5 \\")
    latex2.append(r"          & Módulos por bloque & 1--3 \\")
    latex2.append(r"          & Filtros stem & 8--128 \\")
    latex2.append(r"          & Filtros conv 3$\times$3 & 16--128 \\")
    latex2.append(r"          & Filtros conv 5$\times$5 & 8--64 \\")
    latex2.append(r"\midrule")
    latex2.append(r"Clasificador & Tipo & \{GAP, MLP\} \\")
    latex2.append(r"             & Capas densas (MLP) & 0--3 \\")
    latex2.append(r"             & Unidades por capa & 16--512 \\")
    latex2.append(r"             & Dropout & 0.0--0.5 \\")
    latex2.append(r"\bottomrule")
    latex2.append(r"\end{tabular}")
    latex2.append(r"\end{table}")

    latex2_path = os.path.join(OUTPUT_DIR, 'tabla9b_search_space.tex')
    with open(latex2_path, 'w') as f:
        f.write('\n'.join(latex2))
    ctx(f"  LaTeX → {latex2_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ctx("=" * 80)
    ctx("  GENERADOR DE TABLAS PARA TESIS — Parte 2 (Tablas 5-9)")
    ctx("  Implementación de un Modelo de AutoML en la Nube")
    ctx("  Experimentos con 2 GPUs (NVIDIA RTX 3080)")
    ctx("=" * 80)

    experiments = load_experiments()
    if not experiments:
        print("ERROR: No se encontraron archivos JSON de experimentos.")
        sys.exit(1)

    experiments.sort(key=lambda e: (e['dataset']['name'], e['experiment_id']))

    ctx(f"  Experimentos cargados: {len(experiments)}")
    ctx(f"  Datasets: {', '.join(sorted(set(e['dataset']['name'] for e in experiments)))}")
    ctx("")

    # Generate tables 5-9
    table5_infrastructure(experiments)
    table6_efficiency(experiments)
    table7_cnn_vs_inception(experiments)
    table8_benchmarks(experiments)
    table9_search_space(experiments)

    # ── Save LLM context file ──────────────────────────────────────────────
    llm_path = os.path.join(OUTPUT_DIR, 'llm_context_tables5_9.txt')
    with open(llm_path, 'w') as f:
        f.write('\n'.join(llm_context_lines))

    ctx(f"\n" + "=" * 80)
    ctx(f"  COMPLETADO — Archivos generados en: {OUTPUT_DIR}")
    ctx(f"=" * 80)

    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        ctx(f"  {fname:<55} {size:>6} bytes")

    ctx(f"\n  CONTEXTO LLM guardado en: {llm_path}")
    ctx(f"  (Copia el contenido de ese archivo para dar contexto a otro LLM)")


if __name__ == '__main__':
    main()
