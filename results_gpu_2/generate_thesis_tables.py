#!/usr/bin/env python3
"""
=============================================================================
 GENERADOR DE TABLAS PARA TESIS:
 "Implementación de un Modelo de AutoML en la Nube y su Evaluación
  Utilizando una Base de Datos de Benchmark"
=============================================================================
 Experimentos: 2 GPUs (2 workers × 1× NVIDIA RTX 3080 cada uno)
 Datasets: MNIST (×3), CIFAR-10 (×2), Fashion-MNIST (×2)
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

# Nice colors
COLORS = {
    'exploration': '#3498db',
    'deep_training': '#e74c3c',
    'cnn': '#2ecc71',
    'inception': '#9b59b6',
    'improve': '#f39c12',
}


def load_experiments():
    """Load all experiment JSON files."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')))
    experiments = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            data['_filename'] = os.path.basename(f)
            experiments.append(data)
    print(f"Loaded {len(experiments)} experiments from {RESULTS_DIR}")
    return experiments


def group_by_dataset(experiments):
    """Group experiments by dataset name."""
    groups = defaultdict(list)
    for exp in experiments:
        ds = exp.get('dataset', {}).get('name', 'unknown')
        groups[ds].append(exp)
    return dict(groups)


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 1: Resumen de Mejores Resultados por Experimento
# ═════════════════════════════════════════════════════════════════════════════
def table1_best_accuracy(experiments):
    """Generate main results summary table."""
    print("\n" + "=" * 80)
    print("  TABLA 1: Resumen de Mejores Resultados por Experimento")
    print("=" * 80)

    # ── Console table ───────────────────────────────────────────────────────
    header = f"{'Experimento':<35} {'Dataset':<16} {'Acc Expl.':<12} {'Acc Deep':<12} {'Arq.':<12} {'Clasif.':<8} {'Tiempo':<10}"
    print(header)
    print("-" * len(header))

    rows = []
    for exp in experiments:
        pm = exp.get('performance_metrics', {})
        arch = exp.get('architecture', {})
        rows.append({
            'experiment_id': exp['experiment_id'],
            'dataset': DATASET_DISPLAY.get(exp['dataset']['name'], exp['dataset']['name']),
            'acc_exploration': pm.get('best_exploration_accuracy', 0),
            'acc_deep': pm.get('best_deep_training_accuracy', 0),
            'base_arch': arch.get('base_architecture', '?').upper(),
            'classifier': arch.get('classifier_layer_type', '?').upper(),
            'elapsed': pm.get('elapsed_time', '?'),
            'elapsed_s': pm.get('elapsed_seconds', 0),
        })

    for r in rows:
        print(f"{r['experiment_id']:<35} {r['dataset']:<16} {r['acc_exploration']:<12.4f} {r['acc_deep']:<12.4f} {r['base_arch']:<12} {r['classifier']:<8} {r['elapsed']:<10}")

    # ── LaTeX table ─────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Resumen de mejores resultados por experimento (2 GPUs en la nube)}")
    latex.append(r"\label{tab:resultados_resumen}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llccclr}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Experimento} & \textbf{Acc. Exploración} & \textbf{Acc. Deep Training} & \textbf{$\Delta$ Acc.} & \textbf{Arquitectura} & \textbf{Tiempo} \\")
    latex.append(r"\midrule")

    prev_ds = None
    for r in rows:
        delta = r['acc_deep'] - r['acc_exploration']
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        arch_str = f"{r['base_arch']} + {r['classifier']}"
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex.append(r"\midrule")
        prev_ds = r['dataset']
        # Short experiment ID (remove dataset prefix for readability)
        short_id = r['experiment_id'].split('-', 1)[-1] if '-' in r['experiment_id'] else r['experiment_id']
        latex.append(f"  {ds_display} & {short_id} & {r['acc_exploration']:.4f} & {r['acc_deep']:.4f} & {delta_str} & {arch_str} & {r['elapsed']} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla1_resumen_resultados.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"\n  LaTeX → {latex_path}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 2: Consistencia y Reproducibilidad del AutoML
# ═════════════════════════════════════════════════════════════════════════════
def table2_reproducibility(experiments):
    """Generate reproducibility/consistency table across repeated runs."""
    print("\n" + "=" * 80)
    print("  TABLA 2: Consistencia y Reproducibilidad del AutoML")
    print("=" * 80)

    groups = group_by_dataset(experiments)

    header = f"{'Dataset':<18} {'Runs':<6} {'Media':<10} {'Desv.Est.':<10} {'Mín':<10} {'Máx':<10} {'Rango':<10}"
    print(header)
    print("-" * len(header))

    stats_rows = []
    for ds_key in ['mnist', 'cifar10', 'fashion_mnist']:
        exps = groups.get(ds_key, [])
        if not exps:
            continue
        accs = [e['performance_metrics']['best_deep_training_accuracy'] for e in exps]
        times = [e['performance_metrics']['elapsed_seconds'] for e in exps]
        accs_expl = [e['performance_metrics']['best_exploration_accuracy'] for e in exps]

        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        range_acc = max_acc - min_acc

        stats_rows.append({
            'dataset': ds_name,
            'ds_key': ds_key,
            'n': len(exps),
            'mean': mean_acc,
            'std': std_acc,
            'min': min_acc,
            'max': max_acc,
            'range': range_acc,
            'mean_expl': np.mean(accs_expl),
            'std_expl': np.std(accs_expl),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'all_accs': accs,
            'all_times': times,
        })
        print(f"{ds_name:<18} {len(exps):<6} {mean_acc:<10.4f} {std_acc:<10.4f} {min_acc:<10.4f} {max_acc:<10.4f} {range_acc:<10.4f}")

    # ── LaTeX table ─────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Consistencia y reproducibilidad del AutoML en múltiples ejecuciones}")
    latex.append(r"\label{tab:reproducibilidad}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Runs} & \textbf{Media Acc.} & \textbf{Desv. Est.} & \textbf{Mín.} & \textbf{Máx.} & \textbf{Rango} \\")
    latex.append(r"\midrule")

    for r in stats_rows:
        latex.append(f"  {r['dataset']} & {r['n']} & {r['mean']:.4f} & {r['std']:.4f} & {r['min']:.4f} & {r['max']:.4f} & {r['range']:.4f} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla2_reproducibilidad.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"\n  LaTeX → {latex_path}")

    # ── Extended table: with time stats ─────────────────────────────────────
    latex2 = []
    latex2.append(r"\begin{table}[htbp]")
    latex2.append(r"\centering")
    latex2.append(r"\caption{Estadísticas de tiempo de ejecución por dataset}")
    latex2.append(r"\label{tab:reproducibilidad_tiempo}")
    latex2.append(r"\begin{tabular}{lccc}")
    latex2.append(r"\toprule")
    latex2.append(r"\textbf{Dataset} & \textbf{Runs} & \textbf{Tiempo Medio} & \textbf{Desv. Est. (s)} \\")
    latex2.append(r"\midrule")
    for r in stats_rows:
        mean_t = r['mean_time']
        mean_fmt = f"{int(mean_t//3600):02d}:{int((mean_t%3600)//60):02d}:{int(mean_t%60):02d}"
        latex2.append(f"  {r['dataset']} & {r['n']} & {mean_fmt} & {r['std_time']:.0f} s \\\\")
    latex2.append(r"\bottomrule")
    latex2.append(r"\end{tabular}")
    latex2.append(r"\end{table}")

    latex2_path = os.path.join(OUTPUT_DIR, 'tabla2b_tiempo_reproducibilidad.tex')
    with open(latex2_path, 'w') as f:
        f.write('\n'.join(latex2))
    print(f"  LaTeX → {latex2_path}")

    # ── Box plot: accuracy distribution per dataset ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = []
    labels = []
    data_boxes = []
    for i, r in enumerate(stats_rows):
        data_boxes.append(r['all_accs'])
        positions.append(i)
        labels.append(r['dataset'])

    bp = ax.boxplot(data_boxes, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    box_colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], box_colors[:len(data_boxes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Also scatter individual points
    for i, r in enumerate(stats_rows):
        ax.scatter([i] * len(r['all_accs']), r['all_accs'], c='black', s=40, zorder=5, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Accuracy (Deep Training)', fontsize=12)
    ax.set_title('Distribución de Accuracy por Dataset\n(Múltiples ejecuciones, 2 GPUs)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, 'fig2_reproducibilidad_boxplot.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura → {fig_path}")

    return stats_rows


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 3: Mejora Exploración → Deep Training (+ Gráfica)
# ═════════════════════════════════════════════════════════════════════════════
def table3_exploration_vs_deep(experiments):
    """Bar chart and table: exploration accuracy vs deep training accuracy."""
    print("\n" + "=" * 80)
    print("  TABLA 3: Mejora Exploración → Deep Training")
    print("=" * 80)

    rows = []
    for exp in experiments:
        pm = exp['performance_metrics']
        acc_e = pm['best_exploration_accuracy']
        acc_d = pm.get('best_deep_training_accuracy', acc_e)
        delta = acc_d - acc_e
        delta_pct = (delta / acc_e) * 100 if acc_e > 0 else 0
        tc = exp.get('training_config', {})

        rows.append({
            'experiment_id': exp['experiment_id'],
            'dataset': DATASET_DISPLAY.get(exp['dataset']['name'], exp['dataset']['name']),
            'acc_exploration': acc_e,
            'acc_deep': acc_d,
            'delta': delta,
            'delta_pct': delta_pct,
            'epochs_expl': tc.get('exploration_epochs', '?'),
            'epochs_deep': tc.get('deep_training_epochs', '?'),
        })

    # Console
    header = f"{'Experimento':<35} {'Dataset':<16} {'Acc Expl.':<10} {'Acc Deep':<10} {'Δ Abs.':<10} {'Δ %':<8} {'Ep.E':<6} {'Ep.D':<6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['experiment_id']:<35} {r['dataset']:<16} {r['acc_exploration']:<10.4f} {r['acc_deep']:<10.4f} {r['delta']:<10.4f} {r['delta_pct']:<8.2f} {r['epochs_expl']:<6} {r['epochs_deep']:<6}")

    # ── LaTeX ───────────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Mejora del accuracy entre la fase de exploración y deep training}")
    latex.append(r"\label{tab:mejora_fases}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Experimento} & \textbf{Acc. Exploración} & \textbf{Acc. Deep Training} & \textbf{$\Delta$ Abs.} & \textbf{$\Delta$ \%} \\")
    latex.append(r"\midrule")

    prev_ds = None
    for r in rows:
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex.append(r"\midrule")
        prev_ds = r['dataset']
        short_id = r['experiment_id'].split('-', 1)[-1]
        delta_str = f"+{r['delta']:.4f}"
        pct_str = f"+{r['delta_pct']:.2f}\\%"
        latex.append(f"  {ds_display} & {short_id} & {r['acc_exploration']:.4f} & {r['acc_deep']:.4f} & {delta_str} & {pct_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla3_mejora_fases.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"\n  LaTeX → {latex_path}")

    # ── Bar chart ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grouped bars exploration vs deep
    x = np.arange(len(rows))
    width = 0.35
    labels_short = [f"{r['dataset']}\n#{i+1}" for i, r in enumerate(rows)]

    bars1 = ax1.bar(x - width / 2, [r['acc_exploration'] for r in rows], width,
                    label='Exploración', color=COLORS['exploration'], alpha=0.85)
    bars2 = ax1.bar(x + width / 2, [r['acc_deep'] for r in rows], width,
                    label='Deep Training', color=COLORS['deep_training'], alpha=0.85)

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy: Exploración vs Deep Training', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, fontsize=9)
    ax1.legend(fontsize=11)
    ax1.set_ylim(bottom=0.95)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    # Right: delta improvement bar
    deltas = [r['delta'] * 100 for r in rows]  # in percentage points ×100 for readability
    colors_delta = [COLORS['improve']] * len(deltas)
    bars3 = ax2.barh(x, [r['delta_pct'] for r in rows], color=colors_delta, alpha=0.85, height=0.6)
    ax2.set_xlabel('Mejora Relativa (%)', fontsize=12)
    ax2.set_title('Mejora Deep Training vs Exploración', fontsize=13, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels_short, fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    for bar, r in zip(bars3, rows):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'+{r["delta_pct"]:.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig3_exploracion_vs_deep.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura → {fig_path}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 4: Arquitecturas Óptimas Descubiertas por Dataset
# ═════════════════════════════════════════════════════════════════════════════
def table4_architectures(experiments):
    """Detailed architecture breakdown for each experiment's best model."""
    print("\n" + "=" * 80)
    print("  TABLA 4: Arquitecturas Óptimas Descubiertas")
    print("=" * 80)

    rows = []
    for exp in experiments:
        arch = exp.get('architecture', {})
        base = arch.get('base_architecture', '?')
        pm = exp['performance_metrics']

        if base == 'inception':
            n_blocks = arch.get('inception_blocks_n', 0)
            n_modules = arch.get('inception_modules_n', 0)
            stem_filters = arch.get('inception_stem_block_conv_filters', [])
            stem_sizes = arch.get('inception_stem_block_conv_filter_sizes', [])
            # Total filters in inception modules
            conv3x3 = arch.get('inception_modules_conv3x3_filters', [])
            conv5x5 = arch.get('inception_modules_conv5x5_filters', [])
            conv1x1 = arch.get('inception_modules_conv1x1_filters', [])
            pool_conv = arch.get('inception_modules_pooling_conv_filters', [])
            total_filters_per_block = []
            for i in range(len(conv1x1)):
                total_filters_per_block.append(
                    conv1x1[i] + conv3x3[i] + conv5x5[i] + pool_conv[i]
                )

            detail = {
                'type': 'Inception',
                'blocks': f"{n_blocks} bloques × {n_modules} módulos",
                'stem': f"{stem_filters[0] if stem_filters else '?'} filtros ({stem_sizes[0] if stem_sizes else '?'}×{stem_sizes[0] if stem_sizes else '?'})",
                'filters_summary': f"{total_filters_per_block}" if total_filters_per_block else "N/A",
                'max_filters': max(total_filters_per_block) if total_filters_per_block else 0,
                'conv3x3': str(conv3x3),
                'conv5x5': str(conv5x5),
            }
        else:  # CNN
            n_blocks = arch.get('cnn_blocks_n', 0)
            n_layers = arch.get('cnn_blocks_conv_layers_n', 0)
            filters = arch.get('cnn_block_conv_filters', [])
            filter_sizes = arch.get('cnn_block_conv_filter_sizes', [])
            dropouts = arch.get('cnn_block_dropout_values', [])
            pooling = arch.get('cnn_block_max_pooling_sizes', [])

            detail = {
                'type': 'CNN',
                'blocks': f"{n_blocks} bloques × {n_layers} capas conv",
                'stem': "N/A",
                'filters_summary': f"{filters}",
                'max_filters': max(filters) if filters else 0,
                'filter_sizes': str(filter_sizes),
                'dropouts': str(dropouts),
            }

        classifier = arch.get('classifier_layer_type', '?').upper()
        cls_layers = arch.get('classifier_layers_n', 0)
        cls_units = arch.get('classifier_layers_units', [])
        cls_dropouts = arch.get('classifier_dropouts', [])
        cls_detail = f"{classifier}"
        if cls_layers > 0:
            cls_detail += f" ({cls_layers} capas: {cls_units})"
            if any(d > 0 for d in cls_dropouts):
                cls_detail += f" dropout={cls_dropouts}"

        row = {
            'experiment_id': exp['experiment_id'],
            'dataset': DATASET_DISPLAY.get(exp['dataset']['name'], exp['dataset']['name']),
            'acc_deep': pm.get('best_deep_training_accuracy', 0),
            'base_arch': detail['type'],
            'blocks': detail['blocks'],
            'stem': detail['stem'],
            'filters_summary': detail['filters_summary'],
            'max_filters': detail['max_filters'],
            'classifier_detail': cls_detail,
            'full_detail': detail,
        }
        rows.append(row)

        print(f"\n  {row['experiment_id']} ({row['dataset']}) — Acc: {row['acc_deep']:.4f}")
        print(f"    Tipo:         {row['base_arch']}")
        print(f"    Bloques:      {row['blocks']}")
        print(f"    Stem:         {row['stem']}")
        print(f"    Filtros:      {row['filters_summary']}")
        print(f"    Clasificador: {row['classifier_detail']}")

    # ── LaTeX: Tabla simplificada ───────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Arquitecturas óptimas descubiertas por el AutoML para cada dataset}")
    latex.append(r"\label{tab:arquitecturas_optimas}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llclcp{4cm}}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{Acc.} & \textbf{Tipo} & \textbf{Estructura} & \textbf{Máx. Filtros} & \textbf{Clasificador} \\")
    latex.append(r"\midrule")

    prev_ds = None
    for r in rows:
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex.append(r"\midrule")
        prev_ds = r['dataset']
        latex.append(f"  {ds_display} & {r['acc_deep']:.4f} & {r['base_arch']} & {r['blocks']} & {r['max_filters']} & {r['classifier_detail']} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = os.path.join(OUTPUT_DIR, 'tabla4_arquitecturas_optimas.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"\n  LaTeX → {latex_path}")

    # ── LaTeX: Tabla detallada por arquitectura ─────────────────────────────
    latex2 = []
    latex2.append(r"\begin{table}[htbp]")
    latex2.append(r"\centering")
    latex2.append(r"\caption{Detalle de arquitecturas descubiertas: filtros y configuración}")
    latex2.append(r"\label{tab:arquitecturas_detalle}")
    latex2.append(r"\footnotesize")
    latex2.append(r"\begin{tabular}{llp{3cm}p{5cm}}")
    latex2.append(r"\toprule")
    latex2.append(r"\textbf{Dataset} & \textbf{Tipo} & \textbf{Bloques} & \textbf{Configuración de Filtros} \\")
    latex2.append(r"\midrule")

    prev_ds = None
    for r in rows:
        ds_display = r['dataset'] if r['dataset'] != prev_ds else ""
        if r['dataset'] != prev_ds and prev_ds is not None:
            latex2.append(r"\midrule")
        prev_ds = r['dataset']
        d = r['full_detail']
        if r['base_arch'] == 'Inception':
            config = f"Stem: {d['stem']}; Conv3×3: {d['conv3x3']}; Conv5×5: {d['conv5x5']}"
        else:
            config = f"Filtros: {d['filters_summary']}; Tamaños: {d.get('filter_sizes','?')}; Dropout: {d.get('dropouts','?')}"
        # Escape underscores and special chars for LaTeX
        config = config.replace('_', r'\_')
        latex2.append(f"  {ds_display} & {r['base_arch']} & {r['blocks']} & {config} \\\\")

    latex2.append(r"\bottomrule")
    latex2.append(r"\end{tabular}")
    latex2.append(r"\end{table}")

    latex2_path = os.path.join(OUTPUT_DIR, 'tabla4b_arquitecturas_detalle.tex')
    with open(latex2_path, 'w') as f:
        f.write('\n'.join(latex2))
    print(f"  LaTeX → {latex2_path}")

    # ── Figure: Architecture type distribution per dataset ──────────────────
    groups = group_by_dataset(experiments)
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 5))
    if len(groups) == 1:
        axes = [axes]

    for ax, (ds_key, ds_display) in zip(axes, [(k, DATASET_DISPLAY.get(k, k)) for k in ['mnist', 'cifar10', 'fashion_mnist'] if k in groups]):
        exps = groups[ds_key]
        # Count architectures across all leaderboard entries
        arch_counts = defaultdict(int)
        arch_accs = defaultdict(list)
        for exp in exps:
            for entry in exp.get('leaderboard', []):
                base = entry.get('base_architecture', '?')
                phase = entry.get('phase', 'exploration')
                if phase == 'deep_training':
                    arch_counts[base] += 1
                    perf = entry.get('performance_2', entry.get('performance', 0))
                    if perf and perf > 0:
                        arch_accs[base].append(perf)
                else:
                    arch_counts[base] += 1
                    arch_accs[base].append(entry.get('performance', 0))

        # Winners (best model per experiment)
        winner_counts = defaultdict(int)
        for exp in exps:
            base = exp['architecture']['base_architecture']
            winner_counts[base] += 1

        labels_arch = list(arch_counts.keys())
        counts = [arch_counts[a] for a in labels_arch]
        colors = [COLORS.get(a, '#95a5a6') for a in labels_arch]

        bars = ax.bar(labels_arch, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        # Annotate with winner count
        for i, a in enumerate(labels_arch):
            w = winner_counts.get(a, 0)
            if w > 0:
                ax.text(i, counts[i] + 0.2, f'★{w} best', ha='center', fontsize=9, fontweight='bold', color='#e74c3c')

        ax.set_title(f'{ds_display}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Modelos evaluados' if ax == axes[0] else '')
        ax.set_xlabel('Tipo de arquitectura')

    fig.suptitle('Distribución de Arquitecturas Exploradas y Ganadoras\n(por dataset, todas las ejecuciones)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig4_arquitecturas_distribucion.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura → {fig_path}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  GENERADOR DE TABLAS PARA TESIS")
    print("  Implementación de un Modelo de AutoML en la Nube")
    print("  Experimentos con 2 GPUs (NVIDIA RTX 3080)")
    print("=" * 80)

    experiments = load_experiments()

    if not experiments:
        print("ERROR: No se encontraron archivos JSON de experimentos.")
        sys.exit(1)

    # Sort by dataset then by experiment_id for consistent ordering
    experiments.sort(key=lambda e: (e['dataset']['name'], e['experiment_id']))

    # Generate all tables
    t1 = table1_best_accuracy(experiments)
    t2 = table2_reproducibility(experiments)
    t3 = table3_exploration_vs_deep(experiments)
    t4 = table4_architectures(experiments)

    print("\n" + "=" * 80)
    print(f"  COMPLETADO — Archivos generados en: {OUTPUT_DIR}")
    print("=" * 80)

    # List generated files
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:<50} {size:>6} bytes")


if __name__ == '__main__':
    main()
