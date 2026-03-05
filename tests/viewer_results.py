import os
import json
import glob
import re
from collections import defaultdict
import statistics

def parse_time(time_str):
    """Parse time string and return seconds"""
    # Extract HH:MM:SS format
    hms_match = re.search(r'(\d{2}):(\d{2}):(\d{2})', time_str)
    if hms_match:
        hours, minutes, seconds = map(int, hms_match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Directorio de resultados
results_dir = "/home/p0wden/Documents/mloptimizer/results/"

# Diccionarios para almacenar datos
dataset_performances = defaultdict(list)
dataset_times = defaultdict(list)

# Leer todos los archivos en el directorio
for filepath in glob.glob(os.path.join(results_dir, "*")):
    if os.path.isfile(filepath):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Buscar líneas con JSON válido
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            # Remove trailing quotes that may exist in some files
                            line = line.rstrip('"').rstrip()
                            data = json.loads(line)
                            
                            # Extraer información
                            if 'model_training_request' in data and 'performance' in data:
                                dataset = data['model_training_request'].get('dataset_tag', 'unknown')
                                perf = data.get('performance', 0)
                                perf2 = data.get('performance_2', 0)
                                
                                # Filtrar performances válidos (> 0)
                                if perf > 0:
                                    dataset_performances[dataset].append(perf)
                                if perf2 > 0:
                                    dataset_performances[dataset].append(perf2)
                        except json.JSONDecodeError:
                            continue
                    
                    # Parsear tiempo de optimización
                    elif 'Optimization took:' in line:
                        # Extraer dataset del nombre del archivo
                        dataset = filename.split('-')[0] if '-' in filename else 'unknown'
                        
                        # Extraer tiempo en formato HH:MM:SS
                        time_seconds = parse_time(line)
                        if time_seconds is not None:
                            dataset_times[dataset].append(time_seconds)
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")

# Calcular estadísticas por dataset
print("=" * 80)
print("RESUMEN DE PERFORMANCES POR DATASET")
print("=" * 80)
print()

results_summary = []

for dataset, performances in sorted(dataset_performances.items()):
    if len(performances) > 0:
        # Remover outliers usando método IQR
        sorted_perf = sorted(performances)
        q1 = statistics.quantiles(sorted_perf, n=4)[0] if len(sorted_perf) >= 4 else min(sorted_perf)
        q3 = statistics.quantiles(sorted_perf, n=4)[2] if len(sorted_perf) >= 4 else max(sorted_perf)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filtrar datos sin outliers
        clean_performances = [p for p in performances if lower_bound <= p <= upper_bound]
        
        # Obtener tiempos y filtrar outliers también
        times = dataset_times.get(dataset, [])
        clean_times = []
        if len(times) >= 4:
            sorted_times = sorted(times)
            t_q1 = statistics.quantiles(sorted_times, n=4)[0]
            t_q3 = statistics.quantiles(sorted_times, n=4)[2]
            t_iqr = t_q3 - t_q1
            t_lower = t_q1 - 1.5 * t_iqr
            t_upper = t_q3 + 1.5 * t_iqr
            clean_times = [t for t in times if t_lower <= t <= t_upper]
        elif len(times) > 0:
            clean_times = times
        
        if len(clean_performances) > 0:
            mean_perf = statistics.mean(clean_performances)
            median_perf = statistics.median(clean_performances)
            std_perf = statistics.stdev(clean_performances) if len(clean_performances) > 1 else 0
            min_perf = min(clean_performances)
            max_perf = max(clean_performances)
            
            # Calcular estadísticas de tiempo
            mean_time = statistics.mean(clean_times) if len(clean_times) > 0 else None
            median_time = statistics.median(clean_times) if len(clean_times) > 0 else None
            
            results_summary.append({
                'dataset': dataset,
                'count': len(clean_performances),
                'mean': mean_perf,
                'median': median_perf,
                'std': std_perf,
                'min': min_perf,
                'max': max_perf,
                'outliers_removed': len(performances) - len(clean_performances),
                'mean_time': mean_time,
                'median_time': median_time,
                'time_samples': len(clean_times)
            })
            
            print(f"Dataset: {dataset}")
            print(f"  Samples (limpios):  {len(clean_performances)}/{len(performances)}")
            print(f"  Media:              {mean_perf:.4f} ({mean_perf*100:.2f}%)")
            print(f"  Mediana:            {median_perf:.4f} ({median_perf*100:.2f}%)")
            print(f"  Desv. Estándar:     {std_perf:.4f}")
            print(f"  Rango:              [{min_perf:.4f}, {max_perf:.4f}]")
            print(f"  Outliers removidos: {len(performances) - len(clean_performances)}")
            if mean_time is not None:
                print(f"  ⏱️  Tiempo medio:      {seconds_to_hms(mean_time)} ({mean_time/60:.1f} min)")
                print(f"  Tiempo mediano:     {seconds_to_hms(median_time)} ({median_time/60:.1f} min)")
                print(f"  Muestras tiempo:    {len(clean_times)}/{len(times)}")
            print()

print("=" * 80)
print("TABLA RESUMEN (formato para copiar)")
print("=" * 80)
print()
print("| Dataset          | Samples | Mean Acc | Median Acc | Std Dev | Min    | Max    | Avg Time  |")
print("|------------------|---------|----------|------------|---------|--------|--------|-----------|")
for r in results_summary:
    time_str = seconds_to_hms(r['mean_time']) if r['mean_time'] else "N/A"
    print(f"| {r['dataset']:<16} | {r['count']:>7} | {r['mean']*100:>7.2f}% | {r['median']*100:>9.2f}% | {r['std']:>7.4f} | {r['min']:.4f} | {r['max']:.4f} | {time_str:>9} |")

print()
print("=" * 80)
print("TABLA CSV (para Excel/Google Sheets)")
print("=" * 80)
print("Dataset,Samples,Mean_Accuracy_%,Median_Accuracy_%,Std_Dev,Min,Max,Avg_Time_HH:MM:SS,Avg_Time_Minutes")
for r in results_summary:
    time_hms = seconds_to_hms(r['mean_time']) if r['mean_time'] else "N/A"
    time_min = f"{r['mean_time']/60:.1f}" if r['mean_time'] else "N/A"
    print(f"{r['dataset']},{r['count']},{r['mean']*100:.2f},{r['median']*100:.2f},{r['std']:.4f},{r['min']:.4f},{r['max']:.4f},{time_hms},{time_min}")