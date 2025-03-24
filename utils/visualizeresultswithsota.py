import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definir el directorio que contiene los resultados
results_dir = "~/Documents/IA/MLOptimizer/results"

# Lista de archivos a procesar
files = [
    "fashion_mnist-20241025-185550",
    "fashion_mnist-20241101-142830",
    "fashion_mnist-20241101-205845",
    "fashion_mnist-20241102-181505",
    "mnist-20241019-125514",
    "mnist-20250206-142721"
]

# Inicializar un diccionario para almacenar los resultados
results = {"fashion_mnist": [], "mnist": []}

# Leer cada archivo y extraer las métricas de rendimiento
for file in files:
    file_path = os.path.expanduser(os.path.join(results_dir, file))
    with open(file_path, 'r') as f:
        content = f.read()
        # Encontrar el inicio y el final del objeto JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        # Extraer la parte JSON
        json_data = content[json_start:json_end]
        try:
            data = json.loads(json_data)
            performance = data.get("performance", None)
            if "fashion_mnist" in file:
                results["fashion_mnist"].append(performance)
            elif "mnist" in file:
                results["mnist"].append(performance)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in file {file}: {e}")

# Convertir el diccionario a un DataFrame
df = pd.DataFrame({
    "Dataset": ["Fashion MNIST"] * len(results["fashion_mnist"]) + ["MNIST"] * len(results["mnist"]),
    "Performance": results["fashion_mnist"] + results["mnist"]
})

# Valores del SOTA para Fashion MNIST y MNIST con etiquetas y colores
sota_fashion_mnist = [
    {"value": 0.9906, "label": "pFedBreD_ns_mg", "color": "red"},
    {"value": 0.9503, "label": "LR-Net", "color": "red"},
    {"value": 0.9444, "label": "Inception v3", "color": "red"},
    {"value": 0.9345, "label": "R-ExplaiNet-26", "color": "red"},
    {"value": 0.9330, "label": "ResNet-18 + VEA", "color": "red"},
    {"value": 0.9300, "label": "Tsetlin Comp.", "color": "red"},
    {"value": 0.9280, "label": "StiDi-BP R-CSNN", "color": "red"},
    {"value": 0.9230, "label": "Star Alg. LeNet", "color": "red"},
    {"value": 0.9228, "label": "ResNet-18", "color": "red"}
]

sota_mnist = [
    {"value": 0.9987, "label": "Branch/Merge CNN", "color": "black"},
    {"value": 0.9984, "label": "EnsNet", "color": "black"},
    {"value": 0.9984, "label": "Efficient-CapsNet", "color": "black"},
    {"value": 0.9983, "label": "SOPCNN", "color": "black"},
    {"value": 0.9982, "label": "RMDL", "color": "black"},
    {"value": 0.9980, "label": "R-ExplaiNet-22", "color": "black"},
    {"value": 0.9977, "label": "DropConnect", "color": "black"},
    {"value": 0.9975, "label": "µ2Net", "color": "black"},
    {"value": 0.9972, "label": "VGG-5 Spinal", "color": "black"},
    {"value": 0.9971, "label": "TextCaps", "color": "black"}
]

# Configurar el estilo de la gráfica
sns.set(style="whitegrid")

# Crear una gráfica de caja para comparar el rendimiento
plt.figure(figsize=(14, 8))
sns.boxplot(x="Dataset", y="Performance", data=df)

# Añadir líneas horizontales para los valores del SOTA de Fashion MNIST
for sota in sota_fashion_mnist:
    plt.axhline(y=sota["value"], color=sota["color"], linestyle='--', alpha=0.7,
                label=f'Fashion MNIST: {sota["label"]} ({sota["value"]*100:.2f}%)')

# Añadir líneas horizontales para los valores del SOTA de MNIST
for sota in sota_mnist:
    plt.axhline(y=sota["value"], color=sota["color"], linestyle='--', alpha=0.7,
                label=f'MNIST: {sota["label"]} ({sota["value"]*100:.2f}%)')

# Añadir título y etiquetas
plt.title("Comparación de Resultados con Múltiples Valores del Estado del Arte (SOTA)")
plt.ylabel("Rendimiento (Precisión)")
plt.xlabel("Dataset")

# Mostrar la leyenda fuera del gráfico
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Ajustar el layout para que la leyenda no se solape
plt.tight_layout()
plt.show()