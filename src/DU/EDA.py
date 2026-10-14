import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
print("========== Carga de Datos ==========")
data = pd.read_csv('../../Data/dataset_vacas_anonimo.csv')
print(data.shape)

# Revisión de la calidad de los datos
print("========== Info del DataSet =========")
print("Dimensiones:", data.shape)
print("Filas:", len(data))
print("Columnas:", len(data.columns))

print("Duplicados: ", data.duplicated().sum())
data = data.drop_duplicates()
print("Sin duplicados: ", data.shape)

print("========== Calidad de los datos ==========")
calidad = pd.DataFrame({
    "tipo": data.dtypes.astype(str),
    "no_nulos": data.notna().sum(),
    "faltantes": data.isna().sum(),
    "porcentaje_faltantes": (
        data.isna().mean() * 100
    ).round(2),
    "valores_unicos": data.nunique()
})

print(calidad.to_string())

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

def convertir_segundos(valor):
    try:
        minutos, segundos = str(valor).split(":")
        return int(minutos) * 60 + float(segundos)
    except (ValueError, AttributeError):
        return np.nan

data["duracion_segundos"] = (
    data["main_duracion_mm_ss"].apply(convertir_segundos)
)

sns.histplot(
    data=data,
    x="main_produccion_kg",
    kde=True,
    ax=axes[0]
)
axes[0].set_title("Distribución de producción")
axes[0].set_xlabel("Producción (kg)")

sns.boxplot(
    data=data,
    x="main_produccion_kg",
    ax=axes[1]
)
axes[1].set_title("Valores extremos de producción")

plt.tight_layout()
plt.show()

sns.boxplot(
    data=data,
    x="estado_incompleto",
    y="main_produccion_kg"
)

plt.title("Producción según ordeño incompleto")
plt.xlabel("Ordeño incompleto")
plt.ylabel("Producción (kg)")
plt.show()

cols_corr = [
    "main_produccion_kg",
    "main_duracion_mm_ss",
    "estado_numero_de_ordeno",
    "media_de_los_flujos_kg_min_di",
    "media_de_los_flujos_kg_min_dd",
    "media_de_los_flujos_kg_min_ti",
    "media_de_los_flujos_kg_min_td"
]

corr = data[cols_corr].corr(numeric_only=True)

plt.figure(figsize=(11, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlación entre variables del ordeño")
plt.tight_layout()
plt.show()

cols_cuartos = [
    "producciones_kg_di",
    "producciones_kg_dd",
    "producciones_kg_ti",
    "producciones_kg_td"
]

cuartos = data[cols_cuartos].melt(
    var_name="cuarto",
    value_name="produccion_kg"
)

sns.boxplot(
    data=cuartos,
    x="cuarto",
    y="produccion_kg"
)

plt.title("Producción por cuarto de la ubre")
plt.xlabel("Cuarto")
plt.ylabel("Producción (kg)")
plt.xticks(rotation=20)
plt.show()

sns.scatterplot(
    data=data,
    x="duracion_segundos",
    y="main_produccion_kg",
    alpha=0.4
)

plt.title("Duración del ordeño frente a producción")
plt.xlabel("Duración (segundos)")
plt.ylabel("Producción (kg)")
plt.show()