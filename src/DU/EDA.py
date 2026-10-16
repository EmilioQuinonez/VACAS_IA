import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

# Carga de datos
print("========== Carga de Datos ==========")
data = pd.read_csv('../../Data/dataset_vacas_anonimo.csv')
print("datos de vacas: ",data.shape)
clima = pd.read_csv('../../Data/clima/clima_nasa_power_por_hora.csv')
print("datos de clima: ",clima.shape)

# Unión de las columnas de clima y de datos
vacas = data.copy()
clima_limpio = clima.copy()

fechas = (
    vacas["main_hora_de_inicio"]
    .astype("string")
    .str.strip()
    .str.replace(r"\ba\.\s*m\.", "AM", regex=True, case=False)
    .str.replace(r"\bp\.\s*m\.", "PM", regex=True, case=False)
)

vacas["_fecha_hora_clima"] = pd.to_datetime(
    fechas,
    format="%d/%m/%Y %I:%M %p",
    errors="coerce"
).dt.floor("h")

clima_limpio["_fecha_hora_clima"] = pd.to_datetime(
    clima_limpio["hora"],
    errors="coerce",
    dayfirst=True
).dt.floor("h")

columnas_clima = [
    "t2m",
    "rh2m",
    "t2mdew",
    "t2mwet",
    "qv2m",
    "ps",
    "ws2m",
    "ws10m",
    "wd10m",
    "prectotcorr",
    "allsky_sfc_sw_dwn",
    "allsky_sfc_lw_dwn",
    "cloud_amt"
]

faltantes = [
    col for col in ["hora"] + columnas_clima
    if col not in clima_limpio.columns
]

if faltantes:
    raise ValueError(f"Faltan columnas en el dataset climático: {faltantes}")

clima_limpio = clima_limpio[
    ["_fecha_hora_clima"] + columnas_clima
].copy()

if clima_limpio["_fecha_hora_clima"].duplicated().any():
    raise ValueError(
        "Hay horas duplicadas en el dataset NASA. "
        "Revisa los registros antes de unirlos."
    )

vacas = vacas.merge(
    clima_limpio,
    on="_fecha_hora_clima",
    how="left",
    validate="many_to_one"
)

vacas.drop(columns="_fecha_hora_clima", inplace=True)

print("Dimensiones originales:", data.shape)
print("Dimensiones finales:", vacas.shape)

print("\nRegistros sin temperatura asociada:", vacas["t2m"].isna().sum())
print("Total de registros:", len(vacas))

data = vacas

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
#plt.show()

cols_corr = [
    "main_produccion_kg",
    "main_duracion_mm_ss",
    "estado_numero_de_ordeno",
    "media_de_los_flujos_kg_min_di",
    "media_de_los_flujos_kg_min_dd",
    "media_de_los_flujos_kg_min_ti",
    "media_de_los_flujos_kg_min_td",
    "t2m",
    "rh2m",
    "t2mdew",
    "t2mwet",
    "qv2m",
    "ps",
    "ws2m",
    "ws10m",
    "wd10m",
    "prectotcorr",
    "allsky_sfc_sw_dwn",
    "allsky_sfc_lw_dwn",
    "cloud_amt"
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


datos_numericos = vacas.select_dtypes(include="number").copy()

corr = datos_numericos.corr(method="pearson")

plt.figure(figsize=(24, 20))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=False
)

plt.title("Matriz de correlación de todas las variables numéricas")
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

'''
    Columnas que valen a pena eliminarse por tener poca relación con el resto de datos
    o al tener muchos datos nuelos al revisar su calidad.

    estado_rcs_1000_celulas_ml, misc_usuario -> Columnas vacías
    ps, ws2m, allsky_sfc_sw_dwn -> Relación casi de 0 con todos los datos de las vacas

    La producción tiene varios dato anómalos como se puede ver en la segunda gráfica.
    Se debe hacer un OLS para poder identificar como se relacionan los valores anómalos
    con lo que buscamos predecir.

    Vale la pena indagar con el data set sobre como se relaciona el clima con patadas

    Lo siguiente es una propuesta de búsqueda de correlación entre fallas en el ordeño
    y variables del clima para evitar descartar columnas que si tengan relación.
    Es una propuesta por lo que se hace con una copia de los datos, es una propuesta
    por lo que no esta muy formalizado.
'''

analisis = data.copy()

estados = [
    "estado_patada",
    "estado_incompleto",
    "estado_pezones_no_encontrados"
]

clima_cols = [
    "t2m", "rh2m", "t2mdew", "t2mwet", "qv2m", "ps",
    "ws2m", "ws10m", "wd10m", "prectotcorr",
    "allsky_sfc_sw_dwn", "allsky_sfc_lw_dwn", "cloud_amt"
]

# Revisar qué valores aparecen en cada variable de estado
for col in estados:
    print("=/=/=/=/=/=")
    print(f"\nValores más frecuentes en {col}:")
    print(analisis[col].value_counts(dropna=False).head(20).to_string())

# Crear indicadores de presencia del evento
for col in estados:
    analisis[col + "_presente"] = (
        analisis[col].notna()
        & analisis[col].astype("string").str.strip().ne("")
    ).astype(int)

# Correlación entre la presencia del evento y el clima
resultados = []

for estado in estados:
    indicador = estado + "_presente"

    for variable_clima in clima_cols:
        pares = analisis[[indicador, variable_clima]].dropna()

        # La correlación requiere que existan ambas categorías
        if pares[indicador].nunique() < 2:
            continue

        r, p = pointbiserialr(
            pares[indicador],
            pares[variable_clima]
        )

        resultados.append({
            "estado": estado,
            "variable_climatica": variable_clima,
            "correlacion": r,
            "p_valor": p,
            "observaciones": len(pares)
        })

resultados_corr = pd.DataFrame(resultados)
resultados_corr["correlacion_abs"] = (
    resultados_corr["correlacion"].abs()
)

resultados_corr = resultados_corr.sort_values(
    "correlacion_abs", ascending=False
)

matriz_corr = resultados_corr.pivot(
    index="estado",
    columns="variable_climatica",
    values="correlacion"
)

plt.figure(figsize=(15, 4))

sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-0.10,
    vmax=0.10,
    linewidths=0.5
)

plt.title("Correlación entre eventos de ordeño y variables climáticas")
plt.xlabel("Variables climáticas")
plt.ylabel("Eventos de ordeño")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()