import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

raiz = Path(__file__).resolve().parent.parent.parent
tablas_defecto = raiz.parent / "analisis_descriptivo_datos_vacas" / "tablas"
clima_defecto = raiz / "Data" / "clima" / "clima_nasa_power_por_hora.csv"
resumen_defecto = raiz / "Data" / "eda" / "resumen_exploracion.json"
salida_defecto = raiz / "docs" / "DU" / "figuras"

CUARTOS = ["di", "dd", "ti", "td"]
EVENTOS = ["kick", "incomplete", "teat_missing"]
ETIQUETAS_EVENTO = {"kick": "Patada", "incomplete": "Ordeño incompleto", "teat_missing": "Pezón no encontrado"}
# NASA POWER entrega promedios horarios; correrlo una hora lo alinea con ERA5-Land (ver clima_comparacion.py)
DESFASE_CLIMA_HORAS = 1
UMBRAL_THI = 68
UMBRAL_BRECHA_DIAS = 45

COLOR_BASE = "#2a7a8c"
COLOR_ALERTA = "#b5473a"
COLOR_GRIS = "#c9d6d9"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.dpi": 100})


def guardar(fig, destino, nombre):
    """
        Guarda una figura en la carpeta de salida y la cierra.

        Args:
            fig (Figure): La figura a guardar.
            destino (Path): Carpeta donde se guarda la imagen.
            nombre (str): Nombre del archivo sin extension.
    """
    fig.tight_layout()
    fig.savefig(destino / f"{nombre}.png", dpi=200)
    plt.close(fig)


def cargar_leche(tablas):
    """
        Lee la tabla de eventos de ordeño, agrega variables derivadas y reemplaza el ID real
        por un alias (V001, V002, ...) para que ningun resultado muestre identificadores originales.

        Args:
            tablas (Path): Carpeta con leche_eventos_normalizados.csv.gz.

        Returns:
            DataFrame: Una fila por sesion de ordeño.
    """
    leche = pd.read_csv(tablas / "leche_eventos_normalizados.csv.gz", parse_dates=["timestamp"],
                        dtype={"animal_id": str}).drop(columns=["source_file", "source_row", "source_kind"])

    orden = leche.groupby("animal_id")["timestamp"].min().sort_values().index
    alias = {original: f"V{i:03d}" for i, original in enumerate(orden, start=1)}
    leche["animal_id"] = leche["animal_id"].map(alias)

    leche["fecha"] = leche["timestamp"].dt.normalize()
    leche["hora"] = leche["timestamp"].dt.hour
    leche["dia_semana"] = leche["timestamp"].dt.dayofweek
    leche["mes"] = leche["timestamp"].dt.to_period("M").dt.to_timestamp()
    leche["evento_alguno"] = leche[EVENTOS].max(axis=1)

    # Un 0 en conductividad o flujo no es una medicion real: es un cuarto que no dio leche
    for variable in ["cond", "flow"]:
        columnas = [f"{variable}_{c}" for c in CUARTOS]
        leche[f"{variable}_media"] = leche[columnas].where(leche[columnas] > 0).mean(axis=1)
    return leche


def perfilar_calidad(leche):
    """
        Calcula indicadores de calidad de la tabla: nulos, valores extremos, duplicados y brechas.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.

        Returns:
            dict: Indicadores de calidad.
    """
    numericas = [c for c in leche.columns if c.startswith(("q_", "flow_", "blood_", "cond_"))] + \
                ["milk_kg", "duration_min", "interval_h", "mdi", "rcs", "session_no"]
    nulos = leche[numericas].isna().mean().mul(100).round(2)

    ordenado = leche.sort_values(["animal_id", "timestamp"])
    brechas = ordenado.groupby("animal_id")["timestamp"].diff().dt.days
    suma_cuartos = leche[[f"q_{c}" for c in CUARTOS]].sum(axis=1)

    return {
        "sesiones": len(leche),
        "vacas": int(leche["animal_id"].nunique()),
        "inicio": str(leche["timestamp"].min()),
        "fin": str(leche["timestamp"].max()),
        "nulos_pct": nulos[nulos > 0].sort_values(ascending=False).to_dict(),
        "duplicados_vaca_hora": int(leche.duplicated(["animal_id", "timestamp"]).sum()),
        "milk_kg_menor_igual_0": int((leche["milk_kg"] <= 0).sum()),
        "milk_kg_mayor_40": int((leche["milk_kg"] > 40).sum()),
        "duration_min_mayor_30": int((leche["duration_min"] > 30).sum()),
        "ceros_conductividad_pct": {c: round(float((leche[f"cond_{c}"] == 0).mean() * 100), 1) for c in CUARTOS},
        "ceros_produccion_cuarto_pct": {c: round(float((leche[f"q_{c}"] == 0).mean() * 100), 1) for c in CUARTOS},
        "suma_cuartos_difiere_mas_0_2kg": int(((suma_cuartos - leche["milk_kg"]).abs() > 0.2).sum()),
        "brechas_45_dias_o_mas": int((brechas >= UMBRAL_BRECHA_DIAS).sum()),
        "vacas_con_brecha": int(ordenado.loc[brechas >= UMBRAL_BRECHA_DIAS, "animal_id"].nunique()),
        "sesiones_por_vaca": leche.groupby("animal_id").size().describe().round(0).to_dict(),
        "sesiones_por_vms": leche["machine"].value_counts().to_dict(),
        "destino_leche": leche["destination"].value_counts().to_dict(),
    }


def graficar_distribuciones(leche, destino):
    """
        Grafica los histogramas de las variables numericas principales de cada sesion.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.
    """
    variables = [
        ("milk_kg", "Producción por sesión (kg)", (0, 45)),
        ("duration_min", "Duración de la sesión (min)", (0, 30)),
        ("interval_h", "Intervalo entre ordeños (h)", (0, 24)),
        ("flow_media", "Flujo medio por cuarto (kg/min)", (0, 3)),
        ("cond_media", "Conductividad media (mS/cm)", (3, 8)),
        ("mdi", "MDI", (0.5, 2.5)),
    ]
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    for eje, (variable, etiqueta, limites) in zip(ejes.flat, variables):
        serie = leche[variable].dropna()
        eje.hist(serie[serie.between(*limites)], bins=60, range=limites, color=COLOR_BASE)
        eje.axvline(serie.median(), color=COLOR_ALERTA, linestyle="--", linewidth=1.3)
        eje.set_xlabel(etiqueta)
        eje.set_title(f"mediana {serie.median():.2f}", fontsize=10)
        eje.set_ylabel("Sesiones")
    guardar(fig, destino, "eda_distribuciones")


def graficar_temporal(leche, destino):
    """
        Grafica la distribucion de las sesiones por hora y dia de la semana, y la evolucion mensual
        de la producción por vaca-dia y de las vacas activas.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

    Returns:
        dict: Indicadores de los patrones temporales.
    """
    por_hora = leche["hora"].value_counts(normalize=True).sort_index() * 100
    por_dia = leche["dia_semana"].value_counts(normalize=True).sort_index() * 100
    vaca_dia = leche.groupby(["animal_id", "fecha"]).agg(leche_kg=("milk_kg", "sum"), sesiones=("milk_kg", "size"))
    mensual = vaca_dia.reset_index().assign(mes=lambda d: d["fecha"].dt.to_period("M").dt.to_timestamp())
    mensual = mensual.groupby("mes").agg(leche_vaca_dia=("leche_kg", "mean"), vacas=("animal_id", "nunique"))
    # Se descartan el primer y el ultimo mes por estar incompletos
    mensual = mensual.iloc[1:-1]

    fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5))
    ejes[0, 0].bar(por_hora.index, por_hora.values, color=COLOR_BASE)
    ejes[0, 0].set_xlabel("Hora local de inicio")
    ejes[0, 0].set_ylabel("% de las sesiones")
    ejes[0, 0].set_title("Sesiones por hora del día")

    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    ejes[0, 1].bar(dias, por_dia.values, color=COLOR_BASE)
    ejes[0, 1].set_ylabel("% de las sesiones")
    ejes[0, 1].set_title("Sesiones por día de la semana")

    ejes[1, 0].plot(mensual.index, mensual["leche_vaca_dia"], color=COLOR_BASE, linewidth=2)
    ejes[1, 0].set_ylabel("Leche por vaca-día (kg)")
    ejes[1, 0].set_title("Producción por vaca-día, por mes")

    ejes[1, 1].plot(mensual.index, mensual["vacas"], color=COLOR_ALERTA, linewidth=2)
    ejes[1, 1].set_ylabel("Vacas con al menos una sesión")
    ejes[1, 1].set_title("Vacas activas por mes")
    for eje in ejes[1]:
        eje.tick_params(axis="x", labelrotation=30)
    guardar(fig, destino, "eda_patrones_temporales")

    return {
        "hora_pico": int(por_hora.idxmax()), "pct_hora_pico": round(float(por_hora.max()), 1),
        "pct_sesiones_00_05h": round(float(por_hora.loc[0:5].sum()), 1),
        "pct_por_dia_semana": {d: round(float(v), 1) for d, v in zip(dias, por_dia.values)},
        "sesiones_por_vaca_dia": vaca_dia["sesiones"].describe().round(2).to_dict(),
        "leche_vaca_dia_mediana_kg": round(float(vaca_dia["leche_kg"].median()), 1),
        "leche_vaca_dia_mensual_min_max": [round(float(mensual["leche_vaca_dia"].min()), 1),
                                           round(float(mensual["leche_vaca_dia"].max()), 1)],
        "mes_min_leche": str(mensual["leche_vaca_dia"].idxmin().date()),
        "mes_max_leche": str(mensual["leche_vaca_dia"].idxmax().date()),
        "vacas_activas_primer_ultimo_mes": [int(mensual["vacas"].iloc[0]), int(mensual["vacas"].iloc[-1])],
    }


def graficar_lactancia(leche, destino):
    """
        Grafica una curva de lactancia aproximada: producción diaria segun los dias transcurridos
        desde que la vaca vuelve a ordeñarse tras una brecha larga (probable parto).

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Indicadores de la curva.
    """
    vaca_dia = leche.groupby(["animal_id", "fecha"])["milk_kg"].sum().reset_index().sort_values(["animal_id", "fecha"])
    brecha = vaca_dia.groupby("animal_id")["fecha"].diff().dt.days
    # Solo se cuentan inicios con una brecha previa observada: el primer registro puede no ser un parto
    vaca_dia["episodio"] = (brecha >= UMBRAL_BRECHA_DIAS).groupby(vaca_dia["animal_id"]).cumsum()
    inicio = vaca_dia[brecha >= UMBRAL_BRECHA_DIAS].set_index(["animal_id", "episodio"])["fecha"]
    vaca_dia = vaca_dia[vaca_dia["episodio"] > 0].copy()
    vaca_dia["dim"] = (vaca_dia["fecha"] - vaca_dia.set_index(["animal_id", "episodio"]).index.map(inicio)).dt.days
    vaca_dia = vaca_dia[vaca_dia["dim"].between(0, 305)]
    curva = vaca_dia.groupby("dim")["milk_kg"].quantile([0.25, 0.5, 0.75]).unstack()
    suave = curva.rolling(7, center=True, min_periods=3).mean()

    fig, eje = plt.subplots(figsize=(9, 4.5))
    eje.fill_between(suave.index, suave[0.25], suave[0.75], color=COLOR_GRIS, label="Rango intercuartil")
    eje.plot(suave.index, suave[0.5], color=COLOR_BASE, linewidth=2.2, label="Mediana")
    eje.set_xlabel("Días desde el regreso al ordeño tras una brecha de 45 días o más")
    eje.set_ylabel("Leche por vaca-día (kg)")
    eje.set_title("Curva de lactancia aproximada")
    eje.legend(frameon=False)
    guardar(fig, destino, "eda_lactancia")

    pico = suave[0.5].idxmax()
    return {"episodios": int(vaca_dia.groupby(["animal_id", "episodio"]).ngroups), "dia_pico_mediana": int(pico),
            "leche_pico_kg": round(float(suave[0.5].max()), 1),
            "leche_dia_250_kg": round(float(suave[0.5].loc[245:255].mean()), 1)}


def graficar_cuartos_vms(leche, destino):
    """
        Compara la producción por cuarto y la tasa de eventos por VMS, con pruebas chi-cuadrada.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Tasas por VMS y resultados de las pruebas.
    """
    produccion = {c.upper(): leche[f"q_{c}"].where(leche[f"q_{c}"] > 0).mean() for c in CUARTOS}
    sin_leche = {c.upper(): (leche[f"q_{c}"] == 0).mean() * 100 for c in CUARTOS}
    tasas = leche.groupby("machine")[EVENTOS].mean().mul(100)
    sesiones = leche["machine"].value_counts().sort_index()

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    ejes[0].bar(produccion.keys(), produccion.values(), color=COLOR_BASE)
    ejes[0].set_ylabel("Leche promedio por sesión (kg, sin ceros)")
    ejes[0].set_title("Producción por cuarto")

    ejes[1].bar(sin_leche.keys(), sin_leche.values(), color=COLOR_ALERTA)
    ejes[1].set_ylabel("% de sesiones con 0 kg en el cuarto")
    ejes[1].set_title("Cuartos sin leche")

    ancho = 0.25
    posiciones = np.arange(len(tasas))
    for i, evento in enumerate(EVENTOS):
        ejes[2].bar(posiciones + (i - 1) * ancho, tasas[evento], ancho, label=ETIQUETAS_EVENTO[evento],
                    color=[COLOR_BASE, COLOR_ALERTA, "#e0a458"][i])
    ejes[2].set_xticks(posiciones)
    ejes[2].set_xticklabels([f"{m}\n(n={sesiones[m]:,})" for m in tasas.index])
    ejes[2].set_ylabel("% de las sesiones")
    ejes[2].set_title("Eventos por VMS")
    ejes[2].legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_cuartos_y_vms")

    pruebas = {}
    for evento in EVENTOS:
        tabla = pd.crosstab(leche["machine"], leche[evento])
        chi2, p, _, _ = stats.chi2_contingency(tabla)
        pruebas[evento] = {"chi2": round(float(chi2), 1), "p": float(p),
                           "cramer_v": round(float(np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))), 3)}
    return {"produccion_cuarto_kg": {k: round(float(v), 2) for k, v in produccion.items()},
            "pct_cuarto_sin_leche": {k: round(float(v), 1) for k, v in sin_leche.items()},
            "tasa_eventos_por_vms_pct": tasas.round(2).to_dict(orient="index"), "chi_cuadrada_vms": pruebas}


def graficar_eventos(leche, destino):
    """
        Grafica la prevalencia de los eventos de ordeño, su coocurrencia y su concentracion por vaca.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Prevalencias, coocurrencias y concentracion por vaca.
    """
    prevalencia = leche[EVENTOS + ["evento_alguno"]].mean().mul(100)
    condicional = pd.DataFrame({b: {a: leche.loc[leche[a] == 1, b].mean() * 100 for a in EVENTOS} for b in EVENTOS})

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6))
    nombres = [ETIQUETAS_EVENTO.get(e, "Alguno de los tres") for e in prevalencia.index]
    ejes[0].barh(nombres, prevalencia.values, color=COLOR_BASE)
    for y, valor in enumerate(prevalencia.values):
        ejes[0].text(valor + 0.5, y, f"{valor:.1f} %", va="center", fontsize=9)
    ejes[0].set_xlim(0, prevalencia.max() * 1.25)
    ejes[0].set_xlabel("% de las sesiones")
    ejes[0].set_title("Prevalencia de eventos")

    imagen = ejes[1].imshow(condicional.values, cmap="viridis", vmin=0, vmax=100)
    ejes[1].set_xticks(range(3))
    ejes[1].set_xticklabels([ETIQUETAS_EVENTO[e] for e in EVENTOS], rotation=20, ha="right")
    ejes[1].set_yticks(range(3))
    ejes[1].set_yticklabels([f"Si hay {ETIQUETAS_EVENTO[e].lower()}" for e in EVENTOS])
    for i in range(3):
        for j in range(3):
            ejes[1].text(j, i, f"{condicional.values[i, j]:.0f}", ha="center", va="center",
                         color="white" if condicional.values[i, j] < 60 else "black")
    ejes[1].set_title("% que también presenta cada evento")
    ejes[1].spines[["top", "right", "left", "bottom"]].set_visible(False)

    concentracion = {}
    for evento, color in zip(EVENTOS, [COLOR_BASE, COLOR_ALERTA, "#e0a458"]):
        por_vaca = leche.groupby("animal_id")[evento].sum().sort_values(ascending=False)
        acumulado = por_vaca.cumsum() / por_vaca.sum() * 100
        x = np.arange(1, len(acumulado) + 1) / len(acumulado) * 100
        ejes[2].plot(x, acumulado.values, color=color, linewidth=2, label=ETIQUETAS_EVENTO[evento])
        concentracion[evento] = round(float(acumulado.iloc[int(len(acumulado) * 0.1) - 1]), 1)
    ejes[2].plot([0, 100], [0, 100], color="gray", linestyle=":")
    ejes[2].set_xlabel("% de las vacas (de más a menos eventos)")
    ejes[2].set_ylabel("% acumulado de los eventos")
    ejes[2].set_title("Concentración por vaca")
    ejes[2].legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_eventos")

    return {"prevalencia_pct": prevalencia.round(2).to_dict(),
            "condicional_pct": condicional.round(1).to_dict(orient="index"),
            "pct_eventos_en_10pct_de_vacas": concentracion}


def graficar_correlaciones(leche, destino):
    """
        Grafica la matriz de correlacion de Spearman entre las variables numericas de la sesion.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Correlaciones de la producción con las demas variables.
    """
    columnas = {"milk_kg": "Producción", "duration_min": "Duración", "interval_h": "Intervalo", "flow_media": "Flujo",
                "cond_media": "Conductividad", "mdi": "MDI", "session_no": "N.º de sesión",
                "kick": "Patada", "incomplete": "Incompleto", "teat_missing": "Pezón no hallado"}
    matriz = leche[list(columnas)].corr(method="spearman").rename(index=columnas, columns=columnas)

    fig, eje = plt.subplots(figsize=(8, 6.5))
    imagen = eje.imshow(matriz.values, cmap="RdBu_r", vmin=-1, vmax=1)
    eje.set_xticks(range(len(matriz)))
    eje.set_xticklabels(matriz.columns, rotation=40, ha="right")
    eje.set_yticks(range(len(matriz)))
    eje.set_yticklabels(matriz.index)
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            eje.text(j, i, f"{matriz.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    eje.set_title("Correlación de Spearman")
    eje.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.colorbar(imagen, ax=eje, shrink=0.8)
    guardar(fig, destino, "eda_correlaciones")

    return {"con_produccion": matriz["Producción"].drop("Producción").round(2).to_dict(),
            "con_incompleto": matriz["Incompleto"].drop("Incompleto").round(2).to_dict()}


def graficar_intervalo_conductividad(leche, destino):
    """
        Grafica la tasa de eventos segun el intervalo desde el ordeño anterior y segun la conductividad
        media (hipotesis H5 y H6).

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Tasas de eventos por rango.
    """
    cortes_intervalo = [0, 4, 6, 8, 10, 12, 16, 24.01]
    intervalo = pd.cut(leche["interval_h"], cortes_intervalo, right=False)
    cuantiles = pd.qcut(leche["cond_media"], 10, duplicates="drop")
    por_intervalo = leche.groupby(intervalo, observed=True)[EVENTOS].mean().mul(100)
    por_cond = leche.groupby(cuantiles, observed=True)[EVENTOS].mean().mul(100)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    for eje, tabla, etiqueta, titulo in [
        (ejes[0], por_intervalo, "Intervalo desde el ordeño anterior (h)", "Eventos según el intervalo"),
        (ejes[1], por_cond, "Decil de conductividad media (de menor a mayor)", "Eventos según la conductividad")]:
        for evento, color in zip(EVENTOS, [COLOR_BASE, COLOR_ALERTA, "#e0a458"]):
            eje.plot(range(len(tabla)), tabla[evento].values, marker="o", color=color, label=ETIQUETAS_EVENTO[evento])
        eje.set_xticks(range(len(tabla)))
        eje.set_ylabel("% de las sesiones")
        eje.set_xlabel(etiqueta)
        eje.set_title(titulo)
    ejes[0].set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in por_intervalo.index], rotation=0)
    ejes[1].set_xticklabels(range(1, len(por_cond) + 1))
    ejes[0].legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_intervalo_conductividad")

    return {"incompleto_por_intervalo_pct": {str(k): round(float(v), 1) for k, v in por_intervalo["incomplete"].items()},
            "incompleto_por_decil_conductividad_pct": {str(i + 1): round(float(v), 1)
                                                       for i, v in enumerate(por_cond["incomplete"].values)}}


def cargar_clima(ruta):
    """
        Lee la serie horaria de NASA POWER y la corre DESFASE_CLIMA_HORAS para alinear los promedios horarios.

        Args:
            ruta (Path): CSV generado por clima_comparacion.py.

        Returns:
            DataFrame: Serie horaria local con temperatura, humedad y THI.
    """
    clima = pd.read_csv(ruta, index_col=0, parse_dates=True)
    clima.index = clima.index + pd.Timedelta(hours=DESFASE_CLIMA_HORAS)
    clima.index.name = "hora_local"
    return clima[["t2m", "rh2m", "thi"]]


def analizar_clima(leche, clima, destino):
    """
        Cruza las sesiones con el clima de NASA POWER y grafica la relacion del THI con la producción
        relativa por vaca, la hora de visita y los eventos (hipotesis H1, H2 y parte de H7).
        La producción relativa compara cada vaca-dia con la mediana de sus 28 registros anteriores,
        para quitar las diferencias entre vacas y la etapa de lactancia.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            clima (DataFrame): Serie horaria de clima con THI.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Resultados del cruce con el clima.
    """
    con_clima = leche.assign(hora_clave=leche["timestamp"].dt.floor("h")).merge(
        clima, left_on="hora_clave", right_index=True, how="left")
    cobertura = float(con_clima["thi"].notna().mean() * 100)

    diario = clima.groupby(clima.index.normalize()).agg(thi_medio=("thi", "mean"), thi_max=("thi", "max"),
                                                       temp_media=("t2m", "mean"))
    vaca_dia = leche.groupby(["animal_id", "fecha"]).agg(leche_kg=("milk_kg", "sum")).reset_index()
    vaca_dia = vaca_dia.sort_values(["animal_id", "fecha"])
    por_vaca = vaca_dia.groupby("animal_id")
    vaca_dia["referencia"] = por_vaca["leche_kg"].transform(lambda s: s.rolling(28, min_periods=14).median().shift(1))
    vaca_dia["dias_desde_previo"] = por_vaca["fecha"].diff().dt.days
    vaca_dia = vaca_dia[(vaca_dia["dias_desde_previo"] <= 3) & (vaca_dia["referencia"] > 5)]
    vaca_dia = vaca_dia[vaca_dia["fecha"] < leche["fecha"].max()]
    vaca_dia["relativa"] = vaca_dia["leche_kg"] / vaca_dia["referencia"]
    vaca_dia = vaca_dia.merge(diario, left_on="fecha", right_index=True, how="inner")

    cortes = [0, 62, 66, 70, 74, 100]
    etiquetas = ["< 62", "62-66", "66-70", "70-74", "≥ 74"]
    vaca_dia["grupo_thi"] = pd.cut(vaca_dia["thi_max"], cortes, labels=etiquetas, right=False)
    por_grupo = vaca_dia.groupby("grupo_thi", observed=True)["relativa"].agg(["median", "count"])
    cuartiles = vaca_dia.groupby("grupo_thi", observed=True)["relativa"].quantile([0.25, 0.75]).unstack()

    hato = vaca_dia.groupby("fecha").agg(relativa=("relativa", "median"), thi_medio=("thi_medio", "first"),
                                         thi_max=("thi_max", "first"))
    desfases = {}
    for dias in range(0, 6):
        desfasada = hato["thi_medio"].shift(dias)
        valido = desfasada.notna() & hato["relativa"].notna()
        rho, p = stats.spearmanr(desfasada[valido], hato.loc[valido, "relativa"])
        desfases[dias] = {"rho": round(float(rho), 3), "p": float(p)}

    # Hora de visita en dias calurosos y frescos
    dias_calor = set(diario.index[diario["thi_max"] >= 70])
    es_calor = con_clima["fecha"].isin(dias_calor)
    hora_calor = con_clima.loc[es_calor, "hora"].value_counts(normalize=True).sort_index() * 100
    hora_fresco = con_clima.loc[~es_calor, "hora"].value_counts(normalize=True).sort_index() * 100

    # Eventos segun el THI de la hora de la sesion
    con_clima["grupo_thi_hora"] = pd.cut(con_clima["thi"], [0, 55, 60, 65, 68, 100],
                                         labels=["< 55", "55-60", "60-65", "65-68", "≥ 68"], right=False)
    eventos_thi = con_clima.groupby("grupo_thi_hora", observed=True)[EVENTOS].mean().mul(100)

    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
    x = range(len(por_grupo))
    ejes[0, 0].fill_between(x, cuartiles[0.25], cuartiles[0.75], color=COLOR_GRIS)
    ejes[0, 0].plot(x, por_grupo["median"], color=COLOR_BASE, marker="o", linewidth=2)
    ejes[0, 0].axhline(1, color="gray", linestyle=":")
    ejes[0, 0].set_xticks(list(x))
    ejes[0, 0].set_xticklabels([f"{g}\n(n={n:,})" for g, n in zip(por_grupo.index, por_grupo["count"])])
    ejes[0, 0].set_xlabel("THI máximo del día")
    ejes[0, 0].set_ylabel("Leche del día / mediana de la vaca")
    ejes[0, 0].set_title("Producción relativa por vaca-día según el THI")

    ejes[0, 1].scatter(hato["thi_medio"], hato["relativa"], s=6, alpha=0.4, color=COLOR_BASE)
    ejes[0, 1].set_xlabel("THI medio del día")
    ejes[0, 1].set_ylabel("Mediana del hato de producción relativa")
    ejes[0, 1].set_title(f"Un punto por día (Spearman {desfases[0]['rho']:.2f})")

    ejes[1, 0].plot(hora_fresco.index, hora_fresco.values, color=COLOR_BASE, linewidth=2, label="Días con THI máximo < 70")
    ejes[1, 0].plot(hora_calor.index, hora_calor.values, color=COLOR_ALERTA, linewidth=2, label="Días con THI máximo ≥ 70")
    ejes[1, 0].set_xlabel("Hora local de inicio")
    ejes[1, 0].set_ylabel("% de las sesiones")
    ejes[1, 0].set_title("Hora de visita al robot")
    ejes[1, 0].legend(frameon=False, fontsize=8)

    for evento, color in zip(EVENTOS, [COLOR_BASE, COLOR_ALERTA, "#e0a458"]):
        ejes[1, 1].plot(range(len(eventos_thi)), eventos_thi[evento].values, marker="o", color=color,
                        label=ETIQUETAS_EVENTO[evento])
    ejes[1, 1].set_xticks(range(len(eventos_thi)))
    ejes[1, 1].set_xticklabels(eventos_thi.index)
    ejes[1, 1].set_xlabel("THI de la hora de la sesión")
    ejes[1, 1].set_ylabel("% de las sesiones")
    ejes[1, 1].set_title("Eventos según el THI")
    ejes[1, 1].legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_clima_thi")

    return {"cobertura_clima_pct": round(cobertura, 2),
            "dias_con_thi_max_70_o_mas": int((diario["thi_max"] >= 70).sum()), "dias_totales": int(len(diario)),
            "mediana_relativa_por_thi_max": por_grupo["median"].round(3).to_dict(),
            "vaca_dias_por_grupo": por_grupo["count"].to_dict(),
            "spearman_hato_vs_thi_medio_por_rezago_dias": desfases,
            "pct_sesiones_00_05h": {"fresco": round(float(hora_fresco.loc[0:5].sum()), 1),
                                    "calor": round(float(hora_calor.loc[0:5].sum()), 1)},
            "pct_sesiones_12_17h": {"fresco": round(float(hora_fresco.loc[12:17].sum()), 1),
                                    "calor": round(float(hora_calor.loc[12:17].sum()), 1)},
            "eventos_por_thi_hora_pct": eventos_thi.round(2).to_dict(orient="index"),
            "thi_hora_pct_sesiones_68_o_mas": round(float((con_clima["thi"] >= UMBRAL_THI).mean() * 100), 1)}


def main():
    parser = argparse.ArgumentParser(description="Genera las figuras y los indicadores de la exploración de datos.")
    parser.add_argument("--tablas", type=Path, default=tablas_defecto,
                        help="Carpeta con leche_eventos_normalizados.csv.gz")
    parser.add_argument("--clima", type=Path, default=clima_defecto, help="CSV de NASA POWER de clima_comparacion.py")
    parser.add_argument("--salida", type=Path, default=salida_defecto, help="Carpeta de salida de las imágenes")
    parser.add_argument("--resumen", type=Path, default=resumen_defecto,
                        help="JSON con los indicadores (fuera de Git)")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    args.resumen.parent.mkdir(parents=True, exist_ok=True)

    leche = cargar_leche(args.tablas)
    print(f"Sesiones leídas: {len(leche):,} de {leche['animal_id'].nunique()} vacas")

    resumen = {"calidad": perfilar_calidad(leche)}
    graficar_distribuciones(leche, args.salida)
    resumen["temporal"] = graficar_temporal(leche, args.salida)
    resumen["lactancia"] = graficar_lactancia(leche, args.salida)
    resumen["cuartos_vms"] = graficar_cuartos_vms(leche, args.salida)
    resumen["eventos"] = graficar_eventos(leche, args.salida)
    resumen["correlaciones"] = graficar_correlaciones(leche, args.salida)
    resumen["intervalo_conductividad"] = graficar_intervalo_conductividad(leche, args.salida)
    resumen["clima"] = analizar_clima(leche, cargar_clima(args.clima), args.salida)

    args.resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(resumen, indent=1, ensure_ascii=False, default=str))
    print(f"Figuras guardadas en: {args.salida}")


if __name__ == "__main__":
    main()
