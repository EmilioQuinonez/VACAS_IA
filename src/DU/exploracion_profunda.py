import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from exploracion_datos import (CUARTOS, EVENTOS, ETIQUETAS_EVENTO, UMBRAL_THI, COLOR_BASE,
                               COLOR_ALERTA, COLOR_GRIS, guardar, cargar_leche, cargar_clima, produccion_relativa,
                               datos_defecto, clima_defecto)

raiz = Path(__file__).resolve().parent.parent.parent
resumen_defecto = raiz / "Data" / "eda" / "resumen_profundo.json"
salida_defecto = raiz / "docs" / "DU" / "figuras"

SEMILLA = 42
COLOR_AUXILIAR = "#e0a458"
COLORES_EVENTO = dict(zip(EVENTOS, [COLOR_BASE, COLOR_ALERTA, COLOR_AUXILIAR]))
Z95 = 1.96
VARIABLES_AGREGADAS = ["milk_kg", "duration_min", "interval_h", "flow_media", "cond_media", "mdi"]
ETIQUETAS_VARIABLE = {"milk_kg": "Producción (kg)", "duration_min": "Duración (min)", "interval_h": "Intervalo (h)",
                      "flow_media": "Flujo medio (kg/min)", "cond_media": "Conductividad media (mS/cm)", "mdi": "MDI"}

# (nombre, columnas, tipo, unidad, descripcion)
DICCIONARIO = [
    ("timestamp", ["timestamp"], "Fecha-hora", "hora local", "Inicio de la sesión de ordeño"),
    ("animal_id", ["animal_id"], "Identificador", "—", "Vaca (hash HMAC de 8 caracteres de anonymize_data.py)"),
    ("milk_kg", ["milk_kg"], "Numérica continua", "kg", "Leche de la sesión"),
    ("session_no", ["session_no"], "Discreta", "—", "Número de sesión del día"),
    ("duration_min", ["duration_min"], "Numérica continua", "min", "Duración de la sesión"),
    ("interval_h", ["interval_h"], "Numérica continua", "h", "Tiempo desde el ordeño anterior"),
    ("q_*", [f"q_{c}" for c in CUARTOS], "Numérica continua", "kg", "Leche por cuarto (DI, DD, TI, TD)"),
    ("flow_*", [f"flow_{c}" for c in CUARTOS], "Numérica continua", "kg/min", "Flujo por cuarto"),
    ("blood_*", [f"blood_{c}" for c in CUARTOS], "Numérica continua", "ppm", "Sangre por cuarto"),
    ("cond_*", [f"cond_{c}" for c in CUARTOS], "Numérica continua", "mS/cm", "Conductividad por cuarto"),
    ("mdi", ["mdi"], "Numérica continua", "—", "Índice del robot, sin unidad documentada"),
    ("rcs", ["rcs"], "Numérica continua", "—", "Conteo de células somáticas (vacío)"),
    ("machine", ["machine"], "Categórica", "—", "Robot de ordeño (hash HMAC de anonymize_data.py)"),
    ("destination", ["destination"], "Categórica", "—", "Destino de la leche (Tanque, Drenaje, Divert 1 a 3)"),
    ("kick", ["kick"], "Binaria", "0/1", "Patada"),
    ("incomplete", ["incomplete"], "Binaria", "0/1", "Ordeño incompleto"),
    ("teat_missing", ["teat_missing"], "Binaria", "0/1", "Pezón no encontrado"),
]


def proporcion_ic(positivos, total):
    """
        Calcula una proporcion en porcentaje con su intervalo de confianza del 95 %: p ± 1.96 · EE,
        con EE = raiz(p (1 - p) / n). Supone sesiones independientes, por lo que es optimista
        cuando las sesiones de una misma vaca se parecen entre si.

        Args:
            positivos (float): Cantidad de casos con el evento.
            total (float): Cantidad de casos observados.

        Returns:
            tuple: Porcentaje y semiancho del intervalo, ambos en puntos porcentuales.
    """
    p = positivos / total
    return float(p * 100), float(Z95 * np.sqrt(p * (1 - p) / total) * 100)


def construir_diccionario(leche):
    """
        Resume cada variable de la tabla: tipo, unidad, descripcion, porcentaje de nulos y rango tipico.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.

        Returns:
            DataFrame: Una fila por variable (o grupo de cuatro cuartos).
    """
    filas = []
    for nombre, columnas, tipo, unidad, descripcion in DICCIONARIO:
        valores = leche[columnas]
        fila = {"variable": nombre, "tipo": tipo, "unidad": unidad, "descripcion": descripcion,
                "nulos_pct": round(float(valores.isna().to_numpy().mean() * 100), 1)}
        if tipo.startswith(("Numérica", "Discreta", "Binaria")) and valores.notna().to_numpy().any():
            apilado = valores.to_numpy(dtype=float).ravel()
            apilado = apilado[~np.isnan(apilado)]
            fila["mediana"] = round(float(np.median(apilado)), 2)
            fila["p1_p99"] = f"{np.percentile(apilado, 1):.2f} a {np.percentile(apilado, 99):.2f}"
        elif tipo == "Categórica":
            fila["categorias"] = int(leche[columnas[0]].nunique())
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_pairplot(leche, destino):
    """
        Grafica el diagrama de pares de las variables numericas sobre una muestra aleatoria,
        con las sesiones con algun evento en rojo.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.
    """
    muestra = leche[VARIABLES_AGREGADAS + ["evento_alguno"]].dropna().sample(4000, random_state=SEMILLA)
    muestra = muestra.rename(columns=ETIQUETAS_VARIABLE)
    muestra["Evento"] = muestra["evento_alguno"].map({0: "Sin evento", 1: "Algún evento"})
    grafica = sns.pairplot(muestra.drop(columns="evento_alguno"), hue="Evento", corner=True, diag_kind="hist",
                           palette={"Sin evento": COLOR_GRIS, "Algún evento": COLOR_ALERTA},
                           plot_kws={"s": 5, "alpha": 0.45, "linewidth": 0}, height=1.9)
    grafica.figure.savefig(destino / "eda_pairplot.png", dpi=170, bbox_inches="tight")
    plt.close(grafica.figure)


def describir_variables(leche, destino):
    """
        Calcula media, mediana, desviacion estandar, IQR y atipicos (1.5 IQR) de las variables con cola
        derecha, y grafica su histograma en escala original y en logaritmo.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Estadisticos descriptivos y conteo de atipicos por variable.
    """
    variables = [("milk_kg", "Producción por sesión (kg)"), ("duration_min", "Duración (min)"),
                 ("interval_h", "Intervalo (h)")]
    resultado = {}
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    for columna, (variable, etiqueta) in enumerate(variables):
        serie = leche[variable].dropna()
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        positiva = serie[serie > 0]
        resultado[variable] = {
            "n": int(len(serie)), "media": round(float(serie.mean()), 2), "mediana": round(float(serie.median()), 2),
            "desviacion": round(float(serie.std()), 2), "q1": round(float(q1), 2), "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2), "limite_bajo": round(float(bajo), 2), "limite_alto": round(float(alto), 2),
            "atipicos_bajos": int((serie < bajo).sum()), "atipicos_altos": int((serie > alto).sum()),
            "atipicos_pct": round(float(((serie < bajo) | (serie > alto)).mean() * 100), 2),
            "media_menos_mediana_pct": round(float((serie.mean() - serie.median()) / serie.median() * 100), 1),
            "log_media": round(float(np.log(positiva).mean()), 3), "log_mediana": round(float(np.log(positiva).median()), 3),
            "log_desviacion": round(float(np.log(positiva).std()), 3),
            "no_positivos_excluidos_del_log": int((serie <= 0).sum()),
        }
        recorte = serie[serie.between(serie.quantile(0.001), serie.quantile(0.999))]
        ejes[0, columna].hist(recorte, bins=60, color=COLOR_BASE)
        ejes[0, columna].axvline(serie.mean(), color=COLOR_ALERTA, linewidth=1.5, label=f"media {serie.mean():.1f}")
        ejes[0, columna].axvline(serie.median(), color="black", linestyle="--", linewidth=1.3,
                                 label=f"mediana {serie.median():.1f}")
        ejes[0, columna].set_xlabel(etiqueta)
        ejes[0, columna].set_ylabel("Sesiones")
        ejes[0, columna].legend(frameon=False, fontsize=8)

        logaritmo = np.log(positiva)
        logaritmo = logaritmo[logaritmo.between(logaritmo.quantile(0.001), logaritmo.quantile(0.999))]
        ejes[1, columna].hist(logaritmo, bins=60, color=COLOR_AUXILIAR)
        ejes[1, columna].axvline(logaritmo.mean(), color=COLOR_ALERTA, linewidth=1.5, label=f"media {logaritmo.mean():.2f}")
        ejes[1, columna].axvline(logaritmo.median(), color="black", linestyle="--", linewidth=1.3,
                                 label=f"mediana {logaritmo.median():.2f}")
        ejes[1, columna].set_xlabel(f"ln({etiqueta})")
        ejes[1, columna].set_ylabel("Sesiones")
        ejes[1, columna].legend(frameon=False, fontsize=8)
    fig.suptitle("Histogramas en escala original (arriba) y en logaritmo natural (abajo); se recorta el 0.1 % de cada cola")
    guardar(fig, destino, "eda_histogramas_log")
    return resultado


def comparar_escalas_temporales(leche, destino):
    """
        Compara la distribucion de la producción por sesion, por vaca-dia y por vaca-semana
        (solo semanas con al menos 6 dias observados).

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Media, mediana, desviacion y IQR de cada escala.
    """
    por_dia = leche.groupby(["animal_id", "fecha"]).agg(kg=("milk_kg", "sum"), sesiones=("milk_kg", "size")).reset_index()
    por_dia["semana"] = por_dia["fecha"].dt.to_period("W").dt.start_time
    por_semana = por_dia.groupby(["animal_id", "semana"]).agg(kg=("kg", "sum"), dias=("fecha", "nunique")).reset_index()
    por_semana = por_semana[por_semana["dias"] >= 6]

    escalas = [("Sesión", leche["milk_kg"].dropna(), (0, 45)), ("Vaca-día", por_dia["kg"], (0, 90)),
               ("Vaca-semana", por_semana["kg"], (0, 600))]
    resultado = {}
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
    for eje, (nombre, serie, limites) in zip(ejes, escalas):
        q1, q3 = serie.quantile([0.25, 0.75])
        resultado[nombre] = {"n": int(len(serie)), "media": round(float(serie.mean()), 2),
                             "mediana": round(float(serie.median()), 2), "desviacion": round(float(serie.std()), 2),
                             "iqr": round(float(q3 - q1), 2),
                             "media_menos_mediana_pct": round(float((serie.mean() - serie.median()) / serie.median() * 100), 1)}
        eje.hist(serie[serie.between(*limites)], bins=60, color=COLOR_BASE)
        eje.axvline(serie.mean(), color=COLOR_ALERTA, linewidth=1.5, label=f"media {serie.mean():.1f}")
        eje.axvline(serie.median(), color="black", linestyle="--", linewidth=1.3, label=f"mediana {serie.median():.1f}")
        eje.set_xlabel(f"Leche por {nombre.lower()} (kg)")
        eje.set_ylabel("Frecuencia")
        eje.set_title(f"{nombre} (n = {len(serie):,})")
        eje.legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_escalas_temporales")
    resultado["sesiones_por_vaca_dia"] = por_dia["sesiones"].value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()
    return resultado


def analizar_eventos_en_el_tiempo(leche, destino):
    """
        Grafica la tasa mensual de cada evento y las tasas por robot y por ano con su intervalo de confianza del 95 %.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Tasas con IC por robot, por ano y por mes (resumen).
    """
    mensual = leche.groupby("mes")[EVENTOS].mean().mul(100)
    sesiones_mes = leche.groupby("mes").size()
    resultado = {"por_robot": {}, "por_anio": {}}
    for clave, agrupador in [("por_robot", leche["machine"]), ("por_anio", leche["timestamp"].dt.year)]:
        for grupo, bloque in leche.groupby(agrupador):
            resultado[clave][str(grupo)] = {"sesiones": int(len(bloque))}
            for evento in EVENTOS:
                tasa, ic = proporcion_ic(bloque[evento].sum(), len(bloque))
                resultado[clave][str(grupo)][evento] = {"pct": round(tasa, 2), "ic95_mas_menos": round(ic, 2)}

    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2.2, 1, 1.3]})
    for evento in EVENTOS:
        ejes[0].plot(mensual.index, mensual[evento], color=COLORES_EVENTO[evento], linewidth=2, label=ETIQUETAS_EVENTO[evento])
    ejes[0].set_ylabel("% de las sesiones del mes")
    ejes[0].set_title("Tasa mensual de cada evento")
    ejes[0].legend(frameon=False, fontsize=8, loc="upper left")
    ejes[0].set_ylim(0, None)
    reserva = ejes[0].twinx()
    reserva.bar(sesiones_mes.index, sesiones_mes.values, width=25, color=COLOR_GRIS, alpha=0.5, zorder=0)
    reserva.set_ylabel("Sesiones por mes")
    reserva.spines["top"].set_visible(False)
    ejes[0].set_zorder(reserva.get_zorder() + 1)
    ejes[0].patch.set_visible(False)

    for eje, clave, titulo in [(ejes[1], "por_robot", "Por robot"), (ejes[2], "por_anio", "Por año")]:
        grupos = list(resultado[clave])
        ancho = 0.8 / len(EVENTOS)
        for i, evento in enumerate(EVENTOS):
            valores = [resultado[clave][g][evento]["pct"] for g in grupos]
            errores = [resultado[clave][g][evento]["ic95_mas_menos"] for g in grupos]
            eje.bar(np.arange(len(grupos)) + i * ancho, valores, ancho, yerr=errores, capsize=3,
                    color=COLORES_EVENTO[evento], label=ETIQUETAS_EVENTO[evento])
        eje.set_xticks(np.arange(len(grupos)) + ancho)
        eje.set_xticklabels(grupos)
        eje.set_title(f"{titulo} (IC 95 %)")
        eje.set_ylabel("% de las sesiones")
    guardar(fig, destino, "eda_eventos_en_el_tiempo")

    resultado["mensual_min_max"] = {e: [round(float(mensual[e].min()), 1), round(float(mensual[e].max()), 1)] for e in EVENTOS}
    return resultado


def correlaciones_con_significancia(leche, destino, carpeta_datos):
    """
        Calcula covarianza, correlacion de Pearson, R2 y su prueba t bilateral (t = r raiz(n-2) / raiz(1-r2))
        para cada par de variables numericas y para cada variable contra los tres eventos (correlacion
        punto-biserial). Guarda la tabla completa y grafica la matriz de correlacion.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.
            carpeta_datos (Path): Carpeta (fuera de Git) donde se guarda la tabla completa en CSV.

        Returns:
            dict: Los pares con mayor correlacion absoluta y la cantidad de pares significativos.
    """
    columnas = VARIABLES_AGREGADAS + EVENTOS
    filas = []
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            if a in EVENTOS:
                continue
            par = leche[[a, b]].dropna()
            n = len(par)
            r = par[a].corr(par[b])
            t = r * np.sqrt((n - 2) / (1 - r ** 2))
            filas.append({"x": a, "y": b, "n": n, "covarianza": par[a].cov(par[b]), "r": r, "r2": r ** 2,
                          "spearman": par[a].corr(par[b], method="spearman"), "t": t,
                          "p": float(2 * stats.t.sf(abs(t), n - 2))})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta_datos / "correlaciones_significancia.csv", index=False)

    matriz = leche[columnas].corr().rename(index=ETIQUETAS_VARIABLE | ETIQUETAS_EVENTO,
                                           columns=ETIQUETAS_VARIABLE | ETIQUETAS_EVENTO)
    fig, eje = plt.subplots(figsize=(8.2, 6.8))
    imagen = eje.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)
    eje.set_xticks(range(len(matriz)))
    eje.set_xticklabels(matriz.columns, rotation=45, ha="right")
    eje.set_yticks(range(len(matriz)))
    eje.set_yticklabels(matriz.index)
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            eje.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(imagen, ax=eje, shrink=0.8, label="Correlación de Pearson")
    eje.set_title("Correlación de Pearson (eventos como 0/1)")
    guardar(fig, destino, "eda_correlacion_pearson")

    por_evento = {}
    for evento in EVENTOS:
        r = pd.Series({v: leche[v].corr(leche[evento]) for v in VARIABLES_AGREGADAS})
        por_evento[evento] = {"r": r.round(3).to_dict(), "r2_maximo": round(float((r ** 2).max()), 4)}
    principales = tabla.reindex(tabla["r"].abs().sort_values(ascending=False).index).head(6)
    return {"pares_numericos": principales.round(4).to_dict(orient="records"),
            "pares_numericos_con_p_menor_0_001": int((tabla["p"] < 0.001).sum()), "pares_numericos_total": int(len(tabla)),
            "con_eventos": por_evento}


def analizar_cuartos_sin_leche(leche):
    """
        Mide si los cuartos sin leche (0 kg) y las sesiones sin medida de flujo o conductividad se concentran
        en las sesiones con evento. Sirve para saber si el faltante es informativo y no debe imputarse sin mas.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.

        Returns:
            dict: Por evento, % de sesiones con algun cuarto en 0 kg y % sin flujo medio, con y sin evento.
    """
    algun_cuarto_en_cero = (leche[[f"q_{c}" for c in CUARTOS]] == 0).any(axis=1)
    sin_flujo = leche["flow_media"].isna()
    resultado = {"sesiones_con_algun_cuarto_en_0kg_pct": round(float(algun_cuarto_en_cero.mean() * 100), 1),
                 "sesiones_sin_flujo_medio_pct": round(float(sin_flujo.mean() * 100), 2)}
    for evento in EVENTOS:
        con = leche[evento] == 1
        resultado[evento] = {
            "algun_cuarto_en_0kg_con_evento_pct": round(float(algun_cuarto_en_cero[con].mean() * 100), 1),
            "algun_cuarto_en_0kg_sin_evento_pct": round(float(algun_cuarto_en_cero[~con].mean() * 100), 1),
            "sin_flujo_medio_con_evento_pct": round(float(sin_flujo[con].mean() * 100), 2),
            "sin_flujo_medio_sin_evento_pct": round(float(sin_flujo[~con].mean() * 100), 2)}
    return resultado


def calcular_vif(leche):
    """
        Calcula el factor de inflacion de la varianza (VIF) de dos conjuntos de variables. Un VIF mayor a 5
        indica que la variable se explica bien con las demas. Los ceros de flujo y conductividad son faltantes.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.

        Returns:
            dict: VIF de las variables agregadas y de las variables por cuarto, con el numero de filas usadas.
    """
    conjuntos = {"agregadas": VARIABLES_AGREGADAS,
                 "por_cuarto": ["milk_kg", "duration_min", "interval_h", "mdi"] + [f"flow_{c}" for c in CUARTOS]
                               + [f"cond_{c}" for c in CUARTOS]}
    resultado = {}
    for nombre, columnas in conjuntos.items():
        datos = leche[columnas].astype(float).copy()
        for columna in columnas:
            if columna.startswith(("flow_", "cond_")) and columna not in ("flow_media", "cond_media"):
                datos[columna] = datos[columna].where(datos[columna] > 0)
        datos = datos.dropna()
        usadas = len(datos)
        datos = datos.sample(min(60000, usadas), random_state=SEMILLA)
        exog = sm.add_constant(datos)
        vif = {c: round(float(variance_inflation_factor(exog.to_numpy(), i)), 2)
               for i, c in enumerate(exog.columns) if c != "const"}
        resultado[nombre] = {"filas_completas": int(usadas), "pct_filas_completas": round(usadas / len(leche) * 100, 1),
                             "vif": vif}
    return resultado


def analizar_persistencia(leche, destino):
    """
        Mide si un evento en la sesion anterior de la misma vaca (a menos de 24 h) eleva la probabilidad
        de evento en la sesion actual. Compara con la tasa esperada si solo importara la vaca: el promedio
        de la tasa propia de cada vaca ponderado por las sesiones que siguen a un evento.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Probabilidad condicional observada y esperada por evento.
    """
    ordenado = leche.sort_values(["animal_id", "timestamp"]).copy()
    horas = ordenado.groupby("animal_id")["timestamp"].diff().dt.total_seconds() / 3600
    tasa_vaca = ordenado.groupby("animal_id")[EVENTOS].transform("mean")
    resultado = {}
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for eje, evento in zip(ejes, EVENTOS):
        previo = ordenado.groupby("animal_id")[evento].shift(1)
        valido = (horas <= 24) & previo.notna()
        actual = ordenado.loc[valido, evento]
        previo = previo[valido]
        esperado = tasa_vaca.loc[valido, evento][previo == 1].mean() * 100
        con_previo, ic_con = proporcion_ic(actual[previo == 1].sum(), (previo == 1).sum())
        sin_previo, ic_sin = proporcion_ic(actual[previo == 0].sum(), (previo == 0).sum())
        resultado[evento] = {
            "sesiones_con_previo_valido": int(valido.sum()),
            "p_evento_si_previo_con_evento_pct": round(con_previo, 2), "ic95_con": round(ic_con, 2),
            "p_evento_si_previo_sin_evento_pct": round(sin_previo, 2), "ic95_sin": round(ic_sin, 2),
            "esperado_si_solo_importara_la_vaca_pct": round(float(esperado), 2),
            "razon_observada_sobre_sin_previo": round(con_previo / sin_previo, 2),
            "razon_observada_sobre_esperado": round(con_previo / esperado, 2),
        }
        eje.bar([0, 1], [sin_previo, con_previo], yerr=[ic_sin, ic_con], capsize=4, color=[COLOR_GRIS, COLORES_EVENTO[evento]])
        eje.scatter([1], [esperado], marker="_", s=900, color="black", linewidth=2, zorder=3, label="esperado por la vaca")
        eje.set_xticks([0, 1])
        eje.set_xticklabels(["Sesión previa\nsin evento", "Sesión previa\ncon evento"])
        eje.set_title(ETIQUETAS_EVENTO[evento])
        eje.legend(frameon=False, fontsize=8, loc="upper left")
    ejes[0].set_ylabel("% de sesiones con el evento")
    guardar(fig, destino, "eda_persistencia")
    return resultado


def analizar_adaptacion(leche, destino):
    """
        Mide la tasa de eventos segun los dias desde el primer registro de cada vaca, solo para vacas
        cuyo primer registro es 60 dias o mas posterior al inicio de los datos (las demas ya estaban en el hato).

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Tasa de cada evento con IC por intervalo de dias desde el primer registro.
    """
    primera = leche.groupby("animal_id")["timestamp"].transform("min")
    nuevas = primera > leche["timestamp"].min() + pd.Timedelta(days=60)
    datos = leche[nuevas].assign(dias=(leche["timestamp"] - primera).dt.days)
    cortes = [0, 7, 14, 30, 60, 120, 240, 100000]
    etiquetas = ["0-6", "7-13", "14-29", "30-59", "60-119", "120-239", "≥ 240"]
    datos["intervalo"] = pd.cut(datos["dias"], cortes, labels=etiquetas, right=False)

    resultado = {"vacas_nuevas": int(datos["animal_id"].nunique()), "sesiones": int(len(datos)), "intervalos": {}}
    fig, eje = plt.subplots(figsize=(8.5, 4.4))
    for evento in EVENTOS:
        tasas, errores = [], []
        for etiqueta in etiquetas:
            bloque = datos[datos["intervalo"] == etiqueta]
            tasa, ic = proporcion_ic(bloque[evento].sum(), len(bloque))
            tasas.append(tasa)
            errores.append(ic)
            resultado["intervalos"].setdefault(etiqueta, {"sesiones": int(len(bloque))})[evento] = {
                "pct": round(tasa, 2), "ic95_mas_menos": round(ic, 2)}
        eje.errorbar(range(len(etiquetas)), tasas, yerr=errores, marker="o", capsize=3, color=COLORES_EVENTO[evento],
                     label=ETIQUETAS_EVENTO[evento])
    eje.set_xticks(range(len(etiquetas)))
    eje.set_xticklabels(etiquetas)
    eje.set_xlabel("Días desde el primer registro de la vaca")
    eje.set_ylabel("% de las sesiones (IC 95 %)")
    eje.set_title(f"Adaptación al robot ({resultado['vacas_nuevas']} vacas que ingresan después del inicio)")
    eje.legend(frameon=False, fontsize=8)
    guardar(fig, destino, "eda_adaptacion")
    return resultado


def ajustar_pca(datos, nombre_variables):
    """
        Estandariza las variables (z-score) y ajusta un PCA con todos los componentes.

        Args:
            datos (DataFrame): Filas completas, una columna por variable.
            nombre_variables (list): Columnas a usar.

        Returns:
            tuple: PCA ajustado, puntajes (ndarray) y cargas (DataFrame variable x componente).
    """
    estandarizado = StandardScaler().fit_transform(datos[nombre_variables])
    pca = PCA().fit(estandarizado)
    cargas = pd.DataFrame(pca.components_.T, index=nombre_variables,
                          columns=[f"PC{i + 1}" for i in range(len(nombre_variables))])
    return pca, pca.transform(estandarizado), cargas


def resumir_pca(pca, cargas, componentes=4):
    """
        Resume un PCA: varianza explicada, componentes para el 80 % y variables con mayor carga.

        Args:
            pca (PCA): PCA ajustado.
            cargas (DataFrame): Cargas por variable y componente.
            componentes (int): Cuantos componentes detallar.

        Returns:
            dict: Varianza explicada, acumulada, componentes para 80 % y 90 %, y cargas principales.
    """
    explicada = pca.explained_variance_ratio_
    acumulada = np.cumsum(explicada)
    return {
        "varianza_explicada_pct": (explicada[:componentes] * 100).round(1).tolist(),
        "varianza_acumulada_pct": (acumulada[:componentes + 2] * 100).round(1).tolist(),
        "componentes_para_80pct": int(np.searchsorted(acumulada, 0.80) + 1),
        "componentes_para_90pct": int(np.searchsorted(acumulada, 0.90) + 1),
        "componentes_con_valor_propio_mayor_a_1": int((pca.explained_variance_ > 1).sum()),
        "cargas_principales": {c: cargas[c].round(2).reindex(cargas[c].abs().sort_values(ascending=False).index).head(4).to_dict()
                               for c in cargas.columns[:componentes]},
    }


def separacion_por_evento(datos, puntajes):
    """
        Mide si los primeros tres componentes separan las sesiones con y sin cada evento: diferencia de medias,
        su IC 95 % (t de Welch) y la d de Cohen (diferencia entre la desviacion estandar combinada).

        Args:
            datos (DataFrame): Filas usadas en el PCA, con las columnas de eventos.
            puntajes (ndarray): Puntajes de cada fila en los componentes.

        Returns:
            dict: Por evento y componente, diferencia, IC 95 % y d de Cohen.
    """
    resultado = {}
    for evento in EVENTOS:
        con = datos[evento].to_numpy() == 1
        resultado[evento] = {"sesiones_con_evento": int(con.sum()), "prevalencia_pct": round(float(con.mean() * 100), 1)}
        for k in range(3):
            a, b = puntajes[con, k], puntajes[~con, k]
            diferencia = a.mean() - b.mean()
            error = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            desviacion_comun = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            resultado[evento][f"PC{k + 1}"] = {
                "diferencia_medias": round(float(diferencia), 3),
                "ic95": [round(float(diferencia - Z95 * error), 3), round(float(diferencia + Z95 * error), 3)],
                "d_cohen": round(float(diferencia / desviacion_comun), 3)}
    return resultado


def analizar_pca_ordeno(leche, destino):
    """
        Ajusta dos PCA sobre las variables de la sesion de ordeño: uno con las variables agregadas (conserva casi
        todas las filas) y otro con las variables por cuarto (solo sesiones con los cuatro cuartos ordeñados).
        Grafica la varianza explicada, las cargas y la proyeccion en PC1-PC2 coloreada por evento.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Resumen de ambos PCA y separacion de los eventos en PC1-PC3.
    """
    por_cuarto = ["milk_kg", "duration_min", "interval_h", "mdi"] + [f"flow_{c}" for c in CUARTOS] + [f"cond_{c}" for c in CUARTOS]
    datos_cuarto = leche[por_cuarto + EVENTOS].astype(float).copy()
    for columna in por_cuarto:
        if columna.startswith(("flow_", "cond_")):
            datos_cuarto[columna] = datos_cuarto[columna].where(datos_cuarto[columna] > 0)
    datos_cuarto = datos_cuarto.dropna()
    datos_agregados = leche[VARIABLES_AGREGADAS + EVENTOS].astype(float).dropna()

    pca_a, puntajes_a, cargas_a = ajustar_pca(datos_agregados, VARIABLES_AGREGADAS)
    pca_c, puntajes_c, cargas_c = ajustar_pca(datos_cuarto, por_cuarto)
    resultado = {
        "agregadas": {"filas": int(len(datos_agregados)), "pct_filas": round(len(datos_agregados) / len(leche) * 100, 1),
                      **resumir_pca(pca_a, cargas_a), "separacion_por_evento": separacion_por_evento(datos_agregados, puntajes_a)},
        "por_cuarto": {"filas": int(len(datos_cuarto)), "pct_filas": round(len(datos_cuarto) / len(leche) * 100, 1),
                       **resumir_pca(pca_c, cargas_c), "separacion_por_evento": separacion_por_evento(datos_cuarto, puntajes_c)},
    }

    fig, ejes = plt.subplots(2, 2, figsize=(13, 9.5))
    for pca, nombre, color in [(pca_a, "Agregadas (6 variables)", COLOR_BASE), (pca_c, "Por cuarto (12 variables)", COLOR_ALERTA)]:
        posicion = np.arange(1, len(pca.explained_variance_ratio_) + 1)
        ejes[0, 0].bar(posicion + (0.2 if color == COLOR_ALERTA else -0.2), pca.explained_variance_ratio_ * 100, 0.4,
                       color=color, alpha=0.45)
        ejes[0, 0].plot(posicion, np.cumsum(pca.explained_variance_ratio_) * 100, marker="o", color=color, label=nombre)
    ejes[0, 0].axhline(80, color="gray", linestyle=":")
    ejes[0, 0].set_xlabel("Componente")
    ejes[0, 0].set_ylabel("% de la varianza (barras) y acumulada (líneas)")
    ejes[0, 0].set_title("Varianza explicada")
    ejes[0, 0].legend(frameon=False, fontsize=8, loc="center right")

    for eje, cargas, titulo in [(ejes[0, 1], cargas_a, "Cargas del PCA agregado"), (ejes[1, 0], cargas_c, "Cargas del PCA por cuarto")]:
        imagen = eje.imshow(cargas.iloc[:, :4], cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
        eje.set_xticks(range(4))
        eje.set_xticklabels(cargas.columns[:4])
        eje.set_yticks(range(len(cargas)))
        eje.set_yticklabels(cargas.index, fontsize=8)
        for i in range(len(cargas)):
            for j in range(4):
                eje.text(j, i, f"{cargas.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        eje.set_title(titulo)
        fig.colorbar(imagen, ax=eje, shrink=0.8)

    muestra = np.random.default_rng(SEMILLA).choice(len(datos_agregados), 8000, replace=False)
    marcados = datos_agregados[EVENTOS].to_numpy()[muestra].max(axis=1) == 1
    ejes[1, 1].scatter(puntajes_a[muestra][~marcados, 0], puntajes_a[muestra][~marcados, 1], s=4, color=COLOR_GRIS, alpha=0.5,
                       label="Sin evento")
    ejes[1, 1].scatter(puntajes_a[muestra][marcados, 0], puntajes_a[muestra][marcados, 1], s=4, color=COLOR_ALERTA, alpha=0.6,
                       label="Algún evento")
    ejes[1, 1].set_xlabel(f"PC1 ({pca_a.explained_variance_ratio_[0] * 100:.0f} %)")
    ejes[1, 1].set_ylabel(f"PC2 ({pca_a.explained_variance_ratio_[1] * 100:.0f} %)")
    ejes[1, 1].set_title("Sesiones en PC1-PC2 del PCA agregado (muestra de 8,000)")
    ejes[1, 1].legend(frameon=False, fontsize=8, markerscale=3)
    guardar(fig, destino, "eda_pca_ordeno")
    return resultado


def analizar_pca_clima(clima, destino):
    """
        Ajusta un PCA a trece resumenes diarios del clima (los nueve de temperatura,
        humedad y THI, mas viento, radiacion, lluvia y nubosidad de NASA POWER) para ver cuanta informacion distinta aportan
        y si basta un solo indice (el THI) para describir el dia.

        Args:
            clima (DataFrame): Tabla horaria de clima.
            destino (Path): Carpeta donde se guarda la imagen.

        Returns:
            dict: Resumen del PCA del clima diario.
    """
    dias = clima.groupby(clima.index.normalize())
    diario = pd.DataFrame({
        "t_media": dias["t2m"].mean(), "t_max": dias["t2m"].max(), "t_min": dias["t2m"].min(),
        "rh_media": dias["rh2m"].mean(), "thi_medio": dias["thi"].mean(), "thi_max": dias["thi"].max(),
        "thi_min": dias["thi"].min(), "horas_thi_alto": dias["thi"].apply(lambda s: (s >= UMBRAL_THI).sum()),
        "amplitud_t": dias["t2m"].max() - dias["t2m"].min(), "viento": dias["ws2m"].mean(),
        "radiacion": dias["allsky_sfc_sw_dwn"].mean(), "lluvia": dias["prectotcorr"].sum(min_count=24),
        "nubosidad": dias["cloud_amt"].mean()})
    diario = diario[dias.size() == 24].dropna()
    pca, _, cargas = ajustar_pca(diario, list(diario.columns))
    resultado = {"dias": int(len(diario)), **resumir_pca(pca, cargas, componentes=3)}

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ejes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, color=COLOR_BASE)
    ejes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_) * 100,
                 marker="o", color=COLOR_ALERTA)
    ejes[0].axhline(80, color="gray", linestyle=":")
    ejes[0].set_xlabel("Componente")
    ejes[0].set_ylabel("% de la varianza (barras) y acumulada (línea)")
    ejes[0].set_title("Varianza explicada del clima diario")
    imagen = ejes[1].imshow(cargas.iloc[:, :3], cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ejes[1].set_xticks(range(3))
    ejes[1].set_xticklabels(cargas.columns[:3])
    ejes[1].set_yticks(range(len(cargas)))
    ejes[1].set_yticklabels(cargas.index)
    for i in range(len(cargas)):
        for j in range(3):
            ejes[1].text(j, i, f"{cargas.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ejes[1].set_title("Cargas")
    fig.colorbar(imagen, ax=ejes[1], shrink=0.8)
    guardar(fig, destino, "eda_pca_clima")
    return resultado


def sensibilidad_umbral_clima(leche, clima):
    """
        Repite las comparaciones de las hipotesis H1 y H2 con tres umbrales de THI maximo diario.
        Usa un dato por dia (la mediana del hato de la producción relativa) y una t de Welch, como en W2:
        diferencia de medias = media(dias calurosos) - media(dias frescos), con IC 95 %.

        Args:
            leche (DataFrame): Tabla de sesiones de ordeño.
            clima (DataFrame): Tabla horaria de clima.

        Returns:
            dict: Por umbral, dias en cada grupo, diferencia de producción relativa y de % de visitas de madrugada.
    """
    diario = clima.groupby(clima.index.normalize()).agg(thi_max=("thi", "max"))
    vaca_dia = produccion_relativa(leche).merge(diario, left_on="fecha", right_index=True)
    hato = vaca_dia.groupby("fecha").agg(relativa=("relativa", "median"), thi_max=("thi_max", "first"))
    madrugada = leche.assign(madrugada=leche["hora"] <= 5).groupby("fecha")["madrugada"].mean().mul(100)
    hato = hato.join(madrugada.rename("pct_madrugada"))

    def welch(caliente, fresco):
        a, b = caliente.dropna(), fresco.dropna()
        diferencia = a.mean() - b.mean()
        error = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return {"diferencia": round(float(diferencia), 4), "ic95": [round(float(diferencia - Z95 * error), 4),
                                                                      round(float(diferencia + Z95 * error), 4)],
                "t": round(float(t), 2), "p": float(p)}

    resultado = {}
    for umbral in [68, 70, 72]:
        caliente = hato["thi_max"] >= umbral
        resultado[str(umbral)] = {"dias_calurosos": int(caliente.sum()), "dias_frescos": int((~caliente).sum()),
                                  "produccion_relativa": welch(hato.loc[caliente, "relativa"], hato.loc[~caliente, "relativa"]),
                                  "pct_visitas_00_05h": welch(hato.loc[caliente, "pct_madrugada"], hato.loc[~caliente, "pct_madrugada"])}
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Profundiza la exploración de datos: descriptivos, correlaciones, "
                                                 "persistencia, adaptación y PCA.")
    parser.add_argument("--datos", type=Path, default=datos_defecto, help="CSV anonimizado de anonymize_data.py")
    parser.add_argument("--clima", type=Path, default=clima_defecto, help="CSV de descargar_clima.py")
    parser.add_argument("--salida", type=Path, default=salida_defecto, help="Carpeta de salida de las imágenes")
    parser.add_argument("--resumen", type=Path, default=resumen_defecto, help="JSON con los indicadores (fuera de Git)")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    args.resumen.parent.mkdir(parents=True, exist_ok=True)

    leche = cargar_leche(args.datos)
    clima = cargar_clima(args.clima)
    print(f"Sesiones leídas: {len(leche):,} de {leche['animal_id'].nunique()} vacas")

    diccionario = construir_diccionario(leche)
    diccionario.to_csv(args.resumen.parent / "diccionario_variables.csv", index=False)

    graficar_pairplot(leche, args.salida)
    resumen = {
        "descriptivos": describir_variables(leche, args.salida),
        "escalas_temporales": comparar_escalas_temporales(leche, args.salida),
        "eventos_en_el_tiempo": analizar_eventos_en_el_tiempo(leche, args.salida),
        "correlaciones": correlaciones_con_significancia(leche, args.salida, args.resumen.parent),
        "cuartos_sin_leche": analizar_cuartos_sin_leche(leche),
        "vif": calcular_vif(leche),
        "persistencia": analizar_persistencia(leche, args.salida),
        "adaptacion": analizar_adaptacion(leche, args.salida),
        "pca_ordeno": analizar_pca_ordeno(leche, args.salida),
        "pca_clima": analizar_pca_clima(clima, args.salida),
        "sensibilidad_umbral_clima": sensibilidad_umbral_clima(leche, clima),
    }

    args.resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(resumen, indent=1, ensure_ascii=False, default=str))
    print(f"Figuras guardadas en: {args.salida}")


if __name__ == "__main__":
    main()
