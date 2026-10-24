import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_VACAS = RAIZ / "Data" / "dataset_vacas_anonimo.csv"
RUTA_CLIMA = RAIZ / "Data" / "clima" / "clima_nasa_power_por_hora.csv"
SALIDA_VACAS = RAIZ / "Data" / "dataset_vacas_limpio.csv"
SALIDA_CLIMA = RAIZ / "Data" / "clima" / "clima_nasa_power_limpio.csv"


def _a_nombre_columna(texto):
    """'Echada sin ordeñar' -> 'echada_sin_ordenar' (minúsculas, sin acentos ni espacios)."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "_".join(sin_acentos.lower().split())

def _es_ordeno(vacas):
    """Máscara de las visitas que fueron ordeño: la categoría base, con todas las dummies accion_* en 0."""
    return vacas[[c for c in vacas.columns if c.startswith("accion_")]].sum(axis=1) == 0

def _mediana_previa(vacas, columna):
    """Mediana de la vaca en `columna` con sus ordeños anteriores (estrictamente antes de `inicio`), para cada fila.

    Solo usa el pasado de la misma vaca: no mezcla vacas ni mira el futuro, así que no filtra información
    entre entrenamiento y validación, ni cronológica ni por vaca.
    """
    ordenos = vacas[_es_ordeno(vacas)].sort_values("inicio", kind="stable")
    acumulada = ordenos[columna].groupby(ordenos["id_vaca"]).transform(lambda s: s.expanding().median())
    historial = pd.DataFrame({"id_vaca": ordenos["id_vaca"], "inicio": ordenos["inicio"], "valor": acumulada}).dropna(subset=["valor"])
    filas = vacas[["id_vaca", "inicio"]].reset_index().sort_values("inicio", kind="stable")
    cruce = pd.merge_asof(filas, historial, on="inicio", by="id_vaca", direction="backward", allow_exact_matches=False)
    return cruce.set_index("index")["valor"].reindex(vacas.index)


def convertir_fechas_y_duraciones(vacas):
    """Acción 1: convierte hora de inicio, duración e intervalo a datetime y número."""
    vacas = vacas.copy()

    # El archivo junta dos fuentes con formatos distintos:
    #   csv  (2025):      hora "dd/mm/aaaa hh:mm a. m.", duración "mm:ss"
    #   xlsx (2022-2026): hora "aaaa-mm-dd hh:mm:ss",    duración "hh:mm:ss"
    # La fuente se decide una sola vez, por el formato de la hora (el csv trae "a. m." / "p. m.")
    vacas["fuente"] = np.where(vacas["main_hora_de_inicio"].str.contains(r"[ap]\. m\.", regex=True), "csv", "xlsx")
    es_csv = vacas["fuente"] == "csv"
    print(f"[fuente] {es_csv.sum():,} filas del .csv y {(~es_csv).sum():,} del .xlsx")

    vacas["inicio"] = pd.Series(pd.NaT, index=vacas.index, dtype="datetime64[ns]")
    vacas.loc[es_csv, "inicio"] = pd.to_datetime(
        vacas.loc[es_csv, "main_hora_de_inicio"].str.replace("a. m.", "AM", regex=False).str.replace("p. m.", "PM", regex=False),
        format="%d/%m/%Y %I:%M %p")
    vacas.loc[~es_csv, "inicio"] = pd.to_datetime(vacas.loc[~es_csv, "main_hora_de_inicio"], format="%Y-%m-%d %H:%M:%S")
    print(f"[inicio] {es_csv.sum():,} fechas formateadas desde el formato .csv (dd/mm/aaaa hh:mm a. m.)")
    print(f"[inicio] {(~es_csv).sum():,} fechas formateadas desde el formato .xlsx (aaaa-mm-dd hh:mm:ss)")
    print(f"[inicio] rango: {vacas['inicio'].min()} a {vacas['inicio'].max()}")

    # La duración del csv es "mm:ss" y la del xlsx "hh:mm:ss"
    duracion = vacas["main_duracion_mm_ss"].where(~es_csv, "00:" + vacas["main_duracion_mm_ss"])
    vacas["duracion_seg"] = pd.to_timedelta(duracion).dt.total_seconds()
    print(f"[duracion_seg] {es_csv.sum():,} duraciones convertidas desde mm:ss (.csv) y {(~es_csv).sum():,} desde hh:mm:ss (.xlsx)")

    # Excel guarda los intervalos de 24 h o más como fecha ("1900-01-02 06:30:00"), no como "hh:mm:ss".
    # Con época 1899-12-31 la fecha equivale a la duración (se verificó contra la diferencia entre inicios consecutivos).
    intervalo = vacas["main_intervalo_de_ordeno_hh_mm"]
    es_fecha = intervalo.str.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}$").fillna(False)
    es_hms = intervalo.str.match(r"^\d{1,2}:\d{2}:\d{2}$").fillna(False)
    sin_formato = intervalo.notna() & ~es_fecha & ~es_hms
    if sin_formato.any():
        raise ValueError(f"{sin_formato.sum()} intervalos con formato desconocido, ej.: {intervalo[sin_formato].iloc[0]!r}")
    print(f"[intervalo_horas] {es_hms.sum():,} intervalos convertidos desde hh:mm:ss (menos de 24 h)")
    print(f"[intervalo_horas] {es_fecha.sum():,} intervalos convertidos desde fecha de Excel (24 h o más)")
    horas = pd.Series(np.nan, index=vacas.index)
    horas[es_fecha] = (pd.to_datetime(intervalo[es_fecha]) - pd.Timestamp("1899-12-31")).dt.total_seconds() / 3600
    horas[es_hms] = pd.to_timedelta(intervalo[es_hms]).dt.total_seconds() / 3600
    vacas["intervalo_horas"] = horas
    print(f"[intervalo_horas] {horas.isna().sum():,} sin intervalo (el .csv no lo trae)")
    print("[columnas] se eliminan las 3 columnas de texto originales (ya convertidas)")

    return vacas.drop(columns=["main_hora_de_inicio", "main_duracion_mm_ss", "main_intervalo_de_ordeno_hh_mm"])


def quitar_duplicados(vacas):
    """Acción 2: quita las sesiones repetidas entre csv y xlsx y los duplicados exactos."""
    n_inicial = len(vacas)

    # Los registros de 2025 están en las dos fuentes: se quitan del csv los que ya vienen en el xlsx
    # (misma vaca y mismo minuto de inicio)
    vacas = vacas.assign(minuto=vacas["inicio"].dt.floor("min"))
    xlsx = vacas[vacas["fuente"] == "xlsx"].drop_duplicates(["id_vaca", "minuto"])
    cruce = vacas.reset_index().merge(xlsx, on=["id_vaca", "minuto"], suffixes=("", "_xlsx")).set_index("index")
    # El xlsx solo registra ordeños: un rechazo o una echada sin ordeñar del csv nunca es repetido, aunque caiga en el mismo minuto
    cruce = cruce[(cruce["fuente"] == "csv") & (cruce["main_accion"] == "Ordeño")]
    repetido = vacas.index.isin(cruce.index)
    print(f"[duplicados] {repetido.sum():,} ordeños del .csv que ya estaban en el .xlsx (misma vaca y mismo minuto)")

    # El csv no guarda segundos en la hora de inicio, así que se confirma con la duración (que sí los tiene)
    misma_duracion = ((cruce["duracion_seg"] - cruce["duracion_seg_xlsx"]).abs() <= 1).sum()
    print(f"[duplicados] {misma_duracion:,} de {len(cruce):,} coinciden además en duración (±1 s)")
    vacas = vacas[~repetido].drop(columns="minuto")

    exactos = vacas.duplicated()
    print(f"[duplicados] {exactos.sum():,} filas idénticas en todas las columnas")
    vacas = vacas[~exactos]

    print(f"[duplicados] {n_inicial:,} -> {len(vacas):,} filas ({n_inicial - len(vacas):,} eliminadas)")
    print(f"[duplicados] quedan {(vacas['fuente'] == 'csv').sum():,} filas del .csv y {(vacas['fuente'] == 'xlsx').sum():,} del .xlsx")
    return vacas.reset_index(drop=True)


def quitar_columnas_vacias_o_constantes(vacas):
    """Acción 3: elimina las columnas sin datos o con un solo valor, que no aportan al modelo."""
    n_valores = vacas.nunique(dropna=True)
    vacias = n_valores[n_valores == 0].index.tolist()
    constantes = n_valores[n_valores == 1].index.tolist()

    for col in vacias:
        print(f"[columnas] '{col}' vacía: sin ningún dato")
    for col in constantes:
        valor = vacas[col].dropna().iloc[0]
        print(f"[columnas] '{col}' constante: siempre {valor!r} ({vacas[col].notna().sum():,} registros, el resto nulo)")

    print(f"[columnas] {len(vacias) + len(constantes)} columnas eliminadas, quedan {vacas.shape[1] - len(vacias) - len(constantes)}")
    return vacas.drop(columns=vacias + constantes)


def marcar_acciones(vacas, base="Ordeño", min_registros_maquina=100):
    """Acción 4: convierte main_accion en dummies 0/1, una por cada acción distinta de la categoría base
    (la base queda con todas en 0), y deja fuera las máquinas con muy pocos registros."""
    vacas = vacas.copy()

    # El xlsx solo registra ordeños y no trae main_accion: sus nulos son la categoría base
    n_xlsx = vacas["main_accion"].isna().sum()
    vacas["main_accion"] = vacas["main_accion"].fillna(base)
    print(f"[acción] {n_xlsx:,} filas del .xlsx sin main_accion se marcan como '{base}'")
    print(f"[acción] {(vacas['main_accion'] == base).sum():,} visitas '{base}' (categoría base: todas las dummies en 0)")

    # drop_first quitaría la primera en orden alfabético, no necesariamente la base: se quita por nombre
    dummies = pd.get_dummies(vacas["main_accion"], dtype=int).drop(columns=base)
    dummies.columns = ["accion_" + _a_nombre_columna(c) for c in dummies.columns]
    for columna in dummies:
        print(f"[acción] {columna}: {dummies[columna].sum():,} visitas")
    vacas = pd.concat([vacas.drop(columns="main_accion"), dummies], axis=1)

    # Máquinas con muy pocos registros no permiten comparar nada: se dejan fuera
    por_maquina = vacas["misc_eo_po"].value_counts()
    raras = por_maquina[por_maquina < min_registros_maquina].index
    for maquina in raras:
        print(f"[máquina] '{maquina}' solo tiene {por_maquina[maquina]:,} registros: se deja fuera")
    vacas = vacas[~vacas["misc_eo_po"].isin(raras)].reset_index(drop=True)

    print(f"[acción] quedan {len(vacas):,} visitas en {vacas['misc_eo_po'].nunique()} máquinas")
    return vacas


ALERTAS = ["estado_patada", "estado_incompleto", "estado_pezones_no_encontrados", "estado_amd_pezon"]
CUARTOS = ["di", "dd", "ti", "td"]


def alertas_a_banderas(vacas, alertas=ALERTAS, cuartos=CUARTOS):
    """Acción 5: convierte cada alerta de texto ("DI,TD", "Todos") en una bandera 0/1 por cuarto.

    El nulo significa que el evento no ocurrió (0). Las columnas de texto originales se eliminan.
    """
    vacas = vacas.copy()
    nuevas = []
    for alerta in alertas:
        texto = vacas[alerta].fillna("").str.lower()
        n_todos = (texto == "todos").sum()
        banderas = texto.str.replace("todos", ",".join(cuartos), regex=False).str.get_dummies(sep=",")

        desconocidos = set(banderas.columns) - set(cuartos)
        if desconocidos:
            raise ValueError(f"'{alerta}' trae cuartos desconocidos: {desconocidos}")
        banderas = banderas.reindex(columns=cuartos, fill_value=0)
        banderas.columns = [f"{alerta.removeprefix('estado_')}_{cuarto}" for cuarto in cuartos]
        nuevas.append(banderas)

        con_evento = (texto != "").sum()
        por_cuarto = banderas.sum().rename(lambda c: c[-2:]).to_dict()
        print(f"[alertas] {alerta}: {con_evento:,} visitas con evento ({n_todos:,} marcadas 'Todos'), por cuarto {por_cuarto}")

    print(f"[alertas] {len(alertas)} columnas de texto -> {len(alertas) * len(cuartos)} banderas 0/1 (nulo = no ocurrió)")
    return pd.concat([vacas.drop(columns=alertas), *nuevas], axis=1)


def conductividad_a_faltante(vacas, cuartos=CUARTOS):
    """Acción 6: la conductividad 0 es imposible en leche, es una lectura faltante (NaN).

    No se imputa. La razón entre cuartos no se calcula aquí: es colineal con las conductividades
    y se puede derivar después, en la etapa de variables del modelo, si el objetivo la requiere.
    """
    vacas = vacas.copy()
    for cuarto in cuartos:
        columna = f"conductividad_ms_cm_{cuarto}"
        ceros = vacas[columna] == 0
        vacas.loc[ceros, columna] = np.nan
        por_maquina = ceros.groupby(vacas["misc_eo_po"]).mean()
        print(f"[conductividad] {columna}: {ceros.sum():,} ceros -> NaN ({ceros.mean():.1%}); "
              f"máxima por máquina: {por_maquina.idxmax()} con {por_maquina.max():.1%}")
    return vacas


def corregir_intervalos(vacas, umbral_horas=48):
    """Acción 7: completa el intervalo faltante y marca los regresos tras una pausa larga.

    El intervalo es el tiempo entre el fin del ordeño anterior de la vaca y el inicio del actual.
    Los intervalos de más de `umbral_horas` (periodo seco, ausencia) no son el ritmo normal de ordeño:
    pasan a NaN y quedan marcados con la bandera `regreso_tras_periodo_seco`.
    """
    vacas = vacas.copy()
    es_ordeno = _es_ordeno(vacas)

    # Intervalo calculado con el ordeño anterior de la misma vaca (los rechazos no cuentan como ordeño)
    ordenos = vacas[es_ordeno].sort_values(["id_vaca", "inicio"])
    por_vaca = ordenos.groupby("id_vaca")
    fin_previo = por_vaca["inicio"].shift() + pd.to_timedelta(por_vaca["duracion_seg"].shift(), unit="s")
    calculado = ((ordenos["inicio"] - fin_previo).dt.total_seconds() / 3600).reindex(vacas.index)

    # Se comprueba la definición contra los intervalos que ya trae el xlsx
    con_dato = vacas["intervalo_horas"].notna() & calculado.notna()
    distinto = con_dato & ((calculado - vacas["intervalo_horas"]).abs() > 0.01)
    print(f"[intervalo] el cálculo coincide con {1 - distinto.sum() / con_dato.sum():.1%} de los {con_dato.sum():,} intervalos que ya existían (±36 s)")
    print(f"[intervalo] {distinto.sum():,} no coinciden: el robot reporta ~0 h tras una ausencia larga "
          f"(el cálculo da de {calculado[distinto].min():,.0f} a {calculado[distinto].max():,.0f} h, mediana {calculado[distinto].median():,.0f} h)")

    # Se usa siempre el intervalo calculado cuando existe: completa los faltantes del csv y corrige esos ceros
    faltante = vacas["intervalo_horas"].isna() & es_ordeno
    usar = es_ordeno & calculado.notna()
    vacas.loc[usar, "intervalo_horas"] = calculado[usar]
    sin_previo = (vacas["intervalo_horas"].isna() & es_ordeno).sum()
    print(f"[intervalo] {(faltante & usar).sum():,} intervalos completados con el ordeño anterior")
    print(f"[intervalo] {sin_previo:,} ordeños del .csv sin ordeño anterior en los datos: quedan NaN")
    print(f"[intervalo] {(~es_ordeno).sum():,} visitas sin ordeño: no tienen intervalo")

    larga = vacas["intervalo_horas"] > umbral_horas
    vacas["regreso_tras_periodo_seco"] = larga.astype(int)
    print(f"[intervalo] {larga.sum():,} intervalos de más de {umbral_horas} h ({vacas.loc[larga, 'id_vaca'].nunique()} vacas): "
          f"van de {vacas.loc[larga, 'intervalo_horas'].min():.0f} h a {vacas.loc[larga, 'intervalo_horas'].max():,.0f} h -> NaN y bandera")
    vacas.loc[larga, "intervalo_horas"] = np.nan
    return vacas

def tratar_nulos(vacas, cuartos=CUARTOS):
    """Acción 8: trata los nulos sin usar información del futuro ni de otras vacas.

    Solo se usa la misma visita o el pasado de la misma vaca; lo que no se puede resolver así queda en NaN para el pipeline
    de cada modelo, que lo rellena con estadísticos de entrenamiento (ver rellenar_con_cero_y_bandera).
    """
    vacas = vacas.copy()
    conductividad = [f"conductividad_ms_cm_{cuarto}" for cuarto in cuartos]
    contadores = [c for c in vacas.columns if c.startswith("contadores_monitorizacion")]
    es_ordeno = _es_ordeno(vacas)

    # 1. Cuartos con producción nula: produjeron 0 kg (el total menos la suma de los otros cuartos da 0 en todos los casos).
    #    Los cuartos sin leche ya vienen con 0 en producción, flujos y sangre; aquí se unifica esa convención.
    produccion = vacas[[f"producciones_kg_{cuarto}" for cuarto in cuartos]]
    verificado = es_ordeno & ((vacas["main_produccion_kg"] - produccion.sum(axis=1)).abs() <= 0.05)
    print(f"[nulos] {(produccion.isna().any(axis=1) & es_ordeno).sum():,} ordeños con algún cuarto nulo: "
          f"el total menos los otros cuartos da 0 kg en {(produccion.isna().any(axis=1) & verificado).sum():,}")
    for cuarto in cuartos:
        a_cero = vacas[f"producciones_kg_{cuarto}"].isna() & verificado
        vacas.loc[a_cero, [f"{base}_{cuarto}" for base in ("producciones_kg", "media_de_los_flujos_kg_min", "flujos_maximos_kg_min", "sangre_ppm")]] = 0.0
        print(f"[nulos] cuarto {cuarto.upper()}: {a_cero.sum():,} cuartos nulos pasan a 0 en producción, flujos y sangre")

    # 2. Conductividad: promedio de los otros cuartos de la misma visita; si ninguno tiene lectura, mediana previa de la vaca.
    #    `conductividad_estimada_<cuarto>` marca los valores estimados.
    original = vacas[conductividad]
    validos = original.notna()
    suma = original.sum(axis=1)
    n_validos = validos.sum(axis=1)
    base = original.assign(id_vaca=vacas["id_vaca"], inicio=vacas["inicio"]).join(vacas.filter(like="accion_"))
    for columna, cuarto in zip(conductividad, cuartos):
        faltante = original[columna].isna() & es_ordeno
        vacas[f"conductividad_estimada_{cuarto}"] = faltante.astype(int)
        otros = (suma - original[columna].fillna(0)) / (n_validos - validos[columna]).replace(0, np.nan)
        vacas.loc[faltante, columna] = otros[faltante]
        sin_cuartos = faltante & otros.isna()
        vacas.loc[sin_cuartos, columna] = _mediana_previa(base, columna)[sin_cuartos]
        print(f"[nulos] {columna}: {faltante.sum():,} nulos -> {(faltante & otros.notna()).sum():,} con los otros cuartos de la visita, "
              f"{(sin_cuartos & vacas[columna].notna()).sum():,} con la mediana previa de la vaca, {(faltante & vacas[columna].isna()).sum():,} sin historial")

    # 3. MDI: bandera y mediana previa de la vaca. Los contadores no existen en el csv (se resuelven en el paso 5).
    faltante = vacas["estado_mdi"].isna() & es_ordeno
    vacas["mdi_faltante"] = faltante.astype(int)
    vacas.loc[faltante, "estado_mdi"] = _mediana_previa(vacas, "estado_mdi")[faltante]
    print(f"[nulos] estado_mdi: {faltante.sum():,} faltantes -> {(faltante & vacas['estado_mdi'].notna()).sum():,} con la mediana previa de la vaca")

    # 4. Se quitan los ordeños que siguen sin dato por falta de historial. MDI y contadores solo cuentan en filas del xlsx:
    #    en el csv esas columnas no existen (estructural).
    solo_xlsx = ["estado_mdi"] + contadores
    nulos = vacas[conductividad + solo_xlsx].isna()
    nulos[solo_xlsx] = nulos[solo_xlsx].mul(vacas["fuente"] == "xlsx", axis=0)
    sin_historial = es_ordeno & nulos.any(axis=1)
    print(f"[nulos] {sin_historial.sum():,} ordeños sin historial previo se quitan ({vacas.loc[sin_historial, 'id_vaca'].nunique()} vacas): "
          f"{ {c: int(n) for c, n in nulos[sin_historial].sum().items() if n} }")
    vacas = vacas[~sin_historial].reset_index(drop=True)
    es_ordeno = _es_ordeno(vacas)

    # 5. Visitas sin ordeño: lo que no se ordeñó vale 0 y el destino de la leche es 'Sin ordeño'.
    #    El nulo de misc_razon_de_la_desviacion significa que no hubo desvío.
    ceros = ["main_produccion_kg", "estado_numero_de_ordeno", "estado_amd_ubre"] + [
        f"{base}_{cuarto}" for base in ("producciones_kg", "media_de_los_flujos_kg_min", "flujos_maximos_kg_min", "sangre_ppm") for cuarto in cuartos]
    vacas.loc[~es_ordeno, ceros] = vacas.loc[~es_ordeno, ceros].fillna(0)
    vacas.loc[~es_ordeno, "misc_destino_leche"] = vacas.loc[~es_ordeno, "misc_destino_leche"].fillna("Sin ordeño")
    vacas["misc_razon_de_la_desviacion"] = vacas["misc_razon_de_la_desviacion"].fillna("Sin desvío")
    print(f"[nulos] {(~es_ordeno).sum():,} visitas sin ordeño: producción, flujos y sangre = 0; destino 'Sin ordeño'; desvío nulo = 'Sin desvío'")

    # 6. Lo que falta en las visitas sin ordeño (y el intervalo): mediana previa de la vaca; sin historial queda en NaN.
    for columna, bandera in [(c, f"conductividad_estimada_{q}") for c, q in zip(conductividad, cuartos)] + [("estado_mdi", "mdi_faltante"), ("intervalo_horas", None)]:
        faltante = vacas[columna].isna()
        if bandera:
            vacas[bandera] = (vacas[bandera].astype(bool) | faltante).astype(int)
        propia = _mediana_previa(vacas, columna)
        vacas.loc[faltante, columna] = propia[faltante]
        print(f"[nulos] {columna}: {faltante.sum():,} nulos -> {(faltante & propia.notna()).sum():,} con el historial de la vaca, {vacas[columna].isna().sum():,} sin historial quedan en NaN")

    # 7. Contadores (no existen en el csv): 0 con bandera. Es una constante fija, no un estadístico de los datos.
    vacas["contadores_faltante"] = vacas[contadores].isna().any(axis=1).astype(int)
    vacas[contadores] = vacas[contadores].fillna(0)
    print(f"[nulos] contadores: {vacas['contadores_faltante'].sum():,} visitas sin contadores pasan a 0, bandera 'contadores_faltante'")
    return vacas


def marcar_inconsistencias(vacas, tolerancia_kg=0.05):
    """Acción 9: conserva y marca los ordeños con 0 kg y los que tienen la suma de cuartos distinta del total."""
    vacas = vacas.copy()
    es_ordeno = _es_ordeno(vacas)
    suma = vacas.filter(regex=r"^producciones_kg_").sum(axis=1)
    vacas["ordeno_cero_kg"] = (es_ordeno & (vacas["main_produccion_kg"] == 0)).astype(int)
    vacas["suma_cuartos_distinta"] = (es_ordeno & ((suma - vacas["main_produccion_kg"]).abs() > tolerancia_kg)).astype(int)
    print(f"[inconsistencias] {vacas['ordeno_cero_kg'].sum():,} ordeños con 0 kg y {vacas['suma_cuartos_distinta'].sum():,} con suma de cuartos distinta: marcados, no se quitan")
    return vacas


def marcar_sangre(vacas, umbral_saturacion=7900, umbral_pico=0.15):
    """Acción 15: marca la sangre por máquina sin cambiar su escala (los eventos de salud usan los ppm originales).

    El robot satura la sangre en 8,000 ppm y hay trimestres donde una máquina marca sangre en 14 a 28 % de sus ordeños (contra ~5 %).
    `sangre_saturada`: algún cuarto cerca del tope del sensor. `sangre_tasa_maquina_30d`: tasa de visitas con sangre de la máquina en los
    30 días anteriores (solo el pasado). `sangre_pico_maquina`: esa tasa supera `umbral_pico` (constante fija, no ajustada con los datos).
    """
    vacas = vacas.copy()
    sangre = vacas.filter(regex=r"^sangre_ppm_")
    vacas["sangre_saturada"] = (sangre.max(axis=1) >= umbral_saturacion).astype(int)
    con_sangre = (sangre.max(axis=1) > 0).astype(float)
    tasa = pd.Series(0.0, index=vacas.index)
    for _, filas in vacas.sort_values("inicio").groupby("misc_eo_po"):
        tasa[filas.index] = pd.Series(con_sangre[filas.index].to_numpy(), index=filas["inicio"]).rolling("30D", closed="left").mean().fillna(0).to_numpy()
    vacas["sangre_tasa_maquina_30d"] = tasa
    vacas["sangre_pico_maquina"] = (tasa > umbral_pico).astype(int)
    print(f"[sangre] {vacas['sangre_saturada'].sum():,} visitas cerca del tope del sensor (>= {umbral_saturacion} ppm); "
          f"{vacas['sangre_pico_maquina'].sum():,} visitas en picos de la máquina (tasa a 30 días > {umbral_pico:.0%})")
    return vacas


def preparar_historial(vacas, min_registros=30, dias_reinicio=45):
    """Acciones 11 y 12: deja el dataset listo para calcular historial por vaca (los rezagos los crea integrate_derivates).

    Excluye las vacas con menos de `min_registros` registros, ordena por vaca e inicio y agrega `bloque_historial`: cambia tras una
    ausencia de `dias_reinicio` días o más, para reiniciar ahí los rezagos y promedios móviles.
    """
    pocos = vacas.groupby("id_vaca")["id_vaca"].transform("size") < min_registros
    print(f"[historial] {vacas.loc[pocos, 'id_vaca'].nunique()} vacas con menos de {min_registros} registros se excluyen ({pocos.sum():,} visitas)")
    vacas = vacas[~pocos].sort_values(["id_vaca", "inicio"], kind="stable").reset_index(drop=True)
    ausencia = vacas.groupby("id_vaca")["inicio"].diff() >= pd.Timedelta(days=dias_reinicio)
    vacas["bloque_historial"] = ausencia.groupby(vacas["id_vaca"]).cumsum()
    print(f"[historial] {vacas.groupby(['id_vaca', 'bloque_historial']).ngroups:,} bloques de historial (se reinicia tras ausencias de {dias_reinicio}+ días)")
    return vacas


def rellenar_con_cero_y_bandera(vacas):
    """Cierre: los nulos que quedan (vacas sin historial) pasan a 0 con una bandera `<variable>_sin_dato`.

    Es una constante fija, no un estadístico de los datos, así que no filtra información.
    """
    vacas = vacas.copy()
    for columna in vacas.columns[vacas.isna().any()]:
        bandera = re.sub(r"_(di|dd|ti|td)$", "", columna) + "_sin_dato"
        vacas[bandera] = (vacas[bandera] if bandera in vacas else 0) | vacas[columna].isna().astype(int)
        vacas[columna] = vacas[columna].fillna(0)
    print(f"[cierre] nulos que quedan: {int(vacas.isna().sum().sum()):,}; banderas de relleno: {[c for c in vacas.columns if c.endswith('_sin_dato')]}")
    return vacas


def categoricas_a_dummies(vacas, bases=None):
    """Convierte las categóricas en dummies 0/1 (una categoría base por columna queda con todas en 0).

    Los desvíos `Divert 1/2/3` se agrupan en `Divert`: Divert 1 y 2 tienen 7 visitas en total. `id_vaca` e `inicio` no se tocan:
    son el identificador y el tiempo, no variables.
    """
    bases = bases or {"misc_eo_po": "7063e4b1", "misc_destino_leche": "Tanque", "misc_razon_de_la_desviacion": "Sin desvío", "fuente": "xlsx"}
    vacas = vacas.copy()
    vacas["misc_destino_leche"] = vacas["misc_destino_leche"].str.replace(r"^Divert \d$", "Divert", regex=True)
    prefijo = {"misc_eo_po": "maquina", "misc_destino_leche": "destino", "misc_razon_de_la_desviacion": "desvio", "fuente": "fuente"}
    for columna, base in bases.items():
        dummies = pd.get_dummies(vacas[columna], dtype=int).drop(columns=base)
        dummies.columns = [f"{prefijo[columna]}_{_a_nombre_columna(c)}" for c in dummies.columns]
        vacas = pd.concat([vacas.drop(columns=columna), dummies], axis=1)
        print(f"[dummies] {columna} (base '{base}') -> {dict(dummies.sum())}")
    return vacas


def limpiar_clima(clima, max_horas_faltantes=3):
    """Limpieza del clima de NASA POWER: orden, duplicados, rejilla horaria completa, códigos de faltante y rangos físicos.

    No aplica el desfase de una hora, el THI ni los rezagos: eso lo hace integrar_clima.py al integrarlo con las visitas.
    """
    clima = clima.drop_duplicates("hora").sort_values("hora").set_index("hora")
    horas = pd.date_range(clima.index.min(), clima.index.max(), freq="h")
    clima = clima.reindex(horas).rename_axis("hora")
    clima = clima.mask(clima <= -900)                                   # NASA usa -999 para dato faltante
    clima["rh2m"] = clima["rh2m"].clip(0, 100)
    clima["cloud_amt"] = clima["cloud_amt"].clip(0, 100)
    faltantes = clima.isna().sum().sum()
    clima = clima.interpolate(limit=max_horas_faltantes, limit_area="inside")
    print(f"[clima] {len(clima):,} horas ({clima.index.min():%Y-%m-%d} a {clima.index.max():%Y-%m-%d}); {faltantes:,} valores faltantes, "
          f"{clima.isna().sum().sum():,} tras interpolar huecos de hasta {max_horas_faltantes} h; "
          f"rocío > temperatura en {(clima['t2mdew'] > clima['t2m'] + 0.01).sum():,} horas (se conserva, es del reanálisis)")
    return clima.reset_index()


if __name__ == "__main__":
    vacas = pd.read_csv(RUTA_VACAS, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
    clima = pd.read_csv(RUTA_CLIMA, parse_dates=["hora"])

    vacas = convertir_fechas_y_duraciones(vacas)
    vacas = quitar_duplicados(vacas)
    vacas = quitar_columnas_vacias_o_constantes(vacas)
    vacas = marcar_acciones(vacas)
    vacas = alertas_a_banderas(vacas)
    vacas = conductividad_a_faltante(vacas)
    vacas = corregir_intervalos(vacas)
    vacas = tratar_nulos(vacas)
    vacas = marcar_inconsistencias(vacas)
    vacas = marcar_sangre(vacas)
    vacas = preparar_historial(vacas)
    vacas = rellenar_con_cero_y_bandera(vacas)
    vacas = categoricas_a_dummies(vacas)
    clima = limpiar_clima(clima)

    vacas.to_csv(SALIDA_VACAS, index=False)
    clima.to_csv(SALIDA_CLIMA, index=False)
    print(f"[guardado] {len(vacas):,} visitas x {vacas.shape[1]} columnas -> {SALIDA_VACAS.relative_to(RAIZ)}")
    print(f"[guardado] {len(clima):,} horas x {clima.shape[1]} columnas -> {SALIDA_CLIMA.relative_to(RAIZ)}")
