"""Agrega cuatro atributos históricos de concentrado a dataset_vacas_integrado.csv.

Requiere el concentrado anonimizado con la misma COWSKEY que el ordeño.
Las fechas originales y todas las columnas existentes se conservan.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from .integrar_clima import hora_de_inicio
except ImportError:
    from integrar_clima import hora_de_inicio


RAIZ = Path(__file__).resolve().parents[2]
ORIGEN = RAIZ / "Data" / "dataset_vacas_integrado.csv"
CONCENTRADO = RAIZ / "Data" / "dataset_concentrado_anonimo.csv"
ATRIBUTOS = ["conc_sesion_previa", "tope_sesion_previa", "conc_d1", "conc_media_7d"]
TOPE_KG = 2.5


def agregar_concentrado(sesiones, consumo):
    """Construye atributos disponibles antes de la hora de inicio de cada sesión."""
    faltantes = {"id_vaca", "main_hora_de_inicio", "main_produccion_kg"}.difference(
        sesiones.columns
    )
    faltantes_consumo = {"id_vaca", "hora_evento", "consumido"}.difference(
        consumo.columns
    )
    if faltantes or faltantes_consumo:
        raise ValueError(
            f"Faltan columnas: sesiones={sorted(faltantes)}, concentrado={sorted(faltantes_consumo)}"
        )
    repetidas = set(ATRIBUTOS).intersection(sesiones.columns)
    if repetidas and repetidas != set(ATRIBUTOS):
        raise ValueError(
            f"Solo algunos atributos están integrados: {sorted(repetidas)}"
        )
    if repetidas:
        sesiones = sesiones.drop(columns=ATRIBUTOS)

    base = sesiones.reset_index(drop=True)
    inicio = hora_de_inicio(base["main_hora_de_inicio"])
    if base["id_vaca"].isna().any() or inicio.isna().any():
        raise ValueError("Hay sesiones sin ID o con fecha de inicio inválida")
    eventos = consumo.copy().drop_duplicates()
    eventos["hora_evento"] = pd.to_datetime(
        eventos["hora_evento"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    eventos["consumido"] = pd.to_numeric(eventos["consumido"], errors="raise")
    if eventos[["id_vaca", "hora_evento", "consumido"]].isna().any().any():
        raise ValueError("Hay eventos de concentrado incompletos")
    if eventos["consumido"].lt(0).any():
        raise ValueError("Hay consumos negativos")
    if eventos["hora_evento"].dt.floor("h").ne(eventos["hora_evento"]).any():
        raise ValueError("Hay registros de concentrado fuera de la hora en punto")

    # Se da prioridad a la sesión de la misma hora. Si no existe, el registro
    # puede estar redondeado a la hora posterior al inicio. En cada caso se
    # exige una sola sesión candidata para no repartir un consumo ambiguo.
    horas_sesion = inicio.dt.floor("h")
    # El dataset conserva sesiones repetidas entre CSV y XLSX. Si coinciden
    # vaca, minuto y producción, representan una sola visita para el emparejamiento.
    canonica = pd.Series(base.index, index=base.index)
    equivalencias = pd.DataFrame(
        {
            "vaca": base["id_vaca"],
            "minuto": inicio.dt.floor("min"),
            "produccion": pd.to_numeric(base["main_produccion_kg"], errors="coerce"),
            "csv": base["main_hora_de_inicio"].str.len().eq(22),
        }
    )
    claves = ["vaca", "minuto", "produccion"]
    repetidas = equivalencias[equivalencias.duplicated(claves, keep=False)]
    for _, grupo in repetidas.groupby(claves, dropna=False):
        if len(grupo) > 1 and grupo["csv"].any() and not grupo["csv"].all():
            canonica.loc[grupo.index] = grupo.index[0]
    miembros = defaultdict(list)
    for fila, representante in canonica.items():
        miembros[representante].append(fila)
    por_hora = defaultdict(list)
    for fila, (vaca, hora) in enumerate(zip(base["id_vaca"], horas_sesion)):
        representante = canonica.loc[fila]
        if representante not in por_hora[(vaca, hora)]:
            por_hora[(vaca, hora)].append(representante)
    vinculados = defaultdict(list)
    ambiguos = 0
    sin_sesion = 0
    posteriores = []
    for vaca, hora, kg in eventos[["id_vaca", "hora_evento", "consumido"]].itertuples(
        index=False, name=None
    ):
        candidatos = por_hora[(vaca, hora)]
        if len(candidatos) == 1:
            vinculados[candidatos[0]].append((hora, kg))
        elif candidatos:
            ambiguos += 1
        else:
            posteriores.append((vaca, hora, kg))

    # Un registro de la hora siguiente solo se asigna cuando esa sesión no
    # recibió ya un registro de su propia hora.
    for vaca, hora, kg in posteriores:
        candidatos = por_hora[(vaca, hora - pd.Timedelta(hours=1))]
        if len(candidatos) == 1 and not vinculados[candidatos[0]]:
            vinculados[candidatos[0]].append((hora, kg))
        elif candidatos:
            ambiguos += 1
        else:
            sin_sesion += 1

    # Dos eventos distintos ligados a una misma sesión tampoco permiten saber
    # qué cantidad corresponde a esa visita.
    kg_sesion = pd.Series(float("nan"), index=base.index)
    hora_consumo = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns]")
    for fila, registros in vinculados.items():
        if len(registros) == 1:
            for miembro in miembros[fila]:
                hora_consumo.loc[miembro], kg_sesion.loc[miembro] = registros[0]
        else:
            ambiguos += len(registros)

    # Las dos filas de una misma visita CSV/XLSX comparten antecedente. Para
    # visitas distintas con inicio idéntico se exige un inicio previo estricto.
    orden = pd.DataFrame(
        {
            "vaca": base["id_vaca"],
            "inicio": inicio,
            "fila": base.index,
            "representante": canonica,
        }
    )
    visitas = (
        orden.groupby("representante", sort=False)
        .agg(vaca=("vaca", "first"), inicio=("inicio", "max"))
        .reset_index()
        .sort_values(["vaca", "inicio"], kind="stable")
    )
    tiempos = visitas.drop_duplicates(["vaca", "inicio"], keep="last").copy()
    tiempos["fila_previa"] = tiempos.groupby("vaca", sort=False)["representante"].shift(
        1
    )
    visitas = visitas.merge(
        tiempos[["vaca", "inicio", "fila_previa"]],
        on=["vaca", "inicio"],
        how="left",
        sort=False,
    )
    previa = orden.merge(
        visitas[["representante", "fila_previa"]],
        on="representante",
        how="left",
        sort=False,
    )
    fila_previa = previa.sort_values("fila")["fila_previa"].reset_index(drop=True)
    previo_kg = fila_previa.map(kg_sesion)
    previo_hora = pd.to_datetime(fila_previa.map(hora_consumo))
    # El registro de consumo debe ser anterior incluso a la hora redondeada
    # de la sesión actual; un registro en esa hora puede ser consumo actual.
    disponible = previo_hora.lt(horas_sesion)
    previo_kg = previo_kg.where(disponible)

    # Reindexar al calendario hace que los días sin registro sigan siendo NaN.
    diarios = (
        eventos.groupby(["id_vaca", eventos["hora_evento"].dt.normalize()])["consumido"]
        .sum()
        .rename_axis(index=["id_vaca", "fecha"])
        .reset_index()
    )
    fechas = inicio.dt.normalize()
    d1 = pd.Series(float("nan"), index=base.index)
    media_7d = pd.Series(float("nan"), index=base.index)
    for vaca, grupo in diarios.groupby("id_vaca", sort=False):
        filas = base.index[base["id_vaca"].eq(vaca)]
        if len(filas) == 0:
            continue
        inicio_cal = min(grupo["fecha"].min(), fechas.loc[filas].min())
        fin_cal = max(grupo["fecha"].max(), fechas.loc[filas].max())
        calendario = pd.date_range(inicio_cal, fin_cal, freq="D")
        serie = grupo.set_index("fecha")["consumido"].reindex(calendario)
        d1.loc[filas] = serie.shift(1).reindex(fechas.loc[filas]).to_numpy()
        media_7d.loc[filas] = (
            serie.rolling(7, min_periods=7)
            .mean()
            .shift(1)
            .reindex(fechas.loc[filas])
            .to_numpy()
        )

    resultado = sesiones.copy()
    resultado["conc_sesion_previa"] = previo_kg.to_numpy()
    resultado["tope_sesion_previa"] = (
        previo_kg.ge(TOPE_KG).astype(float).where(previo_kg.notna()).to_numpy()
    )
    resultado["conc_d1"] = d1.to_numpy()
    resultado["conc_media_7d"] = media_7d.to_numpy()
    estadisticas = {
        "duplicados": len(consumo) - len(eventos),
        "eventos_ambiguos": ambiguos,
        "eventos_sin_sesion": sin_sesion,
        "eventos_vinculados": sum(len(v) == 1 for v in vinculados.values()),
        "filas_duplicadas_entre_fuentes": sum(len(v) - 1 for v in miembros.values()),
    }
    return resultado, estadisticas


def main():
    sesiones = pd.read_csv(
        ORIGEN, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False
    )
    consumo = pd.read_csv(CONCENTRADO, dtype={"id_vaca": str}, low_memory=False)
    integrado, estadisticas = agregar_concentrado(sesiones, consumo)
    integrado.to_csv(ORIGEN, index=False)
    print(
        f"Sesiones: {len(integrado):,} | vacas: {integrado.id_vaca.nunique()} | columnas: {integrado.shape[1]}"
    )
    print(
        f"Concentrado: {len(consumo):,} eventos | duplicados eliminados: {estadisticas['duplicados']:,}"
    )
    print(
        f"Vínculos únicos: {estadisticas['eventos_vinculados']:,} | ambiguos: {estadisticas['eventos_ambiguos']:,} | sin sesión: {estadisticas['eventos_sin_sesion']:,}"
    )
    print(
        f"Filas equivalentes CSV/XLSX conservadas: {estadisticas['filas_duplicadas_entre_fuentes']:,}"
    )
    for atributo in ATRIBUTOS:
        print(f"{atributo}: {integrado[atributo].notna().sum():,} sesiones con valor")
    print(f"Guardado: {ORIGEN.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
