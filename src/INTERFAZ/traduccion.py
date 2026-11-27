"""
Traduce lo técnico de la interfaz al lenguaje del rancho:

1. Identificadores: los hashes de vacas y máquinas vuelven a su nombre real (VMS 1, VMS 2, VMS 3 y el
   número de cada vaca) con Data/mapa_id_vacas.csv, que genera anonymize_data.py. El mapa nunca sale de
   Data/ (está en .gitignore): si no existe, la interfaz sigue mostrando los hashes.
2. Variables: cada variable del XGBoost tiene un nombre corto, qué significa sin tecnicismos y qué
   revisar cuando aparece como factor. Los valores se muestran en sus unidades reales (por ejemplo, el
   MDI sin el logaritmo que le aplica clean_data.py).

Los factores son asociaciones que encontró el modelo, no causas: las acciones son sugerencias de qué
revisar, no un diagnóstico.
"""

import math
import re

import pandas as pd

from nueva_sesion import REFERENCE_MACHINE, ROOT

MAP_PATH = ROOT / "Data" / "mapa_id_vacas.csv"

MDI_ALERT = 1.4  # integrate_derivates.py: MDI >= 1.4 es evento de salud
QUARTERS = {"di": "delantero izquierdo", "dd": "delantero derecho", "ti": "trasero izquierdo", "td": "trasero derecho"}
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre")
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")

# Acciones que comparten varias variables.
CHECK_CONNECTION = ("Revisar que los pezones estén limpios y sin heridas y que la posición de pezones guardada en el "
                    "robot sea correcta. Si se repite, ayudarla a conectar las pezoneras y avisar al técnico.")
CHECK_UDDER = ("Revisar la ubre (calor, hinchazón, grumos o leche rara) y hacer la prueba de mastitis (CMT). "
               "Si sale positiva, avisar al veterinario.")
CHECK_KICKS = ("Revisar si tiene heridas o dolor en pezones, ubre o patas y si hay moscas. Manejarla con calma "
               "y vigilar que las pezoneras no se caigan.")
CHECK_HEAT = "Con calor, encender ventiladores o aspersores y asegurar agua y sombra."
CONTEXT_ONLY = "No requiere acción con la vaca: es información de contexto."
HERD_ONLY = ("Habla de todo el hato, no de esta vaca. Si sube, revisar los robots y la rutina general antes que a "
             "la vaca.")

# variable: (nombre corto, qué significa, qué hacer, tipo de valor)
GLOSSARY = {
    # Datos de la sesión que se conocen al entrar al robot.
    "intervalo_horas": (
        "Horas desde su último ordeño",
        "Cuántas horas pasaron desde que terminó su ordeño anterior. En el hato lo normal es entre 5 y 17 horas.",
        "Si son muchas horas, revisar por qué no vino antes (cojera, enfermedad o celo) y considerar traerla. "
        "Si son muy pocas, revisar si el ordeño anterior quedó incompleto.",
        "horas"),
    "intervalo_horas_sin_dato": (
        "Sin registro de horas desde su último ordeño",
        "El robot no registró cuánto tiempo pasó desde su ordeño anterior, así que se usó un valor estimado.",
        "No requiere acción con la vaca. Si pasa seguido, revisar los registros del robot.",
        "si_no"),
    "regreso_tras_periodo_seco": (
        "Vuelve después de más de 2 días sin ordeño",
        "Pasaron más de 48 horas desde su último ordeño, por ejemplo al volver del periodo seco o de una ausencia.",
        "Vigilar sus primeros ordeños: revisar la ubre y que el robot encuentre bien los pezones.",
        "si_no"),
    "estado_numero_de_ordeno": (
        "Número de ordeño del día",
        "Qué ordeño del día es para esta vaca (1.º, 2.º, 3.º…).",
        CONTEXT_ONLY,
        "entero"),
    "maquina": (
        "Robot de ordeño",
        "En qué robot se ordeña. El modelo encontró que en algunos robots los eventos son más frecuentes.",
        "Si el mismo robot aparece seguido en las alertas, pedir al técnico que lo revise (pezoneras, mangueras, "
        "vacío y cámara).",
        "maquina"),
    "t_sesion": (
        "Temperatura",
        "Temperatura del ambiente a la hora del ordeño (dato del clima de la zona).",
        CHECK_HEAT,
        "temperatura"),
    "thi_sesion": (
        "Índice de calor (THI)",
        "Combina temperatura y humedad para medir el calor que siente la vaca. Desde 68 empieza a sufrir calor.",
        "Si está en 68 o más: " + CHECK_HEAT[0].lower() + CHECK_HEAT[1:],
        "thi"),
    "horas_thi_alto_24h": (
        "Horas de calor en el último día",
        "Cuántas de las últimas 24 horas el índice de calor (THI) estuvo en 68 o más.",
        CHECK_HEAT,
        "horas"),
    "sangre_tasa_maquina_30d": (
        "Sangre en la leche en este robot (30 días)",
        "Porcentaje de ordeños con sangre en la leche en este robot durante los últimos 30 días, de todas las vacas.",
        "Si es alto, pedir al técnico que revise el robot (pezoneras gastadas, vacío o pulsación), porque puede "
        "lastimar los pezones.",
        "porcentaje"),

    # Lo que pasó en el ordeño anterior.
    "fme_connection": (
        "Pezón no encontrado",
        "En el ordeño anterior el robot no pudo encontrar o conectar algún pezón.",
        CHECK_CONNECTION,
        "si_no"),
    "fme_milk_production": (
        "Cuarto sin leche",
        "En el ordeño anterior algún cuarto que sí se ordeña dio 0 kg de leche (la pezonera pudo caerse o no conectar).",
        "Revisar ese cuarto y su pezonera. Si el cuarto ya no da leche, marcarlo como inactivo en el robot.",
        "si_no"),
    "fme_incomplete": (
        "Ordeño incompleto",
        "El ordeño anterior quedó incompleto en algún cuarto.",
        "Revisar si quedó leche en la ubre y si hace falta volver a ordeñarla; revisar pezoneras y patas.",
        "si_no"),
    **{f"blood_diff_{q}": (
        f"Sangre en el cuarto {name}",
        f"Compara la sangre en la leche del cuarto {name} con la de sus otros cuartos en el ordeño anterior.",
        "Revisar ese pezón por heridas o golpes. Si la leche trae sangre, separarla del tanque.",
        "sangre") for q, name in QUARTERS.items()},
    "box_time_to_yield_ratio": (
        "Tiempo en el robot por kilo de leche",
        "Segundos que estuvo en el robot por cada kilo de leche en el ordeño anterior. En el hato lo típico es "
        "unos 31 s/kg; mucho más quiere decir que tardó mucho para poca leche.",
        "Revisar si le costó conectar las pezoneras o si bajó su leche por algún problema de salud.",
        "s_kg"),
    "kicks_ratio_7d": (
        "Patadas en la última semana",
        "Promedio de cuartos con patada por ordeño en los últimos 7 días: 0 es ninguna patada; 1 es, en promedio, "
        "un cuarto pateado en cada ordeño.",
        CHECK_KICKS,
        "patadas"),
    "flies_proxy": (
        "Riesgo por moscas",
        "Índice que sube en las tardes (13 a 17 h) cuando hizo calor y la vaca ha pateado en la semana. Es una "
        "aproximación: no se cuentan las moscas.",
        "Reforzar el control de moscas en las tardes calurosas (trampas o repelente).",
        "decimal"),

    # Historial de la vaca.
    "mecanico_previo": (
        "Falla de conexión en el ordeño anterior",
        "Su ordeño anterior tuvo una falla de conexión: pezón no encontrado, cuarto sin leche u ordeño incompleto.",
        CHECK_CONNECTION,
        "si_no"),
    "salud_previo": (
        "Problema de salud en el ordeño anterior",
        "Su ordeño anterior tuvo sangre en la leche, MDI alto o una alerta de mastitis del robot.",
        CHECK_UDDER,
        "si_no"),
    "conducta_previo": (
        "Patada en el ordeño anterior",
        "En su ordeño anterior pateó al menos un cuarto.",
        CHECK_KICKS,
        "si_no"),
    "mecanico_vaca_30d": (
        "Fallas de conexión de la vaca (30 días)",
        "Porcentaje de sus ordeños de los últimos 30 días con falla de conexión (pezón no encontrado, cuarto sin "
        "leche u ordeño incompleto).",
        "Es una vaca con fallas repetidas: revisar la forma de la ubre y los pezones y la posición guardada en el "
        "robot. Considerar ayudarla a conectar.",
        "porcentaje"),
    "salud_vaca_30d": (
        "Problemas de salud de la vaca (30 días)",
        "Porcentaje de sus ordeños de los últimos 30 días con sangre en la leche, MDI alto o alerta de mastitis.",
        CHECK_UDDER,
        "porcentaje"),
    "conducta_vaca_30d": (
        "Patadas de la vaca (30 días)",
        "Porcentaje de sus ordeños de los últimos 30 días con al menos una patada.",
        CHECK_KICKS,
        "porcentaje"),
    "mdi_previo": (
        "MDI del ordeño anterior",
        f"Índice de mastitis que calcula el robot DeLaval en cada ordeño. Desde {MDI_ALERT} el robot lo marca como "
        "alerta.",
        f"Si es {MDI_ALERT} o más: " + CHECK_UDDER[0].lower() + CHECK_UDDER[1:],
        "mdi"),
    "mdi_faltante_previo": (
        "Sin MDI en el ordeño anterior",
        "El robot no registró el MDI en el ordeño anterior; se estimó con sus ordeños previos.",
        "Revisar si el ordeño anterior fue normal. Si el MDI falta seguido, pedir que revisen el sensor.",
        "si_no"),
    "estado_mdi_sin_dato_previo": (
        "Sin MDI ni historial para estimarlo",
        "El robot no registró el MDI en el ordeño anterior y la vaca no tenía ordeños previos para estimarlo.",
        "Revisar si el ordeño anterior fue normal. Si el MDI falta seguido, pedir que revisen el sensor.",
        "si_no"),
    "conductividad_max_previa": (
        "Conductividad de la leche (ordeño anterior)",
        "La conductividad más alta de sus cuatro cuartos en el ordeño anterior. Sube cuando la ubre está "
        "inflamada (mastitis). En el hato casi siempre está entre 4.4 y 6.2 mS/cm.",
        "Si está alta, hacer la prueba de mastitis (CMT) en el cuarto más alto.",
        "conductividad"),
    "produccion_previa": (
        "Leche del ordeño anterior",
        "Kilos de leche que dio en su ordeño anterior.",
        "Si bajó mucho respecto a lo normal de la vaca, revisar su salud.",
        "kg"),
    "cuarto_inactivo_previo": (
        "Cuarto que no se ordeña",
        "En el ordeño anterior tenía al menos un cuarto marcado como inactivo (el robot no lo ordeña).",
        "Confirmar que ese cuarto esté bien marcado en el robot.",
        "si_no"),
    "dias_en_leche": (
        "Días desde que empezó su ciclo de ordeño",
        "Días desde su primer ordeño después de su última ausencia larga (más de 45 días). Se usa como aproximación "
        "de los días en leche: no es la fecha real de parto.",
        "En los primeros 30 días vigilar más la salud de la ubre.",
        "dias"),
    "dias_en_hato": (
        "Días en los registros del hato",
        "Días desde su primer ordeño en los registros. No es su edad ni su número de partos.",
        CONTEXT_ONLY,
        "dias"),

    # Hato y calendario.
    "mecanico_hato_7d": (
        "Fallas de conexión en el hato (7 días)",
        "Porcentaje de los ordeños de todo el hato con falla de conexión en los últimos 7 días.",
        HERD_ONLY,
        "porcentaje"),
    "salud_hato_7d": (
        "Problemas de salud en el hato (7 días)",
        "Porcentaje de los ordeños de todo el hato con sangre, MDI alto o alerta de mastitis en los últimos 7 días.",
        HERD_ONLY,
        "porcentaje"),
    "conducta_hato_7d": (
        "Patadas en el hato (7 días)",
        "Porcentaje de los ordeños de todo el hato con alguna patada en los últimos 7 días.",
        HERD_ONLY,
        "porcentaje"),
    "vacas_activas_7d": (
        "Vacas ordeñadas en la semana",
        "Cuántas vacas distintas se ordeñaron en los últimos 7 días. Habla de la carga del hato, no de esta vaca.",
        "No requiere acción con la vaca. Si hay muchas vacas por robot, revisar si los robots están saturados.",
        "entero"),
    "mes": (
        "Mes",
        "Mes del ordeño. En algunos meses hay más eventos (por calor, lluvias o moscas).",
        "No requiere acción con la vaca; ayuda a anticipar temporadas con más problemas.",
        "mes"),
    "hora": (
        "Hora del ordeño",
        "Hora del día del ordeño. En algunas horas hay más eventos.",
        "No requiere acción con la vaca; puede señalar horarios o turnos en los que conviene vigilar más.",
        "hora"),
    "dia_semana": (
        "Día de la semana",
        "Día de la semana del ordeño.",
        "No requiere acción con la vaca; puede señalar días en que cambia la rutina o el personal.",
        "dia_semana"),
}
# Ícono de cada variable en static/iconos.svg, para reconocerla antes de leerla.
ICONS = {
    **dict.fromkeys(("intervalo_horas", "intervalo_horas_sin_dato", "hora"), "reloj"),
    **dict.fromkeys(("regreso_tras_periodo_seco", "mes", "dia_semana"), "calendario"),
    "estado_numero_de_ordeno": "numero",
    **dict.fromkeys(("maquina", "sangre_tasa_maquina_30d"), "robot"),
    "t_sesion": "termometro",
    **dict.fromkeys(("thi_sesion", "horas_thi_alto_24h"), "sol"),
    **dict.fromkeys(("fme_connection", "fme_milk_production", "fme_incomplete", "mecanico_previo",
                     "mecanico_vaca_30d"), "pezonera"),
    **dict.fromkeys(("mecanico_hato_7d", "salud_hato_7d", "conducta_hato_7d", "vacas_activas_7d"), "hato"),
    **dict.fromkeys((*(f"blood_diff_{q}" for q in QUARTERS), "cuarto_inactivo_previo"), "ubre"),
    "box_time_to_yield_ratio": "reloj-arena",
    **dict.fromkeys(("kicks_ratio_7d", "conducta_previo", "conducta_vaca_30d"), "patada"),
    "flies_proxy": "mosca",
    **dict.fromkeys(("salud_previo", "salud_vaca_30d", "mdi_previo", "mdi_faltante_previo",
                     "estado_mdi_sin_dato_previo"), "salud"),
    "conductividad_max_previa": "rayo",
    "produccion_previa": "leche",
    **dict.fromkeys(("dias_en_leche", "dias_en_hato"), "vaca"),
}

# Dummies de máquina: en la tabla de variables se juntan en una sola fila.
HIDDEN = re.compile(r"^maquina_.+$")


def load_labels(visits):
    """
    Nombres reales: {"vacas": {hash: número}, "maquinas": {hash o "referencia": nombre}}. La máquina de
    referencia (sin dummy) es la del mapa que no tiene columna. Sin el mapa, los diccionarios van vacíos.
    """
    labels = {"vacas": {}, "maquinas": {}}
    if not MAP_PATH.exists():
        return labels
    table = pd.read_csv(MAP_PATH, dtype=str).dropna(subset=["original", "anonimo"])
    cows = table[table["columna"] == "id_vaca"]
    machines = table[table["columna"] == "misc_eo_po"]
    labels["vacas"] = dict(zip(cows["anonimo"], cows["original"].str.strip()))
    labels["maquinas"] = dict(zip(machines["anonimo"], machines["original"].str.strip()))
    dummies = {c.removeprefix("maquina_") for c in visits if c.startswith("maquina_")}
    reference = [name for code, name in labels["maquinas"].items() if code not in dummies]
    if len(reference) == 1:
        labels["maquinas"][REFERENCE_MACHINE] = reference[0]
    return labels


def glossary():
    """El glosario completo para la pestaña de la página."""
    return [{"variable": name, "nombre": entry[0], "significado": entry[1], "accion": entry[2],
             "icono": ICONS.get(name, "info")} for name, entry in GLOSSARY.items()]


def to_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def machine_name(code, labels):
    return labels["maquinas"].get(code, code)


def readable_value(name, value, labels):
    """El valor de una variable en sus unidades y en palabras."""
    if value is None or value == "":
        return ""
    text = str(value).strip()
    if text == "faltante":
        return "sin dato"
    kind = GLOSSARY.get(name, (None, None, None, "texto"))[3]
    if kind == "maquina":
        return machine_name(text, labels)
    number = to_number(text)
    if number is None:
        return text
    if kind == "si_no":
        return "sí" if number >= 0.5 else "no"
    if kind == "porcentaje":
        return f"{100 * number:.0f} %"
    if kind == "horas":
        return f"{number:.0f} h" if number.is_integer() else f"{number:.1f} h"
    if kind == "dias":
        return f"{number:.0f} días"
    if kind == "temperatura":
        return f"{number:.1f} °C"
    if kind == "kg":
        return f"{number:.1f} kg"
    if kind == "conductividad":
        return f"{number:.2f} mS/cm"
    if kind == "s_kg":
        return f"{number:.0f} s/kg"
    if kind == "mdi":
        mdi = math.expm1(number)  # clean_data.py guarda el MDI con log1p
        return f"{mdi:.2f} · {'en alerta' if mdi >= MDI_ALERT else 'normal'} (alerta desde {MDI_ALERT})"
    if kind == "sangre":
        if number > 0.05:
            return "más sangre que sus otros cuartos"
        return "menos sangre que sus otros cuartos" if number < -0.05 else "igual que sus otros cuartos"
    if kind == "patadas":
        return f"{number:.2f} cuartos pateados por ordeño"
    if kind == "thi":
        return f"{number:.1f}" + (" · con calor (68 o más)" if number >= 68 else " · sin calor (menos de 68)")
    if kind == "hora":
        return f"{int(number):02d}:00"
    if kind == "mes":
        return MONTHS[int(number) - 1] if 1 <= number <= 12 else text
    if kind == "dia_semana":
        return WEEKDAYS[int(number)] if 0 <= number <= 6 else text
    if kind == "entero":
        return f"{number:.0f}"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def describe_factor(factor, labels):
    """Agrega a un factor su nombre, valor legible, qué significa y qué hacer."""
    name = factor["variable"]
    entry = GLOSSARY.get(name)
    return {**factor,
            "nombre": entry[0] if entry else name,
            "valor_legible": readable_value(name, factor.get("valor"), labels),
            "significado": entry[1] if entry else "",
            "accion": entry[2] if entry else "",
            "icono": ICONS.get(name, "info")}


def describe_results(results, labels):
    """Los resultados del modelo por evento, con cada factor traducido."""
    return {event: {**result, "factores": [describe_factor(f, labels) for f in result["factores"]]}
            for event, result in results.items()}


def session_variables(row, columns, machine, labels):
    """Filas de la pestaña de variables: nombre legible, columna y valor en sus unidades."""
    rows = [{"variable": "maquina", "nombre": GLOSSARY["maquina"][0], "valor": machine_name(machine, labels),
             "icono": ICONS["maquina"]}]
    for column in columns:
        if HIDDEN.match(column) or column not in row.index:
            continue
        value = row[column]
        value = "faltante" if pd.isna(value) else value
        rows.append({"variable": column, "nombre": GLOSSARY.get(column, (column,))[0],
                     "valor": readable_value(column, value, labels), "icono": ICONS.get(column, "info")})
    return rows
