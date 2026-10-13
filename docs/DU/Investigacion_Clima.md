# Investigación: fuente de datos del clima del CAETEC

Fase *Data Understanding* (CRISP-DM), punto 3 del reporte de exploración: ¿de dónde obtener el clima para el proyecto? Código: [`src/DU/clima_comparacion.py`](../../src/DU/clima_comparacion.py). Fecha de la investigación: 13/10/2026.

## 1. Pregunta y criterios

El Reporte de Business Understanding exige datos del clima "lo más precisos posible, utilizando la ubicación del rancho", y señala que el modelo usará pronósticos. Las opciones del equipo eran: 1) datos de la NASA (recomendados por el socio formador), 2) alguna API de Google, 3) otra API. Criterios para decidir:

| Criterio | Por qué importa |
|---|---|
| Libre uso y compatible con el proyecto | Proyecto académico sin fines comerciales; los términos deben permitirlo |
| Cobertura de toda la serie (13/07/2022 – 28/01/2026) | La descarga previa solo cubría 28/01/2025 – 27/01/2026: el 56.4 % de las sesiones quedaba sin clima |
| Resolución horaria y espacial | Las sesiones se registran al minuto; la hora del día importa |
| Variables útiles | Temperatura y humedad (para el THI), precipitación y viento |
| Pronóstico | El BU indica que el modelo usará pronósticos del clima |
| Almacenable | Reproducibilidad de los resultados |
| Privacidad de la ubicación | La política de uso de datos del CAETEC pide no divulgar ubicaciones |

## 2. Fuentes comparadas

| | 1) NASA POWER (horario) | 2) Google Maps Platform Weather API | 3) Open-Meteo |
|---|---|---|---|
| Costo y acceso | Gratuito. Sin límite fijo, el uso se monitorea y se limita para un acceso equitativo | De pago: 10,000 eventos gratis al mes y luego desde USD 0.15 por 1,000. Requiere cuenta de Google Cloud con facturación | Gratis para uso **no comercial** (menos de 10,000 llamadas al día); licencia de datos CC BY 4.0 |
| Histórico | Horario desde **2001/01/01** hasta casi tiempo real | Solo las **últimas 24 horas** | ERA5 desde 1940 y ERA5-Land desde 1950, con 5 días de retraso |
| Resolución espacial | Malla MERRA-2 de ½° × ⅝° (≈ 50 × 60 km); cada valor es el promedio de la celda | Puntual (modelo propio) | ERA5-Land ≈ 11 km; ERA5 ≈ 25 km; IFS ≈ 9 km |
| Variables | Temperatura, humedad, viento, presión, precipitación, entre otras | Condiciones actuales, pronóstico horario (240 h) y diario (10 días) | Temperatura, humedad, viento, radiación; la precipitación **no existe en ERA5-Land** |
| Pronóstico | No | Sí | Sí (API de pronóstico y pronósticos archivados desde ≈ 2022) |
| Uso y citación | Pide acreditar "POWER Hourly vX.X.X" con la fecha de acceso y avisar al proyecto de las publicaciones | Atribución obligatoria; caché y almacenamiento restringidos (texto oficial por verificar) | Atribución CC BY 4.0 |

Fuentes: [NASA POWER, API horaria](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/), [metodología de meteorología](https://power.larc.nasa.gov/docs/methodology/meteorology/), [cómo referenciar POWER](https://power.larc.nasa.gov/docs/referencing/), [foro sobre límites de uso](https://forum.earthdata.nasa.gov/viewtopic.php?p=15146), [Google Weather API](https://developers.google.com/maps/documentation/weather), [precios de Google Maps Platform](https://developers.google.com/maps/billing-and-pricing/pricing), [términos de Open-Meteo](https://open-meteo.com/en/terms), [API histórica de Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), [API de pronósticos archivados](https://open-meteo.com/en/docs/historical-forecast-api).

### Por qué se descarta Google para entrenar

La API de Google solo guarda 24 horas de historia, así que no puede cubrir ni una sesión de las 280,699 del histórico. Además es de pago, exige facturación y sus términos limitan el almacenamiento de contenido. Solo tendría sentido como fuente de pronóstico, y para eso hay opciones gratuitas.

## 3. Experimento: NASA POWER contra la descarga actual (ERA5-Land)

Se descargó la serie horaria de NASA POWER (temperatura a 2 m, humedad relativa a 2 m, precipitación corregida y viento a 10 m) para todo el periodo de las sesiones y se comparó con ERA5-Land en las 8,784 horas que ambas fuentes comparten (28/01/2025 – 28/01/2026). Para la consulta se usó el **centro de la celda de la malla de POWER**, no la ubicación exacta del establo.

| Resultado | Valor |
|---|---|
| Horas descargadas de NASA POWER | 31,151 (12/07/2022 – 29/01/2026), sin faltantes en temperatura ni humedad |
| Sesiones de ordeño con clima | **100 %** con NASA POWER frente a **43.6 %** con la descarga de ERA5-Land |
| Correlación horaria de la temperatura (tal como se entrega) | 0.915 |
| Correlación de promedios diarios (temperatura, humedad, THI) | 0.968, 0.947, 0.970 |
| Sesgo de la temperatura (NASA POWER − ERA5-Land) | −0.86 °C |
| Sesgo del THI | −1.37 |
| Horas con THI ≥ 68 | 19.8 % (NASA POWER) frente a 20.8 % (ERA5-Land) |

![Comparación de NASA POWER y ERA5-Land](figuras/clima_comparacion.png)

![Cobertura de clima sobre las sesiones](figuras/clima_cobertura.png)

### Hallazgo: NASA POWER está desfasado una hora

La correlación de la temperatura sube de 0.915 a **0.967** al correr la serie de NASA POWER una hora hacia adelante, y el error cuadrático medio de la temperatura baja de 2.58 a 1.79 °C (el del THI, de 3.33 a 2.53). La explicación probable es que POWER entrega promedios horarios y ERA5-Land valores instantáneos. Por eso el sesgo del THI por hora del día (tercer panel de la figura) cambia de signo entre la mañana y la tarde.

**Decisión de integración:** al unir el clima con las sesiones, la serie de NASA POWER se corre +1 hora (`DESFASE_CLIMA_HORAS` en [`exploracion_datos.py`](../../src/DU/exploracion_datos.py)). El desfase se debe confirmar con una estación observada.

### Otras observaciones

- **Elevación:** la celda de POWER tiene una elevación media de 2,145 m y la de Open-Meteo de 1,922 m; parte del sesgo frío de −0.86 °C se explica porque NASA POWER promedia una celda más alta.
- **Precipitación:** NASA POWER sí entrega precipitación (≈ 590 mm al año en el periodo, coherente con los 500–800 mm que indica INEGI para la región). La descarga previa de Open-Meteo llegó vacía porque ERA5-Land no incluye precipitación; se tendría que pedir a ERA5 o IFS.
- **Huso horario:** NASA POWER se pidió en UTC y se convirtió a `America/Mexico_City`. En 2022 México todavía aplicó horario de verano hasta el 30/10, y la hora repetida de ese cambio se promedia. Falta confirmar si el reloj del sistema de ordeño sigue el horario de verano.

## 4. Recomendación

| Uso | Fuente | Motivo |
|---|---|---|
| **Histórico para entrenar y explorar** | **NASA POWER horario** | Libre, sin clave, cubre el 100 % de las sesiones, entrega precipitación y es la fuente que sugirió el socio. Su correlación diaria con ERA5-Land es de 0.97 |
| **Pronóstico para operar el modelo** | Open-Meteo (API de pronóstico) | NASA POWER no pronostica. Es gratis para uso no comercial y permite combinarla con pronósticos archivados desde ≈ 2022 |
| Descartada | Google Weather API | Sin histórico, de pago, con restricciones de almacenamiento |
| Mediano plazo (a proponer al socio) | Sensor de temperatura y humedad en el establo, o una estación observada cercana | Mide el clima real junto a las vacas y permite validar ambas fuentes |

### Riesgos y cómo mitigarlos

1. **Cambio de fuente entre entrenamiento y despliegue.** Entrenar con NASA POWER y operar con pronósticos de Open-Meteo introduce una diferencia que hay que medir. Mitigación: entrenar con una variable de clima suavizada (promedio diario o THI máximo), que es donde las fuentes más coinciden (r = 0.97), y evaluar el modelo con pronósticos archivados.
2. **Resolución gruesa.** La celda de ≈ 50 × 60 km no distingue el microclima del establo. El THI en la sombra o bajo techo será menor al exterior.
3. **Reanálisis, no medición.** Ninguna de las dos series mide el clima junto a las vacas.
4. **Estabilidad de los datos.** POWER avisa que sus datos pueden reemplazarse por versiones mejoradas, por lo que conviene guardar la respuesta original y la versión del servicio (API v2.10.2 al 13/10/2026).

## 5. Uso, citación y privacidad

- **Reconocimiento de NASA POWER.** En cualquier entregable que use estos datos: *"The data was obtained from the POWER Project's Hourly 2.x.x version on YYYY/MM/DD"*, más la referencia al proyecto POWER del Langley Research Center (texto vigente en la [página de referencias](https://power.larc.nasa.gov/docs/referencing/)). La versión usada fue la v2.10.2 y la fecha de acceso, 13/10/2026. POWER pide además avisar de las publicaciones a `larc-power-project@mail.nasa.gov`.
- **Atribución de Open-Meteo.** Los datos son CC BY 4.0: se debe citar a Open-Meteo y a Copernicus/ERA5-Land como origen.
- **Ubicación.** La política de uso de datos del CAETEC pide no divulgar ubicaciones. Las consultas usaron el centro de la celda de la malla de POWER, y los documentos del repositorio no publican las coordenadas del establo.
- **Datos locales.** Las respuestas originales y la serie procesada se guardan en `Data/clima/`, que está fuera del control de versiones.

## 6. Cómo reproducirlo

```bash
python src/DU/clima_comparacion.py
```

Descarga el clima de NASA POWER (con caché en `Data/clima/`), lo compara con la descarga de ERA5-Land, mide la cobertura sobre las sesiones y regenera las figuras. Requiere `numpy`, `pandas`, `matplotlib` y `requests`.

## 7. Pendientes

- Validar ambas series contra una estación observada (SMN/CONAGUA o el aeropuerto de Querétaro) y confirmar el desfase de una hora.
- Confirmar con el socio formador que "datos de la NASA" se refiere a NASA POWER y si el CAETEC tiene estación meteorológica o sensores propios.
- Verificar el texto vigente de los términos de Google Weather API sobre almacenamiento, y los límites actuales de NASA POWER.
- Decidir si el modelo se desplegará con pronóstico o con clima observado.
