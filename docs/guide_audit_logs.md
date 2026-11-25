# Guía de llenado del registro de auditoría de datos

**Archivo que se debe llenar:** [data_audit_logs.csv](data_audit_logs.csv). Este archivo Markdown es únicamente la guía. Los eventos de auditoría se registran en el CSV. Cada fila del CSV representa **un evento real y verificable**.

## Columnas del registro

| Columna | Formato y valores permitidos | Qué registrar |
| --- | --- | --- |
| Fecha | Fecha y hora de 24 horas, DD/MM/YYYY HH/MM, por ejemplo: `11/10/2026 13:53`. | Momento en que ocurrió la acción, no el momento en que se documentó. Si se registra después, aclararlo en Observaciones. |
| Persona | Nombre o identificador interno **(username de github o discord)** de quien realizó la acción. | Evitar datos personales adicionales. |
| dataset_id / versión | Un solo identificador con formato `vacas/x.y`, por ejemplo, `vacas/1.0` o `vacas/2.3`. `x` y `y` son números enteros. | Anotar la versión resultante si la acción publicó una nueva versión; en los demás casos, la versión afectada. |
| Acción | `- Modificación de original`, <br>`- Copia`, <br>`- Integración`, <br>`- Eliminación`, <br>`- Restauración`, <br>`- Exportación o acceso` <br>`- Corrección de registro`. | Elegir la acción que describa el evento. Una corrección de auditoría se registra como evento nuevo, sin reescribir el historial. |
| Propósito | Texto breve y específico. | Explicar por qué se hizo la acción y, si aplica, mencionar la solicitud o autorización que la motivó. |
| Resultado | `Completado`, `Parcial` o `Fallido`. | Indicar qué ocurrió realmente; si fue parcial o falló, detallar el efecto en Observaciones. |
| Observaciones | Texto libre, limitado a información necesaria para reconstruir el evento. | Resumir el cambio, cantidad de registros afectados cuando se conozca, respaldo o comprobación realizada y cualquier incidencia. No copiar aquí los datos modificados. |

## Orden de registro
**Orden de registro:** insertar cada evento nuevo en la **primera fila de datos de `data_audit_logs.csv`, inmediatamente debajo del encabezado**. Los registros anteriores permanecen debajo, así se muestran los registros del más reciente al más antiguo.

Registrar una fila por acción, incluso si el intento fue parcial o fallido. Si se descubre un error en una entrada, añadir una fila de `Corrección de registro` que identifique la entrada afectada y explique la rectificación; conservar la entrada original para mantener la trazabilidad. Para una eliminación, precisar si fue lógica o física y qué respaldo o autorización aplica. Para una restauración, indicar la versión o copia recuperada y la comprobación del resultado.

## Cómo actualizar la versión `vacas/x.y`

- **Versión inicial:** asignar `vacas/1.0` a la primera versión identificada del conjunto. `x` comienza en 1 y `y` en 0.
- **Actualizar `x` (cambio mayor):** aumentar `x` en 1 y reiniciar `y` a 0 cuando cambie la estructura o el significado del dataset de forma que la versión anterior ya no sea directamente comparable. Ejemplos: cambiar columnas o unidades de medida, crear un nuevo dataset, redefinir qué representa cada fila, cambiar los criterios de inclusión o integrar una fuente que altere sustancialmente la interpretación. 
- **Actualizar `y` (cambio menor):** mantener `x` y aumentar `y` en 1 cuando cambie el contenido sin alterar la estructura ni el significado: agregar registros compatibles, corregir valores o retirar duplicados o registros erróneos. De `vacas/1.4` se pasa a `vacas/1.5`.
- **Sin cambio de versión:** representa un acceso o exportación, un intento fallido, una corrección del propio registro de auditoría. Una restauración exacta conserva la versión restaurada.

En cada celda de `dataset_id / versión` escribir **solo una versión**, por ejemplo, `vacas/1.5`.


## Políticas de Uso de Datos del CAETEC
1. Obtener y acceder a los datos
    - Los datos se descargan del Drive proporcionado por los profesores.
    - Crea las carpetas Data/vacas/ y Data/concentrado en la raíz del repositorio.
    - Mueve a la carpeta de Data/vacas/ los archivos Producciones de leche por sesion XXXX.xlsx de TEC-UAQ-UNAM y los archivos .csv de DATOS vacas en establo marzo_junio_2025 cuyo nombre sea un número, los demás puedes ignorarlos.
    - Mueve a la carpeta de Data/concentrado los archivos Concentrado consumido XXXX.xlsx de TEC-UAQ-UNAM
    - Los datos se usan solo para: análisis descriptivo, preparación de datos, modelado de detección de anomalías de ordeño y los entregables académicos y al socio.
2. Proteger el entorno local
    - En caso de que la carpeta de Data y el archivo .env no estén en el gitignore agregarlos.
    - Agregar la llave de cowskey y un valor que servirá para seudonimizar las columnas con datos sensibles.
    - Por cada uno de los scripts ejecutados, se debe agregar a la bitacora la acción que se realizó, la guía para rellenar la bitacora está en docs/guide_audit_logs.md.
    - Ejecutar el script generate_data.py, para generar el dataset con los datos crudos.
    - Ejecutar el script anonymize_data.py, para aplicar un hash y seudonimizar los valores de las columnas sensibles.
3. Construir los datos.
    - Descargar los datos del clima ejecutando el script de descargar_clima.py
    - Limpiar los datos ejecutando el script de clean_data.py
    - Integrar los datos del clima al dataset ejecutando el script de integrar_clima.py
    - Integrar los datos de las variables construidas al dataset ejecutando el script de integrate_derivates.py
    - Obtener los datos de concentrado ejecutando el script de generate_concentrado.py
    - Aplicar una seudonimización a los datos de concentrado ejecutando el script de anonymize_concentrado.py
    - Integrar los datos de concentrado al dataset ejecutando el script de integrar_concentrado.py
4. Trabajar con los datos
    - Antes de trabajar con el dataset se debe ejecutar los pasos anteriores para brindar seguridad a los datos.
    - Consultar los datos originales y el mapa de equivalencias solo cuando sea necesario para verificar la información, corregir un problema, utilizar datos en un reporte o presentación.
    - Mantener de forma local el dataset.
    - Registrar en docs/data_audit_logs.csv los accesos para análisis, entrenamiento y evaluación, así como las copias, modificaciones y exportaciones realizadas, siguiendo el punto 5.
5. Registrar en la bitácora
    - Por cada uno de los scripts ejecutados que se especifican en este archivo, se debe agregar a la bitacora la acción que se realizó, la guía para llenar la bitacora está en docs/guide_audit_logs.md.
    - Cada nueva acción sobre los datos (uso, copia, modificación, integración, eliminación) es una fila en docs/data_audit_logs.csv.
    - La fila nueva se pone en la primera fila de datos, debajo del encabezado, así el registro queda del más reciente al más antiguo.
    - Si hay un error en una fila, se agrega una fila de Corrección de registro que la identifique; la original no se borra.
6. Revisar antes de subir al repositorio
    - Antes de cada commit y push, quien subió el cambio revisa que no incluya datos crudos, anonimizados ni intermedios, el mapa de equivalencias, .env ni llaves. En figuras y reportes solo van resultados agregados y hashes.
    - El revisor, distinto del autor, comprueba que no suban datos ni secretos y que la bitácora esté actualizada. Sin eso, no se aprueba.
7. Publicar y entregar resultados
    - Los entregables académicos y los del socio formador contienen resultados agregados. Cuando se entrega la muestra a los profesores, se registra una fila de Exportación o acceso.
    - Los modelos y resultados se presentan como apoyo a la decisión: no sustituyen el diagnóstico veterinario ni las obligaciones de reportar enfermedades a la autoridad.