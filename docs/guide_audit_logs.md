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
- **Actualizar `x` (cambio mayor):** aumentar `x` en 1 y reiniciar `y` a 0 cuando cambie la estructura o el significado del dataset de forma que la versión anterior ya no sea directamente comparable. Ejemplos: cambiar columnas o unidades de medida, redefinir qué representa cada fila, cambiar los criterios de inclusión o integrar una fuente que altere sustancialmente la interpretación. 
- **Actualizar `y` (cambio menor):** mantener `x` y aumentar `y` en 1 cuando cambie el contenido sin alterar la estructura ni el significado: agregar registros compatibles, corregir valores o retirar duplicados o registros erróneos. De `vacas/1.4` se pasa a `vacas/1.5`.
- **Sin cambio de versión:** representa un acceso o exportación, un intento fallido, una corrección del propio registro de auditoría. Una restauración exacta conserva la versión restaurada.

En cada celda de `dataset_id / versión` escribir **solo una versión**, por ejemplo, `vacas/1.5`.


## Políticas de Uso de Datos del CAETEC
1. Los datos del CAETEC se usan únicamente para los fines autorizados del proyecto: análisis descriptivo, preparación de datos, modelado predictivo de detección de anomalías de ordeño, los entregables académicos y al socio formador.

2. El equipo reconoce que no es titular de los datos. Obtenerlos no otorga derecho de publicarlos, cederlos ni reutilizarlos fuera de lo autorizado.

3. Antes de cualquier uso adicional (nuevo proyecto, publicación, tesis, concurso, entrega a terceros) se debe obtener autorización por escrito del titular.

4. Los datos que se compartieron del CAETEC nunca se versionan en Git ni se publican; permanecen fuera del repositorio, se mantienen de forma local.

5. Se aplica mínimo privilegio: solo accede quien lo necesita para el proyecto, y solo a los datos necesarios (miembros del equipo, profesores y socio formadores).

6. Si se detecta que un archivo contiene datos personales (nombres de operadores, empleados o propietarios, imágenes de personas), se avisa al responsable de datos y se anonimiza o retira antes de continuar. Si llegara a tratarse información de personas, se aplican los principios de la ley federal de protección de datos personales en posesión de particulares.

7. Los datos se tratan como información confidencial del CAETEC; no se divulgan datos, ubicaciones, claves del establo ni parámetros operativos que permitan identificar la unidad de producción sin autorización.

8. Todo uso, copia y modificación de los datos debe quedar trazable (quién, cuándo, con qué versión, con qué código y para qué) en el archivo logs.csv.

    - Modificación de original,
    - Copia,
    - Integración,
    - Eliminación,
    - Restauración,
    - Exportación o acceso
    - Corrección de registro.

9. Los modelos y resultados son apoyo a la decisión: 
no sustituyen el diagnóstico veterinario ni las obligaciones de reportar enfermedades a la autoridad.

10. Se trabaja con los datos directos pero a la hora de publicar descubrimientos y resultados, no se menciona la fuente de los datos ni se comparten los mismos.

11. Ante alguna filtración de información, se para el análisis, se detecta que ocasionó la fuga para corregir, se elimina el dato publicado y se avisa al socio sobre la situación. 