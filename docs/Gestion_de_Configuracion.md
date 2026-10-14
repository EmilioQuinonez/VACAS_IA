# Gestión de la Configuración

## 1. Propósito y alcance

Establecer los lineamientos para identificar, controlar, versionar y proteger los elementos de configuración del proyecto, con el fin de asegurar la integridad, la trazabilidad y la consistencia de los productos de trabajo (código y documentación).

Aplica a todos los miembros del equipo. Cualquier elemento creado o modificado a partir de la fecha de implementación de este documento debe cumplir con lo aquí definido.

## 2. Terminología

| Término | Definición |
|---|---|
| **CI** (Elemento de Configuración) | Producto de trabajo cuya alteración puede afectar la calidad, la trazabilidad, la entrega o el cumplimiento del proyecto, y que por ello se identifica y se controla. |
| **Línea base** | Versión aprobada y congelada de un conjunto de CIs, que sirve de referencia y solo puede modificarse mediante un cambio controlado. |
| **Repositorio** | Sistema oficial donde reside un CI. Para este proyecto, GitHub. |
| **Release / Tag** | Marca inmutable en GitHub que identifica una línea base. |

## 3. Política

1. Todo producto de trabajo relevante debe identificarse como CI cuando su alteración pueda afectar la calidad, la trazabilidad, la entrega o el cumplimiento del proyecto.
2. Los CIs se clasifican por tipo: código, documento, dependencias/configuración, datos de referencia.
3. Todo cambio a un CI debe hacerse mediante el flujo de control de cambios (sección 7). No se aceptan modificaciones directas a `main` ni a `develop`.
4. Los CIs se mantienen bajo control de versiones en Git. El historial de commits y de Pull Requests es el registro de quién cambió qué y cuándo.
5. Las únicas líneas base del proyecto son las ramas `main` y `develop`. Un release/tag se crea directamente sobre una de ellas (sección 6).
6. Nunca se versionan secretos ni credenciales (`.env`, llaves `*.pem`, tokens). Las variables de entorno necesarias se documentan en `.env.example`, sin valores reales; los valores reales viven solo en el `.env` local de cada integrante.
7. Nunca se versionan los datos de la granja (crudos, anonimizados ni intermedios), ni el archivo de equivalencias entre identificadores originales y anónimos; los datos se mantienen fuera del repositorio, de forma local. Los datos se controlan con su propio versionado y bitácora (sección 5.5).

## 4. Identificación de CIs

| Categoría | CI | Ubicación |
|---|---|---|
| Código fuente | Scripts de preparación, anonimización y análisis de datos | `src/` (por ejemplo `src/DU/generate_data.py`, `src/DU/anonymize_data.py`) |
| Documentación de entendimiento del negocio | Reporte de Business Understanding, One Page, Memorándum de entendimiento, WBS | `docs/BU/` |
| Documentación de entendimiento de los datos | Figuras y documentos de descripción de datos | `docs/DU/` |
| Control de datos | Bitácora de auditoría de datos y su guía de llenado | `docs/data_audit_logs.csv`, `docs/guide_audit_logs.md` |
| Dependencias y entorno | Archivo de requerimientos y variables de entorno de ejemplo | `requirements.txt`, `.env.example` |
| Configuración del repositorio | Plantilla de PR y reglas de archivos ignorados | `.github/pull_request_template.md`, `.gitignore` |
| Documentación general | README, licencia | `README.md`, `LICENSE` |
| Gestión de la configuración | Este documento | `docs/Gestion_de_Configuracion.md` |

No se consideran CIs: archivos temporales del sistema (`.DS_Store`), entornos virtuales (`venv/`), cachés y salidas intermedias reproducibles.

## 5. Convenciones de nomenclatura

### 5.1 Ramas

| Rama | Uso | Reglas |
|---|---|---|
| `main` | Versión estable y entregas al socio formador. | Protegida. Línea base de las entregas. Solo recibe merges desde `develop` mediante PR al cerrar una entrega. Nunca commit directo. |
| `develop` | Integración del trabajo en curso. | Protegida. Recibe PRs de ramas de trabajo. Nunca commit directo. |
| `feature/<descripcion>` | Nueva funcionalidad, análisis o documento. | Se crea desde `develop`. |
| `fix/<descripcion>` | Corrección de un error. | Se crea desde `develop` (o desde `main` si es urgente, y luego se integra a `develop`). |
| `docs/<descripcion>` | Cambios únicamente de documentación. | Se crea desde `develop`. |

Reglas de nombre:

- Minúsculas, palabras separadas por guion bajo (`_`), sin tildes ni espacios.
- Descriptivo y corto; debe indicar qué se hace, no quién lo hace.
- Una rama por cambio; se elimina después del merge.

Ejemplos: `feature/gestion_configuracion`, `feature/descripcion_datos`, `fix/path_carpeta_raiz`, `docs/reporte_bu`.

### 5.2 Commits

Formato:

```
<tipo>: <descripción breve en español>
```

| Tipo | Cuándo usarlo |
|---|---|
| `feature` | Se agrega una funcionalidad, un script o un documento nuevo. |
| `fix` | Se corrige un error. |
| `update` | Se modifica algo existente sin ser corrección (renombrar, ajustar, mejorar). |
| `docs` | Cambios menores solo de documentación. |

Reglas:

- Un commit = un cambio lógico. No mezclar temas distintos.
- Commits y PRs deben estar en el mismo idioma (español).
- Prohibido `--force`, `reset --hard` o `--no-verify` sobre ramas compartidas sin acuerdo explícito del equipo.
- Nunca incluir secretos, credenciales ni datos sensibles.

Ejemplos válidos:

```
feature: agrega archivo de requerimientos
fix: corrige el path de la carpeta raiz
update: cambiar el nombre del archivo eda
fix: borrar la carpeta de Data por seguridad
```

### 5.3 Pull Requests

- Todo cambio llega a `develop` (o a `main` desde `develop` al cerrar una entrega) mediante un Pull Request. No se hace push directo.
- **Título:** mismo formato que un commit (`feature: ...`, `fix: ...`).
- **Plantilla:** todo PR debe usar la plantilla del repositorio (`.github/pull_request_template.md`), que GitHub carga automáticamente, y completar todas sus secciones:
  - **Resumen:** el problema y la solución propuesta.
  - **Tipo de cambio:** documentación (`docs/`), código fuente (`src/`) u otro.
  - **Cambios realizados:** lista de los cambios incluidos.
  - **Lista de verificación:** se completa la parte que corresponde al tipo de cambio.
    - Documentación: la información es correcta, clara y actualizada; se revisaron ortografía, enlaces y rutas; los diagramas, imágenes o adjuntos funcionan; se actualizó la bitácora de auditoría de datos.
    - Código: sigue las convenciones del proyecto; se probó localmente; se agregaron o actualizaron pruebas cuando corresponde; se actualizó la documentación (reportes de entrega) cuando corresponde; se actualizó la bitácora de auditoría de datos.
    - **Bitácora de auditoría de datos:** la plantilla pide actualizarla en ambos tipos de cambio. Se registra en `docs/data_audit_logs.csv` (sección 5.5) cuando el PR realiza o implica una acción sobre los datos. Si el PR no actúa sobre los datos, el punto se deja sin marcar y se indica "No aplica" en las notas.
  - **Evidencia:** capturas, resultados de pruebas o enlaces relevantes, si aplica.
  - **Notas para revisión:** decisiones, limitaciones o aspectos que requieran atención especial.
- **Revisor:** una persona distinta del autor. Al aprobar y hacer el merge asume la responsabilidad de que el cambio cumple los estándares, incluidos los de este documento (nomenclatura de ramas y commits, y que no se incluyan secretos, datos crudos ni archivos del sistema).
- **Resultado:** el revisor aprueba, o rechaza con *Request changes* indicando qué corregir. Un PR no se aprueba con la lista de verificación incompleta.
- **Merge:** se hace una vez aprobado.

### 5.4 Versionado y tags

Se usa versionado semántico `vMAJOR.MINOR.PATCH`:

| Componente | Se incrementa cuando… |
|---|---|
| `MAJOR` | Hay un cambio de entrega mayor (por ejemplo, un nuevo modelo o fase completa del proyecto). |
| `MINOR` | Se agrega funcionalidad o documentación nueva compatible. |
| `PATCH` | Se hacen correcciones menores. |

Ejemplos: `v0.1.0` (primera línea base), `v0.2.0` (nueva funcionalidad o documento aprobado), `v1.0.0` (entrega al socio formador).

Los documentos de `docs/` no llevan tabla de control de versiones ni número de versión propios; su trazabilidad la dan el historial de Git y los Pull Requests.

Los tags `vX.Y.Z` versionan el repositorio (código y documentación). No deben confundirse con la versión de los datos (`vacas/x.y`, sección 5.5), que es independiente.

### 5.5 Versionado de datos y bitácora de auditoría

Los datos no se guardan en Git, pero su control sí forma parte de la configuración del proyecto.

- **Versión del dataset:** se identifica con `vacas/x.y` (por ejemplo `vacas/1.0`). `x` aumenta (y `y` vuelve a 0) cuando cambia la estructura o el significado del dataset; `y` aumenta cuando cambia el contenido sin alterar estructura ni significado. Las reglas completas están en `docs/guide_audit_logs.md`.
- **Bitácora:** toda acción sobre los datos se registra en `docs/data_audit_logs.csv`, una fila por acción, siguiendo la guía de llenado: modificación de original, copia, integración, eliminación, restauración, exportación o acceso, y corrección de registro. Los registros nuevos van en la primera fila de datos, debajo del encabezado.
- **Responsabilidad:** quien realiza la acción es quien la registra. El registro se actualiza en el mismo PR que contiene el código o la documentación asociada.
- **Uso de los datos:** se aplican las políticas de uso de datos del CAETEC descritas en `docs/guide_audit_logs.md`.

## 6. Líneas base y releases

### 6.1 Líneas base

Las únicas líneas base del proyecto son dos ramas protegidas, que solo cambian mediante PR aprobado:

| Línea base | Qué representa |
|---|---|
| `main` | Versión estable y aprobada; lo que se entrega al socio formador. |
| `develop` | Estado integrado y aprobado del trabajo en curso. |

No hay ramas `release/*`: el release se crea directamente sobre una de estas dos ramas.

### 6.2 Desde qué rama sacar un release

| Situación | Rama | Tag de ejemplo |
|---|---|---|
| Entrega al socio formador o cierre de una fase. | `main` | `v1.0.0` |
| Hito interno o entregable aprobado que no se entrega todavía. | `develop` | `v0.2.0` |

Si el release es una entrega, primero se integra `develop` a `main` mediante PR y el tag se crea sobre `main`. Para todo lo demás basta con taguear `develop`.

### 6.3 Cómo crear un release en GitHub

1. Verificar que la rama elegida (`main` o `develop`) esté al día y que todos los cambios incluidos hayan sido aprobados mediante PR.
2. En GitHub: **Releases → Draft a new release**.
3. En **Target**, elegir la rama (`main` o `develop`) y crear el tag `vX.Y.Z` según la sección 5.4.
4. Documentar en la descripción los cambios principales y los PRs incluidos.
5. Publicar el release.

Un release solo puede crearse sobre cambios aprobados: no se hace release de algo que está en revisión.

## 7. Control de cambios

Flujo estándar:

```
rama desde develop → commits → Pull Request → revisión → merge
```

1. **Sincronizar:** antes de empezar, `git fetch` y verificar que la rama local esté al día con su upstream y con `develop`.
2. **Crear la rama** con la nomenclatura de la sección 5.1.
3. **Trabajar** con commits pequeños y con el formato de la sección 5.2.
4. **Abrir el PR** hacia `develop` con descripción y referencia al requisito o issue.
5. **Revisión** por una persona distinta del autor, con la checklist de la sección 5.3.
6. **Merge** una vez aprobado; la rama de trabajo se elimina.

Trazabilidad: cualquier cambio en un requisito o en el alcance debe reflejarse en la documentación correspondiente de `docs/` en el mismo PR.

## 8. Rollback (retrocesos)

Cuando un CI o un release genere un problema funcional, documental o de calidad:

- El retroceso se hace únicamente hacia la **última versión aprobada y liberada** anterior, registrada como release o tag.
- Deben cumplirse los criterios: existe una versión anterior aprobada y el rollback queda documentado como un nuevo cambio controlado.
- No se regresa a un commit arbitrario sin trazabilidad ni aprobación formal.

Procedimiento:

1. Identificar el tag o release de destino.
2. Crear una rama `fix/rollback_<descripcion>` desde `develop`.
3. Revertir con `git revert` (conserva el historial) en lugar de reescribirlo.
4. Abrir un PR explicando el motivo y la versión de destino, y seguir el flujo normal de revisión.

Está prohibido reescribir el historial compartido (`reset --hard`, `push --force`) para realizar un rollback.