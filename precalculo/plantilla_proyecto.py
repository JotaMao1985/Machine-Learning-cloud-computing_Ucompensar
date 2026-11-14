"""Notebook plantilla del proyecto final (M9): «Del dato crudo a la decisión».

    python precalculo/plantilla_proyecto.py            # escribe notebooks/plantilla-proyecto.ipynb (sin salidas)
    python precalculo/plantilla_proyecto.py ejecuta    # además lo ejecuta en la réplica de Colab (comprobación)

El ejemplo resuelto usa una base abierta REAL: los accidentes de tránsito de Bucaramanga
(datos.gov.co, conjunto 7cci-nqqb, Alcaldía de Bucaramanga, CC BY-SA 4.0). Esa base es solo
del ejemplo: no se puede elegir para el proyecto. El notebook no cita cifras en el texto:
las ve cada quien al ejecutarlo. Fuentes: F-ML-11, F-ML-12, F-ML-18, F-ML-24 a F-ML-27,
F-DATOS-01, F-DATOS-02 (fuentes/FUENTES.md).
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "plantilla-proyecto.ipynb"
PROYECTO = "https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/proyecto.html"

CELDAS = [
("md", f"""# Proyecto final · Plantilla: del dato crudo a la decisión
**Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**

Este cuaderno (*notebook*) es el esqueleto del proyecto final, con **un ejemplo resuelto** sobre una base abierta real.
El enunciado completo está en [la página del proyecto]({PROYECTO}).

**Cómo usarlo**
1. **Archivo → Guardar una copia en Drive**.
2. Ejecútalo una vez completo (**Entorno de ejecución → Ejecutar todas**) para ver el ejemplo funcionando.
3. Después, **cambia la base y la pregunta por las tuyas** en las celdas marcadas con ✏️, y rehaz cada paso con tus
   decisiones. Las celdas marcadas con 🔁 suelen servir casi igual para cualquier base.
4. Borra lo que sea solo del ejemplo antes de entregar.

**El ejemplo:** los accidentes de tránsito de Bucaramanga, publicados por la Alcaldía en el Portal de Datos Abiertos
(licencia CC BY-SA 4.0). **Esta base es solo del ejemplo: no la puedes elegir para tu proyecto.**

En el código, los decimales van con punto (así los escribe Python); en el texto, con coma."""),

("md", """## 1 · La pregunta y la decisión ✏️
Antes de tocar los datos, escribe en tres líneas:
- **Quién decide qué** con el modelo.
- **La etiqueta**: qué se predice, y si es una categoría (clasificación) o un número (regresión).
- **Qué cuesta equivocarse** en cada sentido.

**En el ejemplo.** Cuando entra el reporte de un accidente (dónde, a qué hora, qué vehículos), la central de Tránsito
quiere saber si **probablemente hay personas heridas o muertas**, para despachar primero una ambulancia. La etiqueta es
`lesionados` (1 si hubo heridos o muertos, 0 si solo daños): **clasificación**. Dejar sin ambulancia un accidente con
heridos (falso negativo) es mucho más grave que enviarla a uno con solo daños (falso positivo). El modelo **sugiere**;
la persona de la central decide."""),

("md", """## 2 · Los datos: de dónde salen ✏️
En la ficha anotas la entidad, la URL exacta, la fecha de descarga y la licencia. La API del portal (Socrata)
**devuelve solo 1000 filas si no le pides más**: agrega `$limit=` a la URL. Las columnas de códigos (como los del DANE)
se leen como texto (`dtype=str`) para no perder el cero inicial."""),

("code", """# ✏️ Cambia estas tres líneas por las de tu base
URL = "https://www.datos.gov.co/resource/7cci-nqqb.csv?$limit=100000"
FUENTE = "Alcaldía de Bucaramanga, «Accidentes de tránsito ocurridos en el municipio de Bucaramanga», Portal de Datos Abiertos www.datos.gov.co"
LICENCIA = "CC BY-SA 4.0"

import datetime
import pandas as pd

crudo = pd.read_csv(URL, dtype=str)   # todo como texto: se convierte después, a propósito
crudo.to_csv("datos_crudos.csv", index=False)   # una copia, por si el portal falla después
print(f"Fuente: {FUENTE}. Licencia: {LICENCIA}.")
print(f"Descargado el {datetime.date.today():%Y-%m-%d}: {crudo.shape[0]} filas y {crudo.shape[1]} columnas")"""),

("md", """## 3 · La auditoría: qué tiene la base, antes de tocar nada 🔁
Cuatro preguntas: ¿qué columnas hay?, ¿cuántos faltantes hay, **incluidos los disfrazados** (códigos como
«NO DISPONIBLE», «SIN INFORMACIÓN», «-1» o «99999»)?, ¿hay categorías escritas de varias formas?, ¿hay filas
duplicadas? Lo que encuentres aquí es la primera tabla de tu informe."""),

("code", """# 🔁 Faltantes reales y disfrazados, por columna
CODIGOS_FALTANTE = ["NO DISPONIBLE", "SIN INFORMACION", "SIN INFORMACIÓN", "NO REGISTRA", "NO APLICA", "ND", "-1", "99999"]

auditoria = pd.DataFrame({
    "vacios_pct": crudo.isna().mean() * 100,
    "codigo_faltante_pct": crudo.apply(lambda col: col.str.strip().str.upper().isin(CODIGOS_FALTANTE).mean() * 100),
    "valores_distintos": crudo.nunique(),
    "ejemplo": crudo.iloc[0],
})
print("Filas duplicadas:", crudo.duplicated().sum())
auditoria.sort_values("codigo_faltante_pct", ascending=False).round(1)"""),

("code", """# 🔁 ¿Categorías escritas de varias formas? Compara cuántos valores hay antes y después de normalizar
import unicodedata

def normalizar(texto):
    \"\"\"Mayúsculas, sin tildes y sin espacios sobrantes: «Con Muertos » y «con muertos» quedan iguales.\"\"\"
    if pd.isna(texto):
        return texto
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.upper().split())

texto = crudo.select_dtypes("object")
resumen = pd.DataFrame({"antes": texto.nunique(), "despues": texto.apply(lambda c: c.map(normalizar)).nunique()})
resumen[resumen["antes"] > resumen["despues"]]   # las columnas con grafías distintas para lo mismo"""),

("code", """# En el ejemplo: la etiqueta tiene dos grafías para lo mismo, y la hora trae una fecha falsa (1899-12-31)
print(crudo["gravedad"].value_counts())
print(crudo["hora"].head(3).tolist())"""),

("md", """## 4 · La limpieza, fila por fila ✏️
Aquí van las correcciones que **no aprenden nada de otras filas**: unificar grafías, pasar los códigos de faltante a
`NaN`, convertir fechas y números, construir la etiqueta y quitar columnas que no sirven. Como cada fila se corrige
sola, se pueden hacer **antes de partir** en entrenamiento y prueba. Lo que aprende de los datos (imputar con la
mediana, escalar, codificar) va después, **dentro del `Pipeline`** (sección 6).

**Cuidado con la fuga de datos.** Quita las columnas que no se conocerían al momento de predecir. En el ejemplo, la
central no sabe la gravedad antes de que alguien llegue al lugar: solo usamos lo que trae el reporte."""),

("code", """import numpy as np

datos = crudo.copy()

# Unificar grafías en las columnas de texto
for columna in ["gravedad", "barrio", "nombrecomuna", "entidad", "propietario_de_veh_culo", "diurnio_nocturno"]:
    datos[columna] = datos[columna].map(normalizar)

# Los códigos de faltante pasan a NaN, para que se vean y se puedan imputar
datos = datos.replace({codigo: np.nan for codigo in CODIGOS_FALTANTE})

# Fechas y números
datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce")
datos["hora"] = pd.to_datetime(datos["hora"], errors="coerce").dt.hour + pd.to_datetime(datos["hora"], errors="coerce").dt.minute / 60
VEHICULOS = ["peaton", "automovil", "campero", "camioneta", "micro", "buseta", "bus", "camion", "volqueta", "moto", "bicicleta", "otro"]
datos[VEHICULOS] = datos[VEHICULOS].apply(pd.to_numeric, errors="coerce")

# La etiqueta: 1 si hubo heridos o muertos
datos["lesionados"] = (datos["gravedad"] != "SOLO DANOS").astype(int)

print(datos["gravedad"].value_counts())
print("Faltantes después de limpiar (%):")
print((datos.isna().mean() * 100).round(1).loc[lambda s: s > 0])"""),

("md", """## 5 · Las características creadas ✏️
Una **característica creada** es una columna nueva calculada a partir de otras. Estas se calculan fila por fila (no
usan información de otras filas), así que también van antes de partir. Cada una lleva su porqué: ¿por qué ayudaría a
la decisión?"""),

("code", """datos["dia_semana"] = datos["fecha"].dt.dayofweek               # 0 = lunes
datos["fin_de_semana"] = (datos["dia_semana"] >= 5).astype(int)   # el tráfico cambia el fin de semana
datos["mes"] = datos["fecha"].dt.month
datos["n_vehiculos"] = datos[VEHICULOS].drop(columns="peaton").sum(axis=1)   # más vehículos, choque más complejo
datos["hay_vulnerable"] = ((datos["peaton"] + datos["moto"] + datos["bicicleta"]) > 0).astype(int)   # peatón, moto o bicicleta

# Las columnas que usa el modelo: solo lo que trae el reporte. Se quitan el identificador, la fecha cruda,
# el año (no sirve para el futuro) y la etiqueta original.
NUMERICAS = ["hora", "dia_semana", "mes", "n_vehiculos"] + VEHICULOS
CATEGORICAS = ["barrio", "nombrecomuna", "entidad", "propietario_de_veh_culo", "diurnio_nocturno"]
BINARIAS = ["fin_de_semana", "hay_vulnerable"]
X = datos[NUMERICAS + CATEGORICAS + BINARIAS]
y = datos["lesionados"]
print(X.shape, "| proporción con lesionados:", round(y.mean(), 3))"""),

("md", """## 6 · Partir, y preparar dentro de un `Pipeline` 🔁
Primero se parte: el 25 % queda **guardado para el final** y no se mira hasta la sección 9. Después, un
`ColumnTransformer` aplica a cada tipo de columna lo suyo, y todo queda dentro de un `Pipeline` que **aprende solo del
entrenamiento** (sesión 2; *Common pitfalls* de scikit-learn):
- **numéricas:** imputar con la **mediana** (no la mueven los atípicos) y escalar. `add_indicator=True` agrega una
  columna que dice dónde faltaba el dato, por si faltar ya dice algo;
- **categóricas:** imputar con la categoría **más frecuente** y codificar con `OneHotEncoder`.
  `handle_unknown="ignore"` deja en ceros una categoría que no se vio al entrenar, y `min_frequency=30` junta las
  categorías raras en una sola: así un barrio con 2 accidentes no se vuelve una columna.

Si tus datos tienen orden en el tiempo (semanas, años), considera dejar los últimos periodos como prueba, en lugar de
partir al azar: así se mide como se usaría el modelo."""),

("code", """from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.25, random_state=2026, stratify=y)

preparar = ColumnTransformer([
    ("numericas", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERICAS),
    ("categoricas", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore", min_frequency=30)), CATEGORICAS),
    ("binarias", "passthrough", BINARIAS),
])
print(len(X_ent), "para entrenar;", len(X_prueba), "guardados para el final")"""),

("md", """## 7 · La línea base y tres modelos, con validación cruzada 🔁
Cada modelo es un `Pipeline` completo (preparar + modelo), así que la validación cruzada repite **también** la
imputación y la codificación en cada pliegue: no hay fuga. Se miden la sensibilidad (a cuántos lesionados encuentra),
la precisión (cuántas ambulancias enviadas eran necesarias) y el F1, con su promedio y su desviación (sesión 2).
Tres algoritmos **distintos** de los que vimos; la línea base es `DummyClassifier` (o `DummyRegressor` si haces
regresión)."""),

("code", """from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score

# La precisión de la línea base no se puede calcular si nunca dice «sí»: zero_division=0 la deja en 0 sin avisos
METRICAS = {"recall": "recall", "precision": make_scorer(precision_score, zero_division=0), "f1": "f1"}

modelos = {
    "Línea base (la clase más frecuente)": DummyClassifier(strategy="most_frequent"),
    "Regresión logística": LogisticRegression(max_iter=2000),
    "Bosque aleatorio": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=2026, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=2026),
}
pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

filas = []
for nombre, modelo in modelos.items():
    r = cross_validate(make_pipeline(preparar, modelo), X_ent, y_ent, cv=pliegues,
                       scoring=METRICAS)
    filas.append({"modelo": nombre,
                  **{f"{m}_promedio": r[f"test_{m}"].mean() for m in ["recall", "precision", "f1"]},
                  **{f"{m}_desv": r[f"test_{m}"].std() for m in ["recall", "precision", "f1"]}})
comparacion = pd.DataFrame(filas).set_index("modelo").round(3)
comparacion"""),

("md", """**Lee la tabla antes de seguir ✏️.** ¿La diferencia entre el primero y el segundo es mayor que su desviación? Si no,
no hay un ganador claro y puedes elegir por otra razón (más simple, más fácil de explicar, más rápido). La línea base
con la clase más frecuente tiene sensibilidad 0 si esa clase es «sin lesionados»: no encuentra a nadie
(sesión 1). Escribe aquí qué modelo eliges y por qué."""),

("md", """## 8 · El umbral, elegido sin mirar la prueba 🔁
Con 0,5 el modelo dice «lesionados» solo si está bastante seguro. Para la central, un falso negativo es grave: conviene
**bajar el umbral** para encontrar más, aunque se envíen más ambulancias (sesión 2). El umbral se elige con
probabilidades de validación cruzada **sobre el entrenamiento** (`cross_val_predict`), no con la prueba."""),

("code", """from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score

# ✏️ En el ejemplo, los tres modelos empatan dentro de su desviación: se elige el más simple de explicar
elegido = make_pipeline(preparar, LogisticRegression(max_iter=2000))
prob_vc = cross_val_predict(elegido, X_ent, y_ent, cv=pliegues, method="predict_proba")[:, 1]

umbrales = pd.DataFrame([{"umbral": u,
                          "sensibilidad": recall_score(y_ent, prob_vc >= u),
                          "precision": precision_score(y_ent, prob_vc >= u, zero_division=0),
                          "alertas_pct": (prob_vc >= u).mean() * 100}
                         for u in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]).round(3)
umbrales"""),

("code", """# ✏️ La regla de decisión del ejemplo: encontrar al menos 8 de cada 10 accidentes con lesionados.
# Se toma el umbral más alto que lo logra (el que menos ambulancias envía de más).
META_SENSIBILIDAD = 0.80
cumplen = umbrales[umbrales["sensibilidad"] >= META_SENSIBILIDAD]
UMBRAL = cumplen["umbral"].max() if len(cumplen) else 0.2
print("Umbral elegido:", UMBRAL)"""),

("md", """## 9 · La prueba, una sola vez 🔁
Ahora sí: se entrena con todo el entrenamiento y se mide **una vez** en los datos guardados. Si después cambias algo
porque no te gustó esta cifra, la prueba deja de ser «datos que el modelo no vio»."""),

("code", """from sklearn.metrics import confusion_matrix, classification_report

elegido.fit(X_ent, y_ent)
prob_prueba = elegido.predict_proba(X_prueba)[:, 1]
pred = (prob_prueba >= UMBRAL).astype(int)
print(pd.DataFrame(confusion_matrix(y_prueba, pred),
                   index=["real: sin lesionados", "real: con lesionados"],
                   columns=["dice: sin lesionados", "dice: con lesionados"]))
print(classification_report(y_prueba, pred, target_names=["sin lesionados", "con lesionados"], digits=3))"""),

("md", """## 10 · La interpretación: qué pesa en el modelo 🔁
La **importancia por permutación** desordena una columna a la vez en los datos de prueba y mide cuánto empeora el
modelo: si empeora mucho, el modelo dependía de esa columna. Se calcula sobre el `Pipeline` completo, con las columnas
originales. Ojo: si dos columnas dicen casi lo mismo (como `hora` y `diurnio_nocturno`), cada una aparece menos
importante de lo que es, porque la otra la reemplaza. Pregúntate si lo que pesa **tiene sentido** en tu contexto."""),

("code", """from sklearn.inspection import permutation_importance

imp = permutation_importance(elegido, X_prueba, y_prueba, scoring="recall", n_repeats=10, random_state=2026, n_jobs=-1)
importancia = (pd.DataFrame({"columna": X_prueba.columns, "caida_promedio": imp.importances_mean, "desv": imp.importances_std})
               .sort_values("caida_promedio", ascending=False).round(3))
importancia.head(10)"""),

("md", """## 11 · Para el Destacado: ¿se equivoca más con algún grupo? (opcional) 🔁
Compara la sensibilidad por grupo. En el ejemplo, por jornada: si de noche el modelo encuentra menos lesionados, la
central debería saberlo. En tu proyecto, el grupo puede ser un municipio, un tipo de colegio o un rango de edad.
Describe grupos; no juzgues personas."""),

("code", """por_grupo = (pd.DataFrame({"grupo": X_prueba["diurnio_nocturno"], "real": y_prueba, "pred": pred})
             .groupby("grupo")
             .apply(lambda g: pd.Series({"casos": len(g), "sensibilidad": recall_score(g["real"], g["pred"])}), include_groups=False)
             .round(3))
por_grupo"""),

("md", """## 12 · Lo que cambias para tu proyecto ✏️
- [ ] La pregunta, la decisión y la etiqueta (sección 1), y la ficha del problema del domingo.
- [ ] `URL`, `FUENTE` y `LICENCIA` (sección 2). Si son datos propios, lee el archivo anonimizado y adjúntalo.
- [ ] Los códigos de faltante de tu base, las grafías que unificas y las columnas que quitas por fuga (secciones 3 y 4).
- [ ] Al menos una característica creada, con su porqué (sección 5).
- [ ] Tres algoritmos distintos; si haces regresión, `DummyRegressor`, regresores y métricas en las unidades de la
      etiqueta (MAE, RMSE), sin umbral (secciones 7 y 8).
- [ ] La regla del umbral, si clasificas, justificada con la decisión (sección 8).
- [ ] La interpretación, contrastada con tu contexto (sección 10).
- [ ] No dejes filas individuales de personas impresas en el notebook.

**Para la ficha de datos y gobierno** (una página): el origen y la licencia; qué columnas son datos personales
(también las que, combinadas, podrían identificar a alguien); dónde vivirían los datos y el modelo y con qué modelo de
servicio; quién accede, con privilegio mínimo; qué se registra. En el ejemplo: la base no trae nombres ni documentos,
pero el barrio, la fecha y la hora juntos podrían señalar un accidente concreto y a quienes estuvieron en él."""),
]


def _lineas(texto):
    partes = texto.split("\n")
    return [p + "\n" for p in partes[:-1]] + [partes[-1]]


def armar():
    celdas = []
    for tipo, texto in CELDAS:
        if tipo == "md":
            celdas.append({"cell_type": "markdown", "metadata": {}, "source": _lineas(texto)})
        else:
            celdas.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _lineas(texto)})
    return {"cells": celdas,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    nb = armar()
    DESTINO.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Notebook:", DESTINO.relative_to(RAIZ))
    if "ejecuta" in sys.argv:
        import tempfile
        import time
        import nbformat
        from nbclient import NotebookClient
        cuaderno = nbformat.reads(json.dumps(nb), as_version=4)
        inicio = time.time()
        with tempfile.TemporaryDirectory() as carpeta:
            NotebookClient(cuaderno, timeout=900, kernel_name="replica-colab",
                           resources={"metadata": {"path": carpeta}}).execute()
        print(f"Ejecutado sin errores en {time.time() - inicio:.0f} s")
        for celda in cuaderno.cells:
            if celda.cell_type != "code":
                continue
            for salida in celda.outputs:
                if salida.output_type == "stream" and salida.name == "stderr":
                    print("AVISO en:", celda.source.split("\n")[0][:60], "→", salida.text[:200])
                texto = salida.get("text") or salida.get("data", {}).get("text/plain", "")
                if texto:
                    print("──", celda.source.split("\n")[0][:60]); print(texto[:1500])


if __name__ == "__main__":
    main()
