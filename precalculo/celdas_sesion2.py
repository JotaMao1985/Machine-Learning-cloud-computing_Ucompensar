# Fuente ÚNICA del código de la Sesión 2. De aquí salen:
#   - el notebook para Colab                    (notebook.py genera 2)
#   - las salidas y las gráficas que muestran las diapositivas
#                                               (notebook.py ejecuta 2 → salidas_sesion2.json y
#                                                fuente/diapositivas/recursos/s2/<id>.svg)
#
# No es un script ejecutable: es el formato «percent» de Jupyter.
#   # %% [markdown]            celda de texto (líneas con «# »)
#   # %% <id> [banderas]       celda de código
# Banderas:
#   generador   en el notebook se antepone el código de descarga_datos.py, ayudas.py y
#               ayudas_sesion2.py (celda oculta); los datos se descargan (M3)
#   solucion    solución de un ejercicio: se ejecuta para verificar, NO va al notebook
#   plantilla   ejercicio por completar: va al notebook, NO se ejecuta al verificar
#   precalculo  cifras que las diapositivas citan en el texto: se ejecuta, NO va al notebook

# %% [markdown]
# # Sesión 2 · Aprender con etiqueta
# **Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**
#
# Este notebook acompaña las [diapositivas de la sesión](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-2-aprender-con-etiqueta.html).
# Cada celda lleva entre corchetes el mismo identificador que el código proyectado, así que puedes seguir
# la clase celda a celda. La portada del curso está en https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/
# y el enunciado del proyecto final, en [su página](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/proyecto.html).
#
# 1. **Archivo → Guardar una copia en Drive** para que tus cambios no se pierdan.
# 2. Ejecuta las celdas **en orden** con **Mayús + Enter**. Varias usan resultados de las anteriores.
#    Si Colab avisa que el notebook no es de Google, elige **Ejecutar de todos modos**.
# 3. Si Colab se reinicia o una celda falla al repetirla, usa **Entorno de ejecución → Ejecutar anteriores**.
#    Las celdas de ejercicio dan error mientras tengan `...` sin reemplazar (si el error menciona `Ellipsis`,
#    te quedó un `...`). Colab se detiene en el primer error: completa los ejercicios en orden, o ejecuta
#    con **Mayús + Enter** solo las celdas que necesites.
# 4. La primera celda **descarga** los datos del caso *Yarumo Fibra S.A.S.*, una empresa **ficticia** (sus
#    datos se generaron para el curso y no corresponden a personas ni a empresas reales), y comprueba que
#    el archivo sea idéntico al del curso. Si la descarga falla, la celda te dice cómo subirlo a mano.
#
# En el código, los decimales van con punto (así los escribe Python); en el texto, con coma.

# %% datos generador
# Descarga datos/clientes.csv: los mismos 8000 clientes de la sesión 1
ruta = descargar_datos("datos")

# %% [markdown]
# ## 1 · Los datos
#
# Cargamos los clientes como en la sesión 1. Hoy usamos también **scikit-learn**, la biblioteca de
# aprendizaje automático que Colab ya trae instalada, y **XGBoost**, que también viene en Colab.
#
# **Repaso: lo nuevo de pandas y numpy que aparece hoy.** No hace falta memorizarlo; vuelve aquí cuando lo veas.
#
# | Escribes | Qué hace |
# |---|---|
# | `tabla.drop(columns=["a", "b"])` | La tabla sin esas columnas (no cambia la original) |
# | `tabla.loc[filas, columnas]` | Elige filas y columnas por su nombre, por ejemplo `clientes.loc[[0, 1], "contrato"]` |
# | `tabla.set_index("id_cliente")` | Usa esa columna como nombre de las filas |
# | `tabla.assign(nueva=valores)` | Una copia de la tabla con una columna más |
# | `tabla.T` | La tabla traspuesta: las filas pasan a columnas |
# | `tabla.round(3)` | Redondea a 3 decimales (con un diccionario, cada columna a los suyos) |
# | `np.sqrt(x)` | La raíz cuadrada, con numpy (`np`) |
# | `f"{x:.2%}"` | Escribe `x` como porcentaje, con 2 decimales: 0.7645 → 76.45% |

# %% cargar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost

clientes = pd.read_csv("datos/clientes.csv")
print(f"scikit-learn {sklearn.__version__} · xgboost {xgboost.__version__}")
clientes.shape

# %% [markdown]
# ### Diccionario de datos (el mismo de la sesión 1)
#
# | Columna | Qué registra | Valores o unidad |
# |---|---|---|
# | <code>id&#95;cliente</code> | Seudónimo del cliente | C00001… |
# | `ciudad` | Ciudad del servicio | Bogotá, Medellín, Cali, Barranquilla, Pereira |
# | `estrato` | Estrato de la **dirección del servicio** (lo asigna el municipio a la vivienda; no es el ingreso) | 1 a 6 |
# | `edad` | Edad del titular | años |
# | <code>tipo&#95;cliente</code> | Hogar o empresa pequeña | Hogar, Pyme |
# | `usuarios` | Personas que usan la conexión (declarado al contratar) | conteo |
# | `tecnologia` | Tecnología de acceso | Fibra, Inalámbrica |
# | <code>plan&#95;mbps</code> | Velocidad contratada | 100, 300, 500, 900 Mbps |
# | `contrato` | Permanencia pactada | Mes a mes, 12 meses, 24 meses |
# | <code>antiguedad&#95;meses</code> | Meses como cliente | 1 a 96 |
# | `tv`, `movil` | Si tiene también TV o línea móvil con Yarumo | Sí / No |
# | <code>medio&#95;pago</code> | Cómo paga | Débito automático, Tarjeta de crédito, Pago en línea, Efectivo |
# | `descuento` | Si tiene un descuento vigente (el de bienvenida dura 12 meses; algunos clientes antiguos tienen otras promociones) | Sí / No |
# | <code>factura&#95;mensual</code> | Valor mensual facturado | pesos (COP), ficticios |
# | <code>consumo&#95;gb</code> | Datos consumidos al mes, en promedio | GB |
# | <code>velocidad&#95;pct</code> | Velocidad medida como % de la contratada (promedio de 3 meses) | % |
# | <code>caidas&#95;3m</code> | Caídas del servicio en los últimos 3 meses | conteo |
# | <code>llamadas&#95;soporte&#95;3m</code> | Llamadas a soporte en los últimos 3 meses | conteo |
# | <code>pagos&#95;tardios&#95;12m</code> | Facturas pagadas tarde en los últimos 12 meses | conteo |
# | `cancelo` | **Etiqueta**: si canceló entre el 1 de julio y el 30 de septiembre de 2026 | Sí / No |

# %% [markdown]
# ## Las preguntas de hoy
#
# Hoy entrenamos modelos **supervisados**: aprenden de ejemplos que traen la respuesta, la **etiqueta**.
#
# - **Regresión**, cuando la etiqueta es un número. El jefe de TI planea la capacidad de la red y pregunta:
#   *¿cuántos GB al mes consumirá cada cliente?* La etiqueta es `consumo_gb`.
# - **Clasificación**, cuando la etiqueta es una categoría. La gerente de experiencia del cliente pregunta,
#   como en la sesión 1: *¿quién va a cancelar?* La etiqueta es `cancelo`.
# - **Nube**, cuando la pregunta es de acceso. El oficial de protección de datos pregunta: *¿quién puede ver
#   qué en Azure?* No se responde con un modelo sino con roles y directivas (parte 3).
#
# Cada modelo se mide en clientes que **no vio** al entrenar y se compara con su **línea base**.
#
# ---
# # Parte 1 · Regresión: ¿cuántos GB consumirá cada cliente?
#
# ## 2 · Las características, en números
#
# Las **características** (las columnas que el modelo puede mirar; aquí las llamamos **predictoras**) tienen que
# ser números, porque los modelos de scikit-learn solo leen números. Una columna de texto, como `contrato`, se vuelve varias
# columnas de 0 y 1, una por categoría: es la codificación ***one-hot***. `pd.get_dummies` la hace; con
# `drop_first=True` omite la primera categoría de cada columna (en orden alfabético), porque se deduce
# de las demás: si un cliente no es «24 meses» ni «Mes a mes», es «12 meses».

# %% predictoras
# Todas las columnas menos el seudónimo, la etiqueta de hoy (consumo_gb)
# y cancelo, que se supo después (de julio a septiembre)
predictoras = clientes.drop(
    columns=["id_cliente", "consumo_gb", "cancelo"])
X = pd.get_dummies(predictoras, drop_first=True, dtype=int)   # one-hot
y = clientes["consumo_gb"]                 # la etiqueta: GB al mes
print(f"{predictoras.shape[1]} columnas → {X.shape[1]} columnas numéricas")

# %% onehot
# Tres clientes, antes y después: cada categoría es una columna de 0 y 1
print(clientes.loc[[0, 1, 2], "contrato"].to_list())
X.loc[[0, 1, 2], ["contrato_24 meses", "contrato_Mes a mes"]]

# %% [markdown]
# El primer cliente tiene contrato a 12 meses: las dos columnas valen 0. El segundo es «Mes a mes» y el
# tercero, «24 meses». La ciudad, el medio de pago y las columnas Sí/No se codifican igual.
#
# Separamos los clientes como en la sesión 1: la cuarta parte se guarda para la **prueba**, con la misma
# semilla (2026), y el modelo nunca la ve al entrenar. Esta vez sin `stratify`, porque la etiqueta es un
# número: no son exactamente los mismos 2000 clientes de prueba de la sesión 1.

# %% particion
from sklearn.model_selection import train_test_split
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y, test_size=0.25, random_state=2026)
print(f"Entrenamiento: {len(X_ent)} clientes · prueba: {len(X_prueba)}")

# %% [markdown]
# ## 3 · La línea base y las tres métricas
#
# La **línea base** de una regresión predice el promedio del entrenamiento para todos los clientes, sin
# mirar ninguna columna (`DummyRegressor`). Un modelo que no le gane no sirve. Tres métricas, siempre
# en el conjunto de **prueba**:
#
# - **MAE** (error absoluto medio): cuántos GB se equivoca, en promedio, por cliente:
#   $\text{MAE} = \frac{1}{n}\sum |y_i - \hat y_i|$.
# - **RMSE** (raíz del error cuadrático medio): también en GB, pero eleva al cuadrado antes de promediar,
#   así que castiga más los errores grandes: $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat y_i)^2}$.
# - **R²**: la parte de la variación del consumo que el modelo reproduce. La línea base da 0; un modelo
#   perfecto, 1. No es un porcentaje de aciertos.

# %% linea_base_reg
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)
base = DummyRegressor()                    # predice el promedio, siempre
base.fit(X_ent, y_ent)                     # solo aprende el promedio
pred_base = base.predict(X_prueba)
print(f"Predice {pred_base[0]:.1f} GB para todos")
print(f"MAE  = {mean_absolute_error(y_prueba, pred_base):.1f} GB")
print(f"RMSE = {root_mean_squared_error(y_prueba, pred_base):.1f} GB")
print(f"R²   = {r2_score(y_prueba, pred_base):.3f}")

# %% [markdown]
# Para no repetir esas líneas con cada modelo, una función las junta: entrena con el conjunto de
# entrenamiento, predice el de prueba y devuelve las tres métricas.

# %% medir
def medir(modelo):
    """Entrena el modelo, predice la prueba y devuelve MAE, RMSE y R²."""
    modelo.fit(X_ent, y_ent)               # aprende de los ejemplos
    pred = modelo.predict(X_prueba)        # predice clientes que no vio
    return {"MAE": mean_absolute_error(y_prueba, pred),
            "RMSE": root_mean_squared_error(y_prueba, pred),
            "R²": r2_score(y_prueba, pred)}

resultados = {"Línea base": medir(DummyRegressor())}

# %% [markdown]
# ## 4 · Regresión lineal
#
# La regresión lineal suma cada columna multiplicada por un coeficiente: una recta (un plano, con
# muchas columnas) que deja la menor suma de errores al cuadrado.

# %% lineal
from sklearn.linear_model import LinearRegression
lineal = LinearRegression()
resultados["Lineal"] = medir(lineal)
pd.DataFrame(resultados).T.round(3)

# %% [markdown]
# ## 5 · K vecinos más cercanos (KNN) y por qué hay que escalar
#
# KNN predice el consumo de un cliente con el promedio de sus *k* vecinos más parecidos del
# entrenamiento. «Parecido» es una **distancia**, y la distancia suma diferencias de columnas con
# unidades distintas: años, personas y pesos. Mira tres clientes reales.

# %% tres
tres = clientes.set_index("id_cliente").loc[
    ["C00053", "C00611", "C01689"],
    ["edad", "usuarios", "factura_mensual", "consumo_gb"]]
tres                                   # A, B y C, en ese orden

# %% [markdown]
# A y B tienen la misma edad y los mismos usuarios, y consumen parecido; B paga 20 000 pesos más porque
# tiene un plan más rápido. C paga lo mismo que A, pero tiene 38 años más y consume mucho menos. ¿Cuál
# es el vecino más cercano de A?

# %% escala
columnas = ["edad", "usuarios", "factura_mensual"]
def distancia(datos, a, b):                # la distancia en línea recta
    return np.sqrt(((datos.loc[a] - datos.loc[b]) ** 2).sum())
crudos = tres[columnas]
print(f"Sin escalar: A–B {distancia(crudos, 'C00053', 'C00611'):.2f}",
      f"· A–C {distancia(crudos, 'C00053', 'C01689'):.2f}")
escalados = crudos / X_ent[columnas].std()   # en desviaciones estándar
print(f"Escalados:   A–B {distancia(escalados, 'C00053', 'C00611'):.2f}",
      f"· A–C {distancia(escalados, 'C00053', 'C01689'):.2f}")

# %% [markdown]
# Con solo estas tres columnas y sin escalar, 20 000 pesos de diferencia pesan más que 38 años: para la
# distancia, el cliente C de 66 años es el «vecino» de A, aunque consume la octava parte. **Escalar** pone
# cada columna en desviaciones estándar: resta su media y divide entre su desviación, las dos aprendidas
# **solo** del entrenamiento (`StandardScaler`). Así, 20 000 pesos son media desviación y 38 años, más de
# dos y media. (En la celda solo dividimos: restar la media no cambia las distancias.)
#
# Un `Pipeline` (aquí con `make_pipeline`) encadena los pasos: primero escala, luego aplica el modelo.
# Al entrenar, el escalador aprende la media y la desviación del entrenamiento; al predecir, usa esas
# mismas con la prueba.

# %% knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
knn = make_pipeline(StandardScaler(),      # primero escala, luego vecinos
                    KNeighborsRegressor(n_neighbors=10))
resultados["KNN (10 vecinos)"] = medir(knn)
pd.DataFrame(resultados).T.round(3)

# %% [markdown]
# KNN no le gana a la regresión lineal. Con 24 columnas escaladas, todas pesan igual en la distancia,
# también las que poco tienen que ver con el consumo (las caídas, el medio de pago, los pagos tardíos),
# y esas estorban. Los árboles, en cambio, eligen qué columnas usar.
#
# Antes de ejecutar la celda siguiente, predice: **sin escalar, ¿KNN saldrá mejor o peor?**

# %% knn_sin
knn_sin = KNeighborsRegressor(n_neighbors=10)   # sin escalar
r_sin = medir(knn_sin)
print(f"KNN sin escalar: MAE {r_sin['MAE']:.1f} GB · R² {r_sin['R²']:.3f}")

# %% [markdown]
# En estos datos, sin escalar sale un poco mejor. No es porque escalar esté mal: sin escalar, la distancia
# la deciden casi solo la factura y el plan, que resumen bien el consumo (sobre todo el de las empresas
# pequeñas), y las columnas que estorban quedan fuera por sus números pequeños. Es suerte de las unidades,
# no una regla: **se escala para que el resultado no dependa de las unidades**, no para ganar puntos. En la
# clasificación veremos un caso en que no escalar es un desastre.
#
# ## 6 · Árbol de decisión, bosque aleatorio y XGBoost
#
# Un **árbol de decisión** parte a los clientes con preguntas sobre una columna («¿tiene más de 3
# usuarios?») hasta formar grupos de consumo parecido, y predice el promedio de cada grupo. Sin límite
# de profundidad, sigue partiendo hasta que cada hoja tiene casi un solo cliente.

# %% arbol
from sklearn.tree import DecisionTreeRegressor
arbol = DecisionTreeRegressor(random_state=2026)   # sin límite
resultados["Árbol sin límite"] = medir(arbol)
r2_entrena = r2_score(y_ent, arbol.predict(X_ent))
print(f"R² en entrenamiento: {r2_entrena:.3f}")
print(f"R² en prueba:        {resultados['Árbol sin límite']['R²']:.3f}")

# %% [markdown]
# En entrenamiento, el árbol sin límite es perfecto; en prueba, peor que la regresión lineal. Memorizó
# a los clientes que vio, con sus rarezas al azar (el ruido): es **sobreajuste** (*overfitting*). Aprender no
# es memorizar.
#
# Un **bosque aleatorio** (*random forest*) entrena muchos árboles, cada uno con una muestra al azar (con
# repetición) de los clientes, y promedia sus predicciones: los errores de un árbol se compensan con los
# de otros. (El bosque de clasificación de la parte 2, además, sortea las columnas en cada pregunta.)
# `min_samples_leaf=5` pide al menos 5 clientes por hoja. Los ajustes que se eligen antes de entrenar, como
# el número de árboles o su profundidad, se llaman **hiperparámetros** (*hyperparameters*).

# %% bosque
from sklearn.ensemble import RandomForestRegressor
bosque = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               random_state=2026, n_jobs=-1)
resultados["Bosque aleatorio"] = medir(bosque)

# %% [markdown]
# **XGBoost** es un método de potenciación del gradiente (*gradient boosting*): entrena árboles
# pequeños **en secuencia**, y cada uno corrige lo que los anteriores todavía no aciertan. El
# `learning_rate` dice cuánto corrige cada árbol; `max_depth`, cuántas preguntas hace cada uno.

# %% xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                   random_state=2026)
resultados["XGBoost"] = medir(xgb)
tabla_reg = pd.DataFrame(resultados).T
tabla_reg.round(3)

# %% comparacion_reg
fig, ax = plt.subplots(figsize=(8.4, 3.9))
orden = tabla_reg.sort_values("MAE", ascending=False)
colores = ["#566B7F" if m == "Línea base" else "#8A2E9E"
           for m in orden.index]
ax.barh(orden.index, orden["MAE"], color=colores)
for i, valor in enumerate(orden["MAE"]):
    ax.text(valor + 2, i, f"{valor:.0f} GB", va="center", fontsize=12)
ax.set_xlim(0, orden["MAE"].max() * 1.18)
ax.set_title("El bosque y XGBoost se equivocan menos")
ax.set_xlabel("Error absoluto medio en prueba (GB por cliente)")
plt.show()

# %% [markdown]
# ¿Por qué los bosques y XGBoost le ganan a la recta? La gráfica compara, por grupos de edad de los
# clientes de **prueba**, el consumo real promedio con lo que predicen la regresión lineal y XGBoost.

# %% curva_edad
grupos = pd.cut(X_prueba["edad"], bins=[17, 24, 29, 34, 39, 44, 49, 54,
                                        59, 64, 69, 90])
curva = pd.DataFrame({"real": y_prueba,
                      "lineal": lineal.predict(X_prueba),
                      "xgboost": xgb.predict(X_prueba)}).groupby(
    grupos, observed=True).mean()
centros = [21, 27, 32, 37, 42, 47, 52, 57, 62, 67, 75]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.plot(centros, curva["real"], "o", color="#1E293B", label="Real")
ax.plot(centros, curva["lineal"], color="#566B7F", linewidth=2.5,
        label="Regresión lineal")
ax.plot(centros, curva["xgboost"], color="#F46201", linewidth=2.5,
        label="XGBoost")
ax.set_title("La recta no sigue al consumo por edad; XGBoost sí")
ax.set_xlabel("Edad del titular (años)")
ax.set_ylabel("Consumo promedio (GB al mes)")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# La recta se aleja en los extremos: para los de 70 años o más llega a predecir un consumo negativo. ¿Qué
# le falta? Probemos dos arreglos: agregarle la **edad al cuadrado** (para que pueda curvarse) o la **edad
# multiplicada por los usuarios** (para que el efecto de la edad dependa de cuántas personas usan la
# conexión). Antes de ejecutar, predice cuál ayuda más.

# %% interaccion
intentos = {"edad al cuadrado": X["edad"] ** 2,
            "edad × usuarios": X["edad"] * X["usuarios"]}
for nombre, columna in intentos.items():
    X2 = X.assign(nueva=columna)               # X con una columna más
    recta2 = LinearRegression().fit(X2.loc[X_ent.index], y_ent)
    pred2 = recta2.predict(X2.loc[X_prueba.index])
    mae2 = mean_absolute_error(y_prueba, pred2)
    r2 = r2_score(y_prueba, pred2)
    print(f"Recta + {nombre}: MAE {mae2:.1f} GB · R² {r2:.3f}")

# %% [markdown]
# La edad al cuadrado casi no ayuda; la edad × usuarios acerca la recta a los árboles. El consumo no es una
# suma de efectos: cumplir años le quita más GB a un hogar de muchas personas que a uno de una o dos. Esa
# combinación de columnas es una **interacción**. La regresión lineal solo suma; los árboles combinan
# columnas solos, porque cada pregunta se hace dentro del grupo que dejó la anterior.
#
# ---
# ## Ejercicio 1 · Un árbol con límite
#
# El árbol sin límite memorizó: R² de 1 en entrenamiento y mucho menos en prueba. Entrena un árbol de
# regresión con **profundidad máxima 6** (`max_depth=6`) y la misma semilla, y calcula su **MAE en
# prueba** y su **R² en entrenamiento y en prueba**. Pista: las celdas [linea_base_reg] y [arbol].
#
# Cambia cada `...` (mientras quede uno, la celda da un error que menciona `Ellipsis`). El verificador te
# dice qué está bien y, en lo demás, te da una pista. Después responde abajo, en «Tus respuestas»: ¿se
# sobreajusta menos que el árbol sin límite? ¿Cómo lo sabes? Si terminas antes, prueba con 4 y con 10
# niveles y mira qué pasa con la diferencia entre el entrenamiento y la prueba.

# %% ej1 plantilla
arbol6 = DecisionTreeRegressor(max_depth=..., random_state=2026)
arbol6.fit(..., ...)                       # entrena con el entrenamiento
pred_entrena = arbol6.predict(X_ent)
pred_prueba = arbol6.predict(...)          # predice la prueba
mae_prueba = ...                           # el MAE en prueba
r2_entrena6 = ...                          # el R² en entrenamiento
r2_prueba6 = ...                           # el R² en prueba
revisar_arbol(mae=mae_prueba, r2_entrenamiento=r2_entrena6,
              r2_prueba=r2_prueba6)

# %% ej1_solucion solucion
arbol6 = DecisionTreeRegressor(max_depth=6, random_state=2026)
arbol6.fit(X_ent, y_ent)
pred_entrena = arbol6.predict(X_ent)
pred_prueba = arbol6.predict(X_prueba)
mae_prueba = mean_absolute_error(y_prueba, pred_prueba)
r2_entrena6 = r2_score(y_ent, pred_entrena)
r2_prueba6 = r2_score(y_prueba, pred_prueba)
revisar_arbol(mae=mae_prueba, r2_entrenamiento=r2_entrena6,
              r2_prueba=r2_prueba6)

# %% [markdown]
# ---
# # Parte 2 · Clasificación: ¿quién va a cancelar?
#
# ## 7 · Las características y la línea base
#
# Ahora la etiqueta es `cancelo`, que convertimos en 1 (canceló) o 0 (no canceló). Las predictoras (las
# características) son todas las demás columnas, ahora **con** `consumo_gb`. Para no mezclarlas con las de la regresión, sus
# nombres llevan una **c** (de cancelación): `Xc_ent`, `yc_prueba`… Con `stratify=yc`, el
# entrenamiento y la prueba quedan con la misma tasa de cancelación, como en la sesión 1.

# %% preparar_c
predictoras_c = clientes.drop(columns=["id_cliente", "cancelo"])
Xc = pd.get_dummies(predictoras_c, drop_first=True, dtype=int)
yc = (clientes["cancelo"] == "Sí").astype(int)   # 1 = canceló
Xc_ent, Xc_prueba, yc_ent, yc_prueba = train_test_split(
    Xc, yc, test_size=0.25, random_state=2026, stratify=yc)
print(f"Prueba: {len(yc_prueba)} clientes; cancelaron {yc_prueba.sum()}")

# %% linea_base_c
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
base_c = DummyClassifier(strategy="most_frequent")   # siempre «No»
base_c.fit(Xc_ent, yc_ent)
pred_base_c = base_c.predict(Xc_prueba)
encontrados_base = ((pred_base_c == 1) & (yc_prueba == 1)).sum()
print(f"Exactitud: {accuracy_score(yc_prueba, pred_base_c):.2%}")
print(f"Encontró a {encontrados_base} de los {yc_prueba.sum()}",
      "que cancelaron")

# %% [markdown]
# Es la trampa de la exactitud de la sesión 1: decir siempre «No» acierta tres de cada cuatro veces y no
# encuentra a nadie.
#
# ## 8 · Regresión logística y la matriz de confusión
#
# La **regresión logística**, pese a su nombre, clasifica: combina las columnas como una regresión lineal
# y pasa el resultado por una curva en S que lo deja entre 0 y 1, la **probabilidad** de cancelar. Con
# `predict`, decide «canceló» si esa probabilidad es de 0,5 o más. Escalamos primero, como con KNN.

# %% logistica
from sklearn.linear_model import LogisticRegression
logistica = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000))
logistica.fit(Xc_ent, yc_ent)
pred_log = logistica.predict(Xc_prueba)    # 1 si P(cancela) pasa de 0.5
print(f"Exactitud: {accuracy_score(yc_prueba, pred_log):.2%}")
confusion_matrix(yc_prueba, pred_log)      # filas: real · columnas: predicho

# %% [markdown]
# La **matriz de confusión** cruza lo real (filas) con lo predicho (columnas). En scikit-learn, la
# primera fila y la primera columna son el 0 («No canceló»):
#
# | | Predijo «No» | Predijo «Sí» |
# |---|---|---|
# | **No canceló** | verdaderos negativos (VN) | falsos positivos (FP) |
# | **Canceló** | falsos negativos (FN) | verdaderos positivos (VP) |

# %% matriz_grafica
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ConfusionMatrixDisplay.from_predictions(
    yc_prueba, pred_log, display_labels=["No canceló", "Canceló"],
    cmap="Purples", colorbar=False, ax=ax)
ax.set_title("La logística encuentra a menos de la mitad")
ax.set_xlabel("Lo que predijo el modelo")
ax.set_ylabel("Lo que pasó")
plt.show()

# %% [markdown]
# Cuatro métricas salen de la matriz:
#
# - **Exactitud** (*accuracy*): de todos, cuántos acertó: $(VP + VN) / n$.
# - **Precisión** (*precision*): de los que señaló como «canceló», cuántos cancelaron: $VP / (VP + FP)$.
# - **Sensibilidad** (exhaustividad, *recall*): de los que cancelaron, a cuántos encontró: $VP / (VP + FN)$.
# - **F1**: la media armónica de la precisión y la sensibilidad, $2PS/(P + S)$; es alta solo si las dos
#   lo son.

# %% metricas_c
from sklearn.metrics import f1_score, precision_score, recall_score
vn, fp, fn, vp = confusion_matrix(yc_prueba, pred_log).ravel()
print(f"VN {vn} · FP {fp} · FN {fn} · VP {vp}")
print(f"Precisión:    {vp / (vp + fp):.3f}",
      f"= {precision_score(yc_prueba, pred_log):.3f}")
print(f"Sensibilidad: {vp / (vp + fn):.3f}",
      f"= {recall_score(yc_prueba, pred_log):.3f}")
print(f"F1:           {f1_score(yc_prueba, pred_log):.3f}")

# %% [markdown]
# Como en la regresión, una función junta las métricas de cada modelo. `encontrados` son los verdaderos
# positivos: a cuántos de los que cancelaron señaló el modelo. Son las cuentas de la sesión 1 con su nombre:
# los «encontrados» son los VP; las «falsas alarmas», los FP, y los que «se escaparon», los FN. «Señalar» a un
# cliente es predecir que cancela: en la campaña, es **llamarlo**.

# %% medir_c
REDONDEO = {"exactitud": 4, "precisión": 3, "sensibilidad": 3, "F1": 3}
def medir_c(modelo):
    """Entrena, decide con el umbral de 0.5 y devuelve las métricas."""
    modelo.fit(Xc_ent, yc_ent)
    pred = modelo.predict(Xc_prueba)
    return {"exactitud": accuracy_score(yc_prueba, pred),
            "precisión": precision_score(yc_prueba, pred, zero_division=0),
            "sensibilidad": recall_score(yc_prueba, pred),
            "F1": f1_score(yc_prueba, pred),
            "encontrados": ((pred == 1) & (yc_prueba == 1)).sum()}

metricas = {"Línea base": medir_c(
                DummyClassifier(strategy="most_frequent")),
            "Logística": medir_c(logistica)}
pd.DataFrame(metricas).T.astype({"encontrados": int}).round(REDONDEO)

# %% [markdown]
# ## 9 · Árboles: el sobreajuste se ve
#
# Un árbol de clasificación sin límite de profundidad hace lo mismo que en la regresión: parte hasta
# memorizar a cada cliente del entrenamiento.

# %% arbol_c
from sklearn.tree import DecisionTreeClassifier
arbol_c = DecisionTreeClassifier(random_state=2026)   # sin límite
metricas["Árbol sin límite"] = medir_c(arbol_c)
exactitud_entrena = accuracy_score(yc_ent, arbol_c.predict(Xc_ent))
print(f"Exactitud en entrenamiento: {exactitud_entrena:.2%}")
print(f"Exactitud en prueba:        "
      f"{metricas['Árbol sin límite']['exactitud']:.2%}")

# %% [markdown]
# Un árbol pequeño, en cambio, se puede **leer**. Con dos niveles de preguntas ya dice dónde está el
# riesgo. Cada hoja muestra cuántos clientes del entrenamiento no cancelaron y cuántos sí
# (`weights: [no, sí]`). Una regla que se puede leer es una explicación que se puede auditar.

# %% reglas
from sklearn.tree import export_text
arbol2 = DecisionTreeClassifier(max_depth=2, random_state=2026)
arbol2.fit(Xc_ent, yc_ent)
print(export_text(arbol2, feature_names=list(Xc.columns),
                  show_weights=True))

# %% [markdown]
# ¿Cómo elige el árbol cada pregunta? Busca la que deja grupos más **puros**: más parecidos entre sí en la
# etiqueta (scikit-learn lo mide con un criterio como el índice de Gini). De los 6000 clientes del
# entrenamiento cancelaron 1413; con contrato mes a mes, el 44,0 % (871 de 1979), y con permanencia, el 13,5 %
# (542 de 4021): ninguna otra pregunta los separa tanto. Por eso un árbol puede hacer una pregunta aunque sus
# dos hojas digan lo mismo, como el de la sesión 1: la pregunta sirve si deja grupos más puros, aunque no
# cambie la respuesta.
#
# Con límite de profundidad (4 niveles), el árbol ya no memoriza. Y un **bosque aleatorio** de 300
# árboles promedia sus votos.

# %% bosque_c
from sklearn.ensemble import RandomForestClassifier
arbol4 = DecisionTreeClassifier(max_depth=4, random_state=2026)
metricas["Árbol 4 niveles"] = medir_c(arbol4)
bosque_c = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  random_state=2026, n_jobs=-1)
metricas["Bosque aleatorio"] = medir_c(bosque_c)
pd.DataFrame(metricas).T.astype({"encontrados": int}).round(REDONDEO)

# %% [markdown]
# ## 10 · Máquina de vectores de soporte (SVM) y XGBoost
#
# Una **SVM** busca la frontera que separa a los que cancelan de los que no con el **margen** más ancho
# posible; con el *kernel* RBF (el de `SVC` por defecto), esa frontera puede ser curva. Como KNN, mide
# distancias, así que necesita escalar. Mira qué pasa si no se escala.

# %% svm
from sklearn.svm import SVC
svm_sin = SVC(random_state=2026)                       # sin escalar
svm = make_pipeline(StandardScaler(), SVC(random_state=2026))
sin_escalar = medir_c(svm_sin)
print(f"Sin escalar: exactitud {sin_escalar['exactitud']:.2%},",
      f"encontrados {sin_escalar['encontrados']}")
metricas["SVM"] = medir_c(svm)
print(f"Escalada:    exactitud {metricas['SVM']['exactitud']:.2%},",
      f"encontrados {metricas['SVM']['encontrados']}")

# %% [markdown]
# Sin escalar, la SVM hace lo mismo que la línea base: con la factura en pesos (cientos de miles) y el
# consumo en GB (cientos), la distancia la deciden casi solo esas dos columnas, que no separan a los que
# cancelan; el contrato, la antigüedad o la velocidad casi no cuentan, y el modelo termina diciendo «No» a
# todos. Escalada, encuentra a cientos.
#
# **XGBoost** para clasificar funciona como en la regresión: árboles pequeños en secuencia.

# %% xgboost_c
from xgboost import XGBClassifier
xgb_c = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                      random_state=2026)
metricas["XGBoost"] = medir_c(xgb_c)
tabla_c = pd.DataFrame(metricas).T.astype({"encontrados": int})
tabla_c.round(REDONDEO)

# %% [markdown]
# Con el umbral de 0,5, el mejor modelo encuentra a menos de la mitad de los que cancelan. Si la
# campaña de retención quiere encontrar a más, hay que **bajar el umbral**: llamar también a clientes con
# una probabilidad menor. Se encuentran más, pero se llama a más que no iban a cancelar.
#
# ## 11 · El umbral
#
# `predict_proba` da la probabilidad de cada clase; la columna `[:, 1]` es la de cancelar. Con la
# logística, probamos cuatro umbrales.

# %% umbral
prob_log = logistica.predict_proba(Xc_prueba)[:, 1]  # P(cancelar)
filas = {}
for umbral in [0.5, 0.4, 0.3, 0.2]:
    pred = (prob_log >= umbral).astype(int)          # 1 si llega al umbral
    filas[umbral] = {"llamadas": int(pred.sum()),
                     "encontrados": int(pred[yc_prueba == 1].sum()),
                     "precisión": precision_score(yc_prueba, pred),
                     "sensibilidad": recall_score(yc_prueba, pred),
                     "exactitud": accuracy_score(yc_prueba, pred)}
tabla_umbral = pd.DataFrame.from_dict(filas, orient="index")
tabla_umbral.round({"precisión": 3, "sensibilidad": 3, "exactitud": 4})

# %% umbral_grafica
umbrales = np.arange(0.05, 0.81, 0.01)
precisiones = [precision_score(yc_prueba, (prob_log >= u).astype(int),
                               zero_division=0) for u in umbrales]
sensibilidades = [recall_score(yc_prueba, (prob_log >= u).astype(int))
                  for u in umbrales]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.plot(umbrales, sensibilidades, color="#8A2E9E", linewidth=2.5,
        label="Sensibilidad")
ax.plot(umbrales, precisiones, color="#F46201", linewidth=2.5,
        label="Precisión")
ax.axvline(0.5, color="#566B7F", linestyle="--", linewidth=1)
ax.set_ylim(0, 1)
ax.set_title("Bajar el umbral encuentra más, con menos precisión")
ax.set_xlabel("Umbral de probabilidad para llamar")
ax.set_ylabel("Proporción")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 12 · Validación cruzada, sesgo y varianza
#
# Una sola partición puede tener suerte. La **validación cruzada** parte el **entrenamiento** en 5
# pliegues; entrena 5 veces, cada vez con 4 pliegues, y mide en el que dejó por fuera. Da 5 puntajes:
# su promedio estima el desempeño y su dispersión, qué tanto depende de los datos. La prueba sigue
# guardada para el final. `StratifiedKFold` conserva la tasa de cancelación en cada pliegue.

# %% validacion_cruzada
from sklearn.model_selection import StratifiedKFold, cross_val_score
pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
f1_xgb = cross_val_score(xgb_c, Xc_ent, yc_ent, cv=pliegues,
                         scoring="f1")
print("F1 de cada pliegue:", np.round(f1_xgb, 3))
print(f"F1 medio: {f1_xgb.mean():.3f} · desviación: {f1_xgb.std():.3f}")

# %% [markdown]
# Con la validación cruzada se ve el **sesgo** y la **varianza** de un modelo. Un árbol con pocos niveles
# es demasiado simple: se equivoca parecido en entrenamiento y en validación (**sesgo**, subajuste). Uno
# con muchos acierta casi todo en entrenamiento y falla en validación (**varianza**, sobreajuste). Entre
# los dos está la profundidad que mejor generaliza.

# %% curva_profundidad
from sklearn.model_selection import cross_validate
profundidades = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
entrena, valida = [], []
for p in profundidades:
    cv = cross_validate(DecisionTreeClassifier(max_depth=p,
                                               random_state=2026),
                        Xc_ent, yc_ent, cv=pliegues,
                        scoring="accuracy", return_train_score=True)
    entrena.append(cv["train_score"].mean())
    valida.append(cv["test_score"].mean())
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.plot(profundidades, entrena, "o-", color="#8A2E9E", linewidth=2.5,
        label="Entrenamiento")
ax.plot(profundidades, valida, "o-", color="#F46201", linewidth=2.5,
        label="Validación cruzada")
ax.set_title("Más niveles: el árbol memoriza y generaliza peor")
ax.set_xticks(profundidades)
ax.set_xlabel("Profundidad máxima del árbol (niveles)")
ax.set_ylabel("Exactitud")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 13 · ¿Qué modelo le sirve a la gerente?
#
# La gerente de experiencia del cliente no decide con el umbral de 0,5: decide **a cuántos llamar**. Su
# meta es encontrar a 7 de cada 10 de los que van a cancelar. Ordenamos a los clientes de prueba de
# mayor a menor riesgo según cada modelo y contamos cuántas llamadas hacen falta para llegar a la meta.
# Un modelo que ordena mejor necesita menos llamadas. La SVM no da probabilidades, pero su
# `decision_function` ordena igual: a más alta, más riesgo. Los clientes con el mismo riesgo se llaman
# todos juntos (el árbol de 4 niveles solo da unas pocas probabilidades distintas).

# %% llamadas
def llamadas_para(riesgo, meta=0.70):
    """Llamadas, de mayor a menor riesgo, para encontrar a la meta."""
    y = yc_prueba.to_numpy()
    for corte in np.sort(np.unique(riesgo))[::-1]:   # de mayor a menor
        llamados = riesgo >= corte          # los empatados, todos juntos
        if y[llamados].sum() >= meta * y.sum():
            return int(llamados.sum())

riesgos = {"Logística": prob_log,
           "Árbol 4 niveles": arbol4.predict_proba(Xc_prueba)[:, 1],
           "Bosque aleatorio": bosque_c.predict_proba(Xc_prueba)[:, 1],
           "SVM": svm.decision_function(Xc_prueba),
           "XGBoost": xgb_c.predict_proba(Xc_prueba)[:, 1]}
tabla_llamadas = pd.DataFrame(
    {"llamadas para 7 de cada 10":
         {m: llamadas_para(r) for m, r in riesgos.items()}})
tabla_llamadas["exactitud con 0.5"] = tabla_c["exactitud"].round(4)
tabla_llamadas.sort_values("llamadas para 7 de cada 10")

# %% [markdown]
# Si llamara al azar, para encontrar a 7 de cada 10 tendría que llamar a 7 de cada 10 clientes. XGBoost
# llega a la meta con menos llamadas que la logística y la SVM; el bosque queda casi empatado, y con una
# sola partición esa diferencia no se puede dar por segura. Pero no es lo único que cuenta: la gerente también
# tiene que poder **explicar** por qué se llama a un cliente, y el árbol pequeño y la logística se
# explican mejor. Esa es la «explicación auditable» del proyecto final.
#
# ---
# ## Ejercicio 2 · La matriz de confusión del bosque
#
# Con el bosque aleatorio ya entrenado (`bosque_c`, en la celda [bosque_c]), predice la prueba con el
# umbral de 0,5 (`predict`), calcula su matriz de confusión y saca de ella los cuatro números. Pista: las
# celdas [logistica] y [metricas_c]: el orden de `.ravel()` es VN, FP, FN, VP.
#
# Después responde abajo: el bosque acierta un poco más que la logística; con sus VP y FP, ¿cuál le sirve
# más a la gerente?

# %% ej2 plantilla
pred_bosque = ...                          # predice la prueba
matriz = ...                               # la matriz de confusión
vn_b = fp_b = fn_b = vp_b = ...   # sus cuatro números: cambia toda la línea
revisar_matriz(vp=vp_b, fp=fp_b, fn=fn_b, vn=vn_b)

# %% ej2_solucion solucion
pred_bosque = bosque_c.predict(Xc_prueba)
matriz = confusion_matrix(yc_prueba, pred_bosque)
vn_b, fp_b, fn_b, vp_b = matriz.ravel()
revisar_matriz(vp=vp_b, fp=fp_b, fn=fn_b, vn=vn_b)

# %% [markdown]
# ## Ejercicio 3 · El umbral de la gerente
#
# La gerente quiere encontrar **al menos a 7 de cada 10** de los que cancelan (sensibilidad de 0,70 o
# más), con el menor número de llamadas. Con las probabilidades de **XGBoost**, completa el ciclo: para
# cada umbral, la sensibilidad, las llamadas (cuántos clientes quedan por encima del umbral) y la
# exactitud. Elige el umbral **más alto** que cumple la meta, y entrégalo con sus llamadas. Pista: la
# celda [umbral].
#
# Para simplificar, aquí elegimos el umbral mirando la prueba. En el proyecto, elíjanlo con validación
# cruzada sobre el entrenamiento y midan la prueba una sola vez, al final.
#
# Después responde abajo: la gerente solo puede pagar 500 llamadas. ¿Qué umbral de la lista usarías?
# ¿Cuántos encontrarías? ¿Sigues cumpliendo la meta? Y una más: la lista salta de 0,05 en 0,05; prueba
# con pasos de 0,01 cerca del umbral que elegiste: ¿cuántas llamadas te ahorras?

# %% ej3 plantilla
prob_xgb = xgb_c.predict_proba(Xc_prueba)[:, 1]      # P(cancelar)
for umbral in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
    pred = (prob_xgb >= umbral).astype(int)
    sensibilidad = ...                     # recall_score(...)
    llamadas = ...                         # cuántos 1 hay en pred
    exactitud = ...                        # accuracy_score(...)
    print(f"umbral {umbral:.2f} ·", "sensibilidad", sensibilidad,
          "· llamadas", llamadas, "· exactitud", exactitud)
umbral_elegido = ...                       # el más alto que cumple
llamadas_elegido = ...                     # sus llamadas
revisar_umbral(umbral=umbral_elegido, llamadas=llamadas_elegido)

# %% ej3_solucion solucion
prob_xgb = xgb_c.predict_proba(Xc_prueba)[:, 1]
for umbral in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
    pred = (prob_xgb >= umbral).astype(int)
    sensibilidad = round(recall_score(yc_prueba, pred), 3)
    llamadas = int(pred.sum())
    exactitud = round(accuracy_score(yc_prueba, pred), 4)
    print(f"umbral {umbral:.2f} ·", "sensibilidad", sensibilidad,
          "· llamadas", llamadas, "· exactitud", exactitud)
umbral_elegido = 0.25
llamadas_elegido = int((prob_xgb >= 0.25).sum())
revisar_umbral(umbral=umbral_elegido, llamadas=llamadas_elegido)

# %% [markdown]
# ## Ejercicio 4 (opcional) · La profundidad que elige la validación cruzada
#
# Para cada profundidad de 1 a 10, calcula el **F1 medio** de un árbol de clasificación en la validación
# cruzada de 5 pliegues (`pliegues`, sobre el **entrenamiento**). Elige la profundidad con el F1 medio
# más alto y entrégala con ese F1. Pista: la celda [validacion_cruzada].
#
# Después responde abajo: ¿por qué no se elige la profundidad con la prueba?

# %% ej4 plantilla
for profundidad in range(1, 11):
    arbol_p = DecisionTreeClassifier(max_depth=profundidad,
                                     random_state=2026)
    f1_medio = ...                         # promedio de cross_val_score
    print(profundidad, f1_medio)
mejor_profundidad = ...
mejor_f1 = ...
revisar_profundidad(profundidad=mejor_profundidad, f1_medio=mejor_f1)

# %% ej4_solucion solucion
f1_por_profundidad = {}
for profundidad in range(1, 11):
    arbol_p = DecisionTreeClassifier(max_depth=profundidad,
                                     random_state=2026)
    f1_medio = cross_val_score(arbol_p, Xc_ent, yc_ent,
                               cv=pliegues, scoring="f1").mean()
    f1_por_profundidad[profundidad] = f1_medio
    print(profundidad, round(f1_medio, 3))
mejor_profundidad = max(f1_por_profundidad, key=f1_por_profundidad.get)
mejor_f1 = f1_por_profundidad[mejor_profundidad]
revisar_profundidad(profundidad=mejor_profundidad, f1_medio=mejor_f1)

# %% [markdown]
# ---
# # Parte 3 · Nube: ¿quién puede ver qué en Azure?
#
# ## Ejercicio de diseño · La matriz de roles de Yarumo
#
# Este ejercicio no lleva código: se hace en las salas, con las diapositivas de la sesión. Yarumo guarda
# en Azure, en el grupo de recursos `rg-yarumo-churn`, una cuenta de almacenamiento con tres contenedores:
#
# - `datos-clientes`: la tabla seudonimizada, que sigue teniendo datos personales. La carga cada mes un
#   proceso automático desde el CRM (el sistema de gestión de clientes).
# - `modelos`: el modelo entrenado.
# - `informes`: los resultados de cada campaña, sin datos de clientes uno a uno. Los escribe la aplicación.
#
# Los grupos: **Analítica** (tu equipo) entrena y evalúa los modelos; **TI** administra la infraestructura
# (crea, configura y borra recursos) y asigna los accesos; la **gerencia** lee los resultados; el
# **proveedor externo** es una consultora que revisa el modelo durante un mes, y **la aplicación que
# predice** calcula el riesgo de cada cliente con su identidad administrada.
#
# Para cada grupo, escribe el **rol integrado** de Azure y el **ámbito** más estrecho que le basta, o
# «sin acceso», y justifica con el privilegio mínimo. *Doble clic para escribir aquí*; si trabajan en sala,
# pueden escribir una línea por grupo en el chat.
#
# | Grupo | `datos-clientes` | `modelos` | `informes` | Por qué |
# |---|---|---|---|---|
# | Analítica | … | … | … | … |
# | TI | … | … | … | … |
# | Gerencia | … | … | … | … |
# | Proveedor externo | … | … | … | … |
# | La aplicación que predice (identidad administrada) | … | … | … | … |
#
# ---
# ### Tus respuestas
#
# *Doble clic para escribir aquí.*
#
# 1. Ejercicio 1: el árbol de 6 niveles tiene un R² de … en entrenamiento y de … en prueba (el sin límite:
#    1 y …). ¿Se sobreajusta menos? ¿Cómo lo sabes? ¿Lo elegirías antes que el bosque solo porque la
#    diferencia entre entrenamiento y prueba es menor? ¿Qué mirarías? …
# 2. Ejercicio 2: el bosque encontró a … y señaló a … que no cancelaron; la logística, a 186 y a 98. ¿Cuál le
#    sirve más a la gerente y por qué? ¿Cambia tu respuesta si cada llamada es muy cara? …
# 3. Ejercicio 3: con el umbral de …, XGBoost encuentra a … con … llamadas, y la exactitud pasa de … a … Con
#    500 llamadas, ¿qué umbral usarías y cumplirías la meta? Con pasos de 0,01, ¿cuántas llamadas ahorras? …
# 4. Ejercicio 4 (opcional): la validación cruzada elige … niveles. ¿Por qué no se elige con la prueba? …
#
# ### Autoevaluación (respóndela en el cierre de la sesión)
#
# ¿Verdadero o falso? Explica cada una.
#
# 1. «Si un modelo tiene una exactitud del 76 %, encuentra a 3 de cada 4 de los que cancelan». …
# 2. «Si un árbol tiene un R² de 1 en entrenamiento, con clientes nuevos tendrá un R² parecido». …
# 3. «Bajar el umbral sube la sensibilidad y suele bajar la precisión». …
# 4. «Un proveedor con el rol Lector de datos de blobs en el contenedor `modelos` no puede leer
#    `datos-clientes`». …

# %% [markdown]
# ---
# ## La respuesta a la gerencia
#
# **¿Cuántos GB consumirá cada cliente?** XGBoost y el bosque aleatorio se equivocan en promedio unos
# 70 GB por cliente, frente a unos 170 de la línea base, porque combinan columnas, como la edad con el
# número de usuarios, y la regresión lineal solo las suma.
#
# **¿Quién va a cancelar?** Con el umbral de 0,5, el mejor modelo encuentra a menos de la mitad de los
# que cancelan. Para encontrar a 7 de cada 10, XGBoost necesita menos llamadas que los demás modelos.
# La exactitud no sirve para elegir: la línea base acierta el 76 % y no encuentra a nadie. Y un modelo
# que se usa para llamar a clientes se tiene que poder explicar.
#
# **¿Quién puede ver qué en Azure?** Cada grupo recibe un rol integrado en el ámbito más estrecho que le
# basta: los datos personales, solo para Analítica y para la aplicación que predice; TI administra los
# recursos sin leer el contenido, y Azure Policy fija lo que nadie puede saltarse.

# %% cifras precalculo
# Cifras que las diapositivas y sus notas citan (no va al notebook)
import json
import time
np.set_printoptions(legacy="1.25")   # que los números salgan sin np.float64(...)
_cif = {}
_cif["n"] = len(clientes)
_cif["consumo"] = {"media": round(y.mean(), 1), "mediana": round(y.median(), 1),
                   "min": float(y.min()), "max": float(y.max()),
                   "p90": round(y.quantile(0.9), 1), "tope_2500": int((y == 2500).sum())}
_cif["tasa_cancelacion_total"] = round(yc.mean() * 100, 2)
_cif["prueba_c"] = {"n": int(len(yc_prueba)), "cancelaron": int(yc_prueba.sum()),
                    "tasa": round(yc_prueba.mean() * 100, 2)}
_cif["columnas"] = {"predictoras": predictoras.shape[1], "X": X.shape[1], "Xc": Xc.shape[1],
                    "Xc_lista": list(Xc.columns)}
_cif["reg"] = tabla_reg.round(4).to_dict("index")
_cif["reg_mejora_mae_xgb_vs_base"] = round(1 - tabla_reg.loc["XGBoost", "MAE"] / tabla_reg.loc["Línea base", "MAE"], 4)
_cif["reg_rmse_sobre_mae"] = (tabla_reg["RMSE"] / tabla_reg["MAE"]).round(3).to_dict()
_cif["arbol_reg"] = {"r2_entrena": round(r2_entrena, 4),
                     "hojas": int(arbol.get_n_leaves()), "profundidad": int(arbol.get_depth())}
_cif["curva_edad"] = curva.astype(float).round(1).to_dict("index")
_cif["curva_edad"] = {str(k): v for k, v in _cif["curva_edad"].items()}
_cif["escala"] = {"std_entrena": X_ent[columnas].std().round(2).to_dict(),
                  "dif_factura_AB": int(tres.loc["C00611", "factura_mensual"] - tres.loc["C00053", "factura_mensual"]),
                  "dif_edad_AC": int(tres.loc["C01689", "edad"] - tres.loc["C00053", "edad"]),
                  "factura_AB_en_desv": round(20000 / X_ent["factura_mensual"].std(), 3),
                  "edad_AC_en_desv": round(38 / X_ent["edad"].std(), 3),
                  "consumo_A_sobre_C": round(tres.loc["C00053", "consumo_gb"] / tres.loc["C01689", "consumo_gb"], 2)}
_cif["clf"] = tabla_c.round(4).to_dict("index")
_cif["svm_sin_escalar"] = {k: round(float(v), 4) for k, v in sin_escalar.items()}
_cif["arbol_c"] = {"exactitud_entrena": round(exactitud_entrena, 4),
                   "profundidad": int(arbol_c.get_depth()), "hojas": int(arbol_c.get_n_leaves())}
_cif["matriz_log"] = {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp),
                      "se_van_sin_encontrar": round(fn / (vp + fn) * 100, 1)}
_cif["matrices"] = {_m: dict(zip(["vn", "fp", "fn", "vp"], map(int, confusion_matrix(
    yc_prueba, _o.predict(Xc_prueba)).ravel()))) for _m, _o in
    {"logística": logistica, "árbol sin límite": arbol_c, "árbol 4": arbol4, "bosque": bosque_c,
     "svm": svm, "xgboost": xgb_c}.items()}
_cif["umbral_log"] = pd.DataFrame(filas).T.round(4).to_dict("index")
_cif["umbral_log"] = {str(k): v for k, v in _cif["umbral_log"].items()}
_pr = xgb_c.predict_proba(Xc_prueba)[:, 1]
_cif["umbral_xgb"] = {f"{u:.2f}": {"sens": round(recall_score(yc_prueba, (_pr >= u).astype(int)), 4),
                                   "prec": round(precision_score(yc_prueba, (_pr >= u).astype(int)), 4),
                                   "exact": round(accuracy_score(yc_prueba, (_pr >= u).astype(int)), 4),
                                   "llamadas": int((_pr >= u).sum())}
                      for u in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]}
_cif["cv_xgb_f1"] = {"pliegues": [round(v, 4) for v in f1_xgb], "media": round(f1_xgb.mean(), 4),
                     "desv": round(f1_xgb.std(), 4)}
_cif["curva_profundidad"] = {p: [round(a, 4), round(b, 4)] for p, a, b in zip(profundidades, entrena, valida)}
_cif["llamadas_70"] = {m: llamadas_para(r) for m, r in riesgos.items()}
_cif["llamadas_70_azar"] = int(np.ceil(0.70 * len(yc_prueba)))
_cif["meta_70_clientes"] = int(np.ceil(0.70 * yc_prueba.sum()))
_cif["llamadas_500"] = {m: int(yc_prueba.to_numpy()[np.argsort(-r, kind="stable")[:500]].sum()) for m, r in riesgos.items()}
_cif["valores_distintos_riesgo"] = {m: int(len(np.unique(r))) for m, r in riesgos.items()}
# Reglas del árbol de 2 niveles
_hojas = arbol2.apply(Xc_ent)
_cif["reglas"] = {int(h): {"n": int((_hojas == h).sum()), "cancelaron": int(yc_ent[_hojas == h].sum()),
                           "tasa": round(yc_ent[_hojas == h].mean() * 100, 1)} for h in np.unique(_hojas)}
# El árbol de dos niveles de la sesión 1 (mes a mes y antigüedad), para el puente
_X1 = pd.DataFrame({"mes_a_mes": (clientes["contrato"] == "Mes a mes").astype(int),
                    "antiguedad_meses": clientes["antiguedad_meses"]})
_X1e, _X1p, _y1e, _y1p = train_test_split(_X1, yc, test_size=0.25, random_state=2026, stratify=yc)
_a1 = DecisionTreeClassifier(max_depth=2, random_state=2026).fit(_X1e, _y1e)
_p1 = _a1.predict(_X1p)
_cif["arbol_s1"] = {"exactitud": round(accuracy_score(_y1p, _p1), 4), "encontrados": int(((_p1 == 1) & (_y1p == 1)).sum()),
                    "misma_prueba": bool((_X1p.index == Xc_prueba.index).all())}
# Tasas de cancelación de la S1 (glosario) por contrato
_cif["tasa_contrato"] = (yc.groupby(clientes["contrato"]).mean() * 100).round(1).to_dict()
# Validación cruzada de cuatro modelos (exactitud, F1, sensibilidad)
_cv4 = {}
for _nombre, _m in {"Logística": logistica, "Árbol 4 niveles": arbol4, "Bosque aleatorio": bosque_c,
                    "XGBoost": xgb_c}.items():
    _s = cross_validate(_m, Xc_ent, yc_ent, cv=pliegues, scoring=["accuracy", "f1", "recall"])
    _cv4[_nombre] = {k: [round(_s["test_" + k].mean(), 4), round(_s["test_" + k].std(), 4)]
                     for k in ("accuracy", "f1", "recall")}
_cif["cv4"] = _cv4
# F1 por profundidad (ejercicio 4) y pistas
_f1p = {p: round(cross_val_score(DecisionTreeClassifier(max_depth=p, random_state=2026), Xc_ent, yc_ent,
                                 cv=pliegues, scoring="f1").mean(), 4) for p in range(1, 11)}
_cif["ej4_f1_por_profundidad"] = _f1p
# Ejercicio 1: el árbol de 6 niveles
_a6 = DecisionTreeRegressor(max_depth=6, random_state=2026).fit(X_ent, y_ent)
_cif["ej1"] = {"mae": round(mean_absolute_error(y_prueba, _a6.predict(X_prueba)), 3),
               "r2_entrena": round(r2_score(y_ent, _a6.predict(X_ent)), 4),
               "r2_prueba": round(r2_score(y_prueba, _a6.predict(X_prueba)), 4)}
# Ejercicio 2: la matriz del bosque
_cif["ej2"] = dict(zip(["vn", "fp", "fn", "vp"], map(int, confusion_matrix(yc_prueba, bosque_c.predict(Xc_prueba)).ravel())))
# Cifras agregadas tras la revisión (6 de noviembre)
# KNN sin escalar, con la factura en pesos y en millones
_r = {"pesos": medir(KNeighborsRegressor(n_neighbors=10))}
_Xm = X.assign(factura_mensual=X["factura_mensual"] / 1e6)
_km = KNeighborsRegressor(n_neighbors=10).fit(_Xm.loc[X_ent.index], y_ent)
_pm = _km.predict(_Xm.loc[X_prueba.index])
_r["millones"] = {"MAE": mean_absolute_error(y_prueba, _pm), "R²": r2_score(y_prueba, _pm)}
_cif["knn_sin_escalar"] = {k: {m: round(float(v[m]), 4) for m in ("MAE", "R²")} for k, v in _r.items()}
# Distancias de A a B y a C con las 24 columnas, escaladas con el entrenamiento
_z = (X - X_ent.mean()) / X_ent.std()
_ids = clientes["id_cliente"]
_ia, _ib, _ic = (int(_ids[_ids == c].index[0]) for c in ("C00053", "C00611", "C01689"))
_cif["distancias_24"] = {"escaladas": {"A-B": round(float(np.sqrt(((_z.loc[_ia] - _z.loc[_ib]) ** 2).sum())), 2),
                                       "A-C": round(float(np.sqrt(((_z.loc[_ia] - _z.loc[_ic]) ** 2).sum())), 2)},
                         "sin_escalar": {"A-B": round(float(np.sqrt(((X.loc[_ia] - X.loc[_ib]) ** 2).sum())), 1),
                                         "A-C": round(float(np.sqrt(((X.loc[_ia] - X.loc[_ic]) ** 2).sum())), 1)}}
# La recta: predicciones negativas en la prueba y error absoluto de XGBoost
_pl = lineal.predict(X_prueba)
_ex = np.abs(y_prueba - xgb.predict(X_prueba))
_cif["recta_negativas"] = {"n": int((_pl < 0).sum()), "de": int(len(_pl))}
_cif["xgb_error_abs"] = {"mediana": round(float(_ex.median()), 1), "p90": round(float(_ex.quantile(0.9)), 1)}
# XGBoost sin la edad (la edad sí importa) y el consumo de los hogares por edad y usuarios
_xe = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=2026)
_xe.fit(X_ent.drop(columns="edad"), y_ent)
_pe = _xe.predict(X_prueba.drop(columns="edad"))
_cif["xgb_sin_edad"] = {"MAE": round(float(mean_absolute_error(y_prueba, _pe)), 2), "R²": round(float(r2_score(y_prueba, _pe)), 4)}
_ho = clientes[clientes["tipo_cliente"] == "Hogar"]
_ge = pd.cut(_ho["edad"], [17, 29, 48, 59, 90], labels=["18-29", "30-48", "49-59", "60+"])
_gu = pd.cut(_ho["usuarios"], [0, 2, 4, 20], labels=["1-2", "3-4", "5+"])
_cif["hogar_edad_usuarios"] = _ho.groupby([_ge, _gu], observed=True)["consumo_gb"].mean().round(0).unstack().to_dict("index")
# La pregunta de la apertura: con las llamadas del árbol sin limite, ¿cuántos encuentra la logística?
_n_arbol = int(arbol_c.predict(Xc_prueba).sum())
_cif["apertura"] = {"llamadas_arbol": _n_arbol, "llamadas_logistica": int(pred_log.sum()),
                    "logistica_con_llamadas_del_arbol": int(yc_prueba.to_numpy()[np.argsort(-prob_log, kind="stable")[:_n_arbol]].sum()),
                    "precision_log": round(float(precision_score(yc_prueba, pred_log)), 4),
                    "precision_arbol": round(float(precision_score(yc_prueba, arbol_c.predict(Xc_prueba))), 4)}
# Pureza de la primera pregunta del árbol (la promesa de la sesión 1), con el entrenamiento
_mm = Xc_ent["contrato_Mes a mes"] == 1
_cif["pureza"] = {"todos": [int(yc_ent.sum()), int(len(yc_ent)), round(float(yc_ent.mean()) * 100, 1)],
                  "mes_a_mes": [int(yc_ent[_mm].sum()), int(_mm.sum()), round(float(yc_ent[_mm].mean()) * 100, 1)],
                  "permanencia": [int(yc_ent[~_mm].sum()), int((~_mm).sum()), round(float(yc_ent[~_mm].mean()) * 100, 1)]}
# AUC por validación cruzada (cuatro modelos)
_cif["cv4_auc"] = {n: [round(float(s.mean()), 4), round(float(s.std()), 4)] for n, s in
                   {n: cross_val_score(m, Xc_ent, yc_ent, cv=pliegues, scoring="roc_auc") for n, m in
                    {"Logística": logistica, "Árbol 4 niveles": arbol4, "Bosque aleatorio": bosque_c,
                     "XGBoost": xgb_c}.items()}.items()}
# R² de entrenamiento y de prueba (la brecha) de los modelos de regresión
_cif["r2_entrena_prueba"] = {n: [round(float(r2_score(y_ent, m.predict(X_ent))), 4), round(float(r2_score(y_prueba, m.predict(X_prueba))), 4)]
                             for n, m in {"Lineal": lineal, "Árbol sin límite": arbol, "Bosque aleatorio": bosque, "XGBoost": xgb}.items()}
# Ejercicio 3: la exactitud con el umbral elegido y con pasos de 0,01
_u = np.round(np.arange(0.15, 0.51, 0.01), 2)
_s = {float(u): (float(recall_score(yc_prueba, (_pr >= u).astype(int))), int((_pr >= u).sum())) for u in _u}
_mejor01 = max(u for u, (s, n) in _s.items() if s >= 0.70)
_cif["ej3_paso_001"] = {"umbral": _mejor01, "sens": round(_s[_mejor01][0], 4), "llamadas": _s[_mejor01][1]}
print(json.dumps(_cif, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
