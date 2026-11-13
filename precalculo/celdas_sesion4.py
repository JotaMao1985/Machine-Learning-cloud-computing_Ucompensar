# Fuente ÚNICA del código de la Sesión 4. De aquí salen:
#   - el notebook para Colab                    (notebook.py genera 4)
#   - las salidas y las gráficas que muestran las diapositivas
#                                               (notebook.py ejecuta 4 → salidas_sesion4.json y
#                                                fuente/diapositivas/recursos/s4/<id>.svg)
#
# No es un script ejecutable: es el formato «percent» de Jupyter.
#   # %% [markdown]            celda de texto (líneas con «# »)
#   # %% <id> [banderas]       celda de código
# Banderas:
#   generador   en el notebook se antepone el código de descarga_datos.py, ayudas.py y
#               ayudas_sesion4.py (celda oculta); los datos se descargan (M3)
#   solucion    solución de un ejercicio: se ejecuta para verificar, NO va al notebook
#   plantilla   ejercicio por completar: va al notebook, NO se ejecuta al verificar
#   precalculo  cifras que las diapositivas citan en el texto: se ejecuta, NO va al notebook
#
# Keras: el backend se fija antes de importarlo, cada red empieza con
# keras.utils.set_random_seed(2026), fit y predict van con verbose=0 y la entrada se declara con
# keras.Input. Las cifras de Keras pueden variar un poco entre equipos: las diapositivas dicen
# «cerca de» y los verificadores usan rangos, que la celda [semillas] comprueba con otras semillas.

# %% [markdown]
# # Sesión 4 · Redes y datos en Azure
# **Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**
#
# Este notebook acompaña las [diapositivas de la sesión](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-4-redes-y-datos-en-azure.html).
# Cada celda lleva entre corchetes el mismo identificador que el código proyectado, así que puedes seguir
# la clase celda a celda. La portada del curso está en https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/
# y el enunciado del proyecto final, en [su página](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/proyecto.html).
#
# 1. **Archivo → Guardar una copia en Drive** para que tus cambios no se pierdan.
# 2. Ejecuta las celdas **en orden** con **Mayús + Enter**. Varias usan resultados de las anteriores.
#    Si Colab avisa que el notebook no es de Google, elige **Ejecutar de todos modos**.
# 3. Si Colab se reinicia o una celda falla al repetirla, usa **Entorno de ejecución → Ejecutar anteriores**.
#    Las celdas de ejercicio dan error mientras tengan `...` sin reemplazar (si el error menciona `Ellipsis`
#    o `ellipsis`, te quedó un `...`). Colab se detiene en el primer error: completa los ejercicios en orden,
#    o ejecuta con **Mayús + Enter** solo las celdas que necesites.
# 4. La primera celda **descarga** los datos del caso *Yarumo Fibra S.A.S.*, una empresa **ficticia** (sus
#    datos se generaron para el curso y no corresponden a personas ni a empresas reales), y comprueba que
#    el archivo sea idéntico al del curso. Si la descarga falla, la celda te dice cómo subirlo a mano.
# 5. Hoy entrenamos redes neuronales con **Keras**, que Colab ya trae. Basta el entorno normal, sin GPU: el
#    notebook entero tarda unos minutos. Las redes empiezan con pesos al azar; aunque fijamos la semilla,
#    en otro equipo (o con GPU) las cifras pueden salir un poco distintas de las de las diapositivas. Los
#    verificadores lo tienen en cuenta.
#
# En el código, los decimales van con punto (así los escribe Python); en el texto, con coma.

# %% datos generador
# Descarga datos/clientes.csv: los mismos 8000 clientes de las sesiones 1 a 3
ruta = descargar_datos("datos")

# %% reloj precalculo
import time
_t_inicio = time.time()   # para medir cuánto tarda el notebook del estudiante

# %% [markdown]
# ## 1 · Los datos y la partición de la sesión 2
#
# Volvemos a la pregunta de la gerente de experiencia del cliente: *¿quién va a cancelar?* Preparamos los
# datos exactamente como en la sesión 2: cada categoría en columnas de 0 y 1, la etiqueta `cancelo` en 1 o 0,
# y la cuarta parte de los clientes guardada para la **prueba**, con la semilla 2026 y la misma tasa de
# cancelación (`stratify`). Así, los 2000 clientes de prueba son los mismos de la sesión 2 y podemos comparar.

# %% preparar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

clientes = pd.read_csv("datos/clientes.csv")
Xc = pd.get_dummies(clientes.drop(columns=["id_cliente", "cancelo"]),
                    drop_first=True, dtype=int)
yc = (clientes["cancelo"] == "Sí").astype(int)   # 1 = canceló
Xc_ent, Xc_prueba, yc_ent, yc_prueba = train_test_split(
    Xc, yc, test_size=0.25, random_state=2026, stratify=yc)
print(f"{Xc.shape[1]} columnas · prueba: {len(yc_prueba)} clientes;",
      f"cancelaron {yc_prueba.sum()}")

# %% [markdown]
# ---
# # Parte 1 · De la neurona a la red
#
# ## 2 · Una neurona es una regresión logística
#
# Una **neurona** hace cuatro cosas: multiplica cada entrada por un **peso**, suma, agrega un **sesgo** (*bias*) y
# pasa el resultado por una **función de activación**. Con la activación **sigmoide**, $\sigma(z) = 1/(1 + e^{-z})$,
# el resultado queda entre 0 y 1: es la regresión logística de la sesión 2. Lo comprobamos con un cliente:
# calculamos a mano la suma ponderada y la sigmoide, y la comparamos con `predict_proba`.

# %% neurona
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
logistica = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000))
logistica.fit(Xc_ent, yc_ent)                  # la de la sesión 2
escalador, neurona = logistica[0], logistica[1]
x = escalador.transform(Xc_prueba.iloc[[0]])   # un cliente, escalado
z = x @ neurona.coef_.ravel() + neurona.intercept_[0]  # pesos y sesgo
p = 1 / (1 + np.exp(-z[0]))                    # activación sigmoide
print(f"{neurona.coef_.size} pesos y 1 sesgo · z = {z[0]:.3f}",
      f"→ P(cancela) = {p:.3f}")
p_sk = logistica.predict_proba(Xc_prueba.iloc[[0]])[0, 1]
print(f"predict_proba: {p_sk:.3f}")

# %% [markdown]
# ## 3 · Lo que una sola neurona no puede
#
# En la sesión 2, la regresión lineal no siguió el consumo de los clientes: el consumo baja con la edad, y baja
# más rápido en los hogares con muchas personas (la **interacción** edad × usuarios). La regresión lineal es una
# neurona **sin activación**: solo suma. Una **capa oculta** pone varias neuronas entre las columnas y la
# salida; cada una aprende su propia combinación, y la activación ReLU deja que la red se doble. Con el
# consumo (`consumo_gb`) como etiqueta y la partición de la regresión de la sesión 2:

# %% capa_oculta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
predictoras = clientes.drop(columns=["id_cliente", "consumo_gb", "cancelo"])
X = pd.get_dummies(predictoras, drop_first=True, dtype=int)
y = clientes["consumo_gb"]
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y, test_size=0.25, random_state=2026)
recta = LinearRegression().fit(X_ent, y_ent)  # una neurona, sin activar
capa = make_pipeline(StandardScaler(), MLPRegressor(
    hidden_layer_sizes=(32,), learning_rate_init=0.01,
    early_stopping=True, random_state=2026))   # 32 neuronas ocultas
capa.fit(X_ent, y_ent)
for nombre, modelo in {"Una neurona (la recta)": recta,
                       "Una capa oculta de 32": capa}.items():
    mae = mean_absolute_error(y_prueba, modelo.predict(X_prueba))
    print(f"{nombre}: MAE {mae:.1f} GB")

# %% [markdown]
# ## 4 · Las activaciones
#
# Sin activación, apilar capas no sirve: una suma de sumas sigue siendo una suma (una recta). La activación
# dobla la línea. Dos son las de esta parte (la **softmax** llega con los dígitos):
#
# - **ReLU**, $\max(0, z)$: deja pasar lo positivo y apaga lo negativo. Es la opción por defecto en las capas
#   ocultas.
# - **Sigmoide**: lleva cualquier número a una probabilidad entre 0 y 1. Va en la salida cuando la pregunta
#   es de sí o no.

# %% activaciones
z = np.linspace(-6, 6, 200)
fig, ejes = plt.subplots(1, 2, figsize=(8.4, 3.2))
ejes[0].plot(z, np.maximum(0, z), color="#8A2E9E", linewidth=2.5)
ejes[0].set_title("ReLU: max(0, z)")
ejes[1].plot(z, 1 / (1 + np.exp(-z)), color="#F46201", linewidth=2.5)
ejes[1].set_title("Sigmoide: entre 0 y 1")
for eje in ejes:
    eje.axhline(0, color="#94A3B8", linewidth=0.8)
    eje.axvline(0, color="#94A3B8", linewidth=0.8)
    eje.set_xlabel("z, la suma ponderada más el sesgo")
plt.show()

# %% [markdown]
# ## 5 · Entrenar es bajar la pérdida, paso a paso
#
# La **función de pérdida** dice cuánto se equivoca la red. Para un sí o un no se usa la **entropía cruzada
# binaria**: $-[\,y \log p + (1-y)\log(1-p)\,]$, donde $p$ es la probabilidad de cancelar que da la red; castiga
# mucho una probabilidad de cancelar alta para quien no canceló (o baja para quien sí). Entrenar es mover los pesos para que la pérdida baje: el **descenso del gradiente**
# mira hacia dónde baja la pérdida (la pendiente) y da un paso en esa dirección. El tamaño del paso es la
# **tasa de aprendizaje**. La gráfica lo muestra con un solo peso y una pérdida en forma de U.

# %% descenso
def bajar(tasa, inicio=-1.0, pasos=12):
    """Descenso del gradiente en la pérdida (w − 3)² + 1."""
    w, camino = inicio, [inicio]
    for _ in range(pasos):
        w = w - tasa * 2 * (w - 3)         # la pendiente es 2(w − 3)
        camino.append(w)
    return np.array(camino)

ws = np.linspace(-2, 8, 200)
fig, ejes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for eje, (tasa, nombre) in zip(ejes, [(0.03, "Paso corto: lento"),
                                      (0.3, "Paso bueno"),
                                      (0.97, "Paso largo: rebota")]):
    camino = bajar(tasa)
    eje.plot(ws, (ws - 3) ** 2 + 1, color="#94A3B8", linewidth=2)
    eje.plot(camino, (camino - 3) ** 2 + 1, "o-", color="#8A2E9E",
             markersize=5, linewidth=1.2)
    eje.set_title(f"{nombre} ({str(tasa).replace('.', ',')})")
    eje.set_xlabel("Peso w")
ejes[0].set_ylabel("Pérdida")
plt.show()

# %% [markdown]
# Con un paso corto, la red avanza poco y tarda en llegar; con uno largo, salta de un lado al otro del valle. Una red
# tiene miles de pesos, y la **retropropagación** (*backpropagation*) calcula la pendiente de la pérdida para
# todos a la vez. Tres palabras más:
#
# - una **época** es una pasada completa por los clientes del entrenamiento;
# - un **lote** (*batch*) es el grupo de clientes con que se calcula cada paso (aquí, 32);
# - **Adam** es el optimizador que usamos: ajusta por separado el tamaño del paso de cada peso.
#
# ---
# # Parte 2 · La red en Yarumo
#
# ## 6 · Una red de scikit-learn: `MLPClassifier`
#
# `MLPClassifier` es un **perceptrón multicapa** (MLP): capas ocultas con ReLU y una salida con la
# probabilidad de cancelar. Como KNN y la SVM de la sesión 2, es sensible a la escala de las columnas, así que
# va en un `Pipeline` que primero escala. `early_stopping=True` aparta el 10 % del entrenamiento y para
# cuando la red deja de mejorar en esos clientes.

# %% mlp
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
mlp = make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(32,), early_stopping=True, random_state=2026))
mlp.fit(Xc_ent, yc_ent)
pred_mlp = mlp.predict(Xc_prueba)
print(f"Precisión {precision_score(yc_prueba, pred_mlp):.3f}",
      f"· sensibilidad {recall_score(yc_prueba, pred_mlp):.3f}",
      f"· F1 {f1_score(yc_prueba, pred_mlp):.3f}")

# %% [markdown]
# ## 7 · La misma idea en Keras
#
# **Keras** arma una red capa por capa y la entrena con TensorFlow por debajo. El *backend* (el motor que hace las
# cuentas: TensorFlow, JAX o PyTorch) se fija **antes** de importar Keras. Además, en cada red se fija la semilla antes de armarla,
# para que los pesos al azar del comienzo sean los mismos en cada ejecución.
#
# Para ver si la red memoriza hace falta un tercer grupo de clientes, la **validación**: sale del
# entrenamiento (la prueba sigue guardada para el final) y la red no aprende de él; solo lo mira al final de
# cada época. La partimos a mano, estratificada, y el escalador aprende solo de los clientes con que se entrena.

# %% keras
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # antes de importar Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # menos avisos técnicos
import keras
from keras import layers

Xc_ent2, Xc_val, yc_ent2, yc_val = train_test_split(
    Xc_ent, yc_ent, test_size=0.2, random_state=2026, stratify=yc_ent)
escala = StandardScaler().fit(Xc_ent2)   # solo del entrenamiento
Xr_ent = escala.transform(Xc_ent2).astype("float32")
Xr_val = escala.transform(Xc_val).astype("float32")
Xr_prueba = escala.transform(Xc_prueba).astype("float32")
yr_ent, yr_val = yc_ent2.to_numpy(), yc_val.to_numpy()
print(f"Keras {keras.__version__} · entrenamiento {len(yr_ent)}",
      f"· validación {len(yr_val)} · prueba {len(Xr_prueba)}")

# %% [markdown]
# ## 8 · Una red grande, sin regularizar
#
# Dos capas ocultas de 256 neuronas para 4800 clientes: muchos más pesos que clientes. La entrenamos 40 épocas
# y guardamos el historial (`h_grande`): la pérdida en el entrenamiento y en la validación, época por época.

# %% red_grande
keras.utils.set_random_seed(2026)            # misma semilla, mismos pesos
grande = keras.Sequential([
    keras.Input(shape=(Xr_ent.shape[1],)),   # una entrada por columna
    layers.Dense(256, activation="relu"),    # capa oculta 1
    layers.Dense(256, activation="relu"),    # capa oculta 2
    layers.Dense(1, activation="sigmoid"),   # salida: P(cancela)
])
grande.compile(optimizer="adam", loss="binary_crossentropy")
h_grande = grande.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val),
                      epochs=40, batch_size=32, verbose=0)
print(f"{grande.count_params()} pesos y sesgos para {len(yr_ent)} clientes")

# %% curva_perdida
perdida = h_grande.history["loss"]
perdida_val = h_grande.history["val_loss"]
epocas = np.arange(1, len(perdida) + 1)
mejor = int(np.argmin(perdida_val)) + 1      # la época con menos pérdida
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.plot(epocas, perdida, color="#8A2E9E", linewidth=2.5,
        label="Entrenamiento")
ax.plot(epocas, perdida_val, color="#F46201", linewidth=2.5,
        label="Validación")
ax.axvline(mejor, color="#566B7F", linestyle="--", linewidth=1)
ax.set_title("La red grande memoriza: la validación sube casi desde el comienzo")
ax.set_xlabel("Época")
ax.set_ylabel("Pérdida (entropía cruzada)")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# En el entrenamiento, la pérdida baja casi hasta cero: la red memoriza a los 4800 clientes. En la validación
# toca fondo en las primeras épocas (la línea punteada) y después sube: con clientes que no vio, cada época la
# hace peor. Es el sobreajuste de la sesión 2, ahora época por época.
#
# ## 9 · La comparación honesta, con las métricas de la sesión 2
#
# La gerente no decide con la exactitud. Comparamos con lo que ya sabemos leer: la **precisión**, la
# **sensibilidad** y el **F1** con el umbral de 0,5. También contamos las **llamadas** que hacen falta para
# encontrar a 7 de cada 10 de los que cancelan, ordenando a los clientes de mayor a menor riesgo (la función de
# la sesión 2).
# Entrenamos también XGBoost, como en la sesión 2.

# %% comparacion
from xgboost import XGBClassifier
xgb_c = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                      random_state=2026).fit(Xc_ent, yc_ent)

def llamadas_para(riesgo, meta=0.70):
    """Llamadas, de mayor a menor riesgo, para llegar a la meta."""
    y_real = yc_prueba.to_numpy()
    for corte in np.sort(np.unique(riesgo))[::-1]:
        llamados = riesgo >= corte          # los empatados, todos juntos
        if y_real[llamados].sum() >= meta * y_real.sum():
            return int(llamados.sum())

def metricas(riesgo):
    """Las métricas de la sesión 2 para un riesgo por cliente."""
    pred = (riesgo >= 0.5).astype(int)      # el umbral de 0.5
    return {"precisión": precision_score(yc_prueba, pred),
            "sensibilidad": recall_score(yc_prueba, pred),
            "F1": f1_score(yc_prueba, pred),
            "llamadas 7 de 10": llamadas_para(riesgo)}

riesgos = {"Logística": logistica.predict_proba(Xc_prueba)[:, 1],
           "MLP (scikit-learn)": mlp.predict_proba(Xc_prueba)[:, 1],
           "Red grande (Keras)": grande.predict(Xr_prueba,
                                                verbose=0).ravel(),
           "XGBoost": xgb_c.predict_proba(Xc_prueba)[:, 1]}
tabla = pd.DataFrame({m: metricas(r) for m, r in riesgos.items()}).T
tabla.astype({"llamadas 7 de 10": int}).round(3)

# %% [markdown]
# El MLP iguala a la logística; la red grande que memorizó necesita más llamadas que las dos; XGBoost sigue al
# frente. En una tabla de clientes, con columnas que ya dicen mucho, los árboles son una línea base muy
# difícil de superar. Las redes ganan cuando el dato es una **imagen**, un **texto** o una **secuencia**:
# la parte 3.
#
# ## 10 · Regularizar: que la red no memorice
#
# Tres formas de frenar a una red que memoriza:
#
# - **Parada temprana** (*early stopping*): mira la pérdida de validación al final de cada época y para cuando
#   lleva `patience` épocas sin mejorar; con `restore_best_weights=True`, la red vuelve a los pesos de su mejor
#   época. En Keras es un *callback* (una función que Keras llama durante el entrenamiento, al final de cada época),
#   `keras.callbacks.EarlyStopping`.
# - ***Dropout***: en cada paso del entrenamiento apaga al azar una fracción de las neuronas (`rate`, entre 0 y
#   1), y así ninguna puede depender de otra en particular. Al predecir, todas trabajan. En Keras es una capa,
#   `layers.Dropout(rate)`, después de la capa que quieres frenar.
# - **L2**: castiga los pesos grandes. En `MLPClassifier` es el parámetro `alpha`.
#
# ---
# ## Ejercicio 1 · Escalar importa
#
# Entrena un MLP de **dos capas ocultas, de 16 y 8 neuronas**, dos veces: sin escalar y con escalado (en un
# `Pipeline` que primero escala), y calcula el **F1 en prueba** de cada uno. El código está casi completo:
# reemplaza cada `...`. Pista: la celda [mlp]. `zero_division=0` dice qué hacer si el modelo no señala a nadie:
# en ese caso, el F1 es 0.
#
# Después responde en «Tus respuestas», al final del notebook: ¿por qué cambia tanto? ¿Qué modelo de la sesión 2
# también dijo «No» a todos sin escalar?

# %% ej1 plantilla
mlp_sin = MLPClassifier(hidden_layer_sizes=(16, 8), early_stopping=True,
                        random_state=2026)            # sin escalar
mlp_con = make_pipeline(..., MLPClassifier(
    hidden_layer_sizes=(16, 8), early_stopping=True,
    random_state=2026))                               # primero escala
mlp_sin.fit(Xc_ent, yc_ent)
mlp_con.fit(..., ...)                       # entrena con el entrenamiento
f1_sin = f1_score(yc_prueba, mlp_sin.predict(Xc_prueba), zero_division=0)
f1_con = ...                                # el F1 en prueba del escalado
revisar_escalado(f1_sin=f1_sin, f1_con=f1_con)

# %% ej1_solucion solucion
mlp_sin = MLPClassifier(hidden_layer_sizes=(16, 8), early_stopping=True,
                        random_state=2026)
mlp_con = make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(16, 8), early_stopping=True,
    random_state=2026))
mlp_sin.fit(Xc_ent, yc_ent)
mlp_con.fit(Xc_ent, yc_ent)
f1_sin = f1_score(yc_prueba, mlp_sin.predict(Xc_prueba), zero_division=0)
f1_con = f1_score(yc_prueba, mlp_con.predict(Xc_prueba), zero_division=0)
revisar_escalado(f1_sin=f1_sin, f1_con=f1_con)

# %% [markdown]
# ## Ejercicio 2 · Que no memorice
#
# La red es la misma de la celda [red_grande]: dos capas ocultas de 256. Le agregamos un *dropout* después de
# cada capa oculta y la parada temprana. Completa dos valores:
#
# - `tasa_dropout`: la tasa de *dropout*, la fracción de neuronas que se apagan en cada paso (un número entre 0 y 1;
#   prueba entre 0.2 y 0.5). No es la tasa de aprendizaje;
# - `paciencia`: cuántas épocas seguidas sin mejorar en la validación espera antes de parar (prueba entre 3 y 15).
#
# Después calcula el **F1** y la **sensibilidad** en la prueba, con el umbral de 0,5. Pista: las celdas [mlp] y
# [comparacion]. El verificador revisa tus cifras con tu propia red y mira su historial.
#
# Cuando el verificador diga «¡Listo!», ejecuta la celda otra vez con `tasa_dropout = 0` y después con `0.8` (la
# misma paciencia), y anota en «Tus respuestas» la época en que paró, su mejor época y el F1 de cada una. El
# verificador te dirá que esas dos no son la respuesta: es lo esperado. ¿Cuánto aporta el *dropout*? ¿Qué relación
# hay entre la época en que paró, la mejor y la paciencia?

# %% ej2 plantilla
tasa_dropout = ...         # entre 0 y 1: la fracción que se apaga
paciencia = ...            # épocas sin mejorar antes de parar
keras.utils.set_random_seed(2026)
red = keras.Sequential([
    keras.Input(shape=(Xr_ent.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(tasa_dropout),            # apaga neuronas al azar
    layers.Dense(256, activation="relu"),
    layers.Dropout(tasa_dropout),
    layers.Dense(1, activation="sigmoid"),
])
red.compile(optimizer="adam", loss="binary_crossentropy")
parada = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=paciencia, restore_best_weights=True)
h = red.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val), epochs=100,
            batch_size=32, verbose=0, callbacks=[parada])
pred_red = (red.predict(Xr_prueba, verbose=0).ravel() >= 0.5).astype(int)
f1_red = ...               # el F1 en prueba
sens_red = ...             # la sensibilidad en prueba
revisar_red(red=red, historial=h, f1=f1_red, sensibilidad=sens_red)

# %% ej2_solucion solucion
tasa_dropout = 0.3
paciencia = 10
keras.utils.set_random_seed(2026)
red = keras.Sequential([
    keras.Input(shape=(Xr_ent.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(tasa_dropout),
    layers.Dense(256, activation="relu"),
    layers.Dropout(tasa_dropout),
    layers.Dense(1, activation="sigmoid"),
])
red.compile(optimizer="adam", loss="binary_crossentropy")
parada = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=paciencia, restore_best_weights=True)
h = red.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val), epochs=100,
            batch_size=32, verbose=0, callbacks=[parada])
pred_red = (red.predict(Xr_prueba, verbose=0).ravel() >= 0.5).astype(int)
f1_red = f1_score(yc_prueba, pred_red)
sens_red = recall_score(yc_prueba, pred_red)
revisar_red(red=red, historial=h, f1=f1_red, sensibilidad=sens_red)

# %% [markdown]
# ---
# # Parte 3 · Imágenes y secuencias
#
# Si no terminaste el ejercicio 1 o el 2, ejecuta estas celdas una por una con **Mayús + Enter**: no dependen de
# los ejercicios.
#
# ## 11 · Dígitos escritos a mano, de 8 × 8 píxeles
#
# En la tabla de clientes, la red no le ganó a XGBoost. Donde las redes brillan es en datos con **forma**: una
# imagen, donde cada píxel tiene vecinos. scikit-learn trae 1797 dígitos escritos a mano (`load_digits`), de
# 8 × 8 píxeles con valores de 0 a 16. No hay que descargar nada. Los dividimos entre 16 para que vayan de
# 0 a 1, y les agregamos un **canal** (la última dimensión), como espera Keras.

# %% digitos
from sklearn.datasets import load_digits
digitos = load_digits()
# de 0 a 1, y con un canal: (imágenes, alto, ancho, 1)
imagenes = (digitos.images / 16).reshape(-1, 8, 8, 1).astype("float32")
I_ent, I_prueba, d_ent, d_prueba = train_test_split(
    imagenes, digitos.target, test_size=0.25, random_state=2026,
    stratify=digitos.target)
print(imagenes.shape, "→", len(I_ent), "para entrenar y",
      len(I_prueba), "para probar")

# %% digitos_figura
fig, ejes = plt.subplots(1, 10, figsize=(10.5, 1.6))
for k, eje in enumerate(ejes):
    eje.imshow(digitos.images[k], cmap="Purples")
    eje.set_title(str(digitos.target[k]))
    eje.axis("off")
plt.show()

# %% [markdown]
# ## 12 · Un filtro de 3 × 3 que recorre la imagen
#
# Una **convolución** pasa un **filtro**, una ventana de 3 × 3 pesos, por cada posición de la imagen: en cada
# una multiplica los 9 píxeles por los 9 pesos y suma. Los **mismos 9 pesos** sirven en toda la imagen. Este
# filtro, hecho a mano, responde a los bordes verticales: positivo donde la tinta aparece de izquierda a derecha
# y negativo donde desaparece. Una CNN no recibe los filtros hechos: los aprende.

# %% filtro
filtro = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])              # bordes verticales
imagen = digitos.images[0]                   # un 0
mapa = np.zeros((6, 6))
for i in range(6):                           # cada posición de la ventana
    for j in range(6):
        mapa[i, j] = (imagen[i:i + 3, j:j + 3] * filtro).sum()
fig, ejes = plt.subplots(1, 2, figsize=(6.4, 3.0))
ejes[0].imshow(imagen, cmap="Purples")
ejes[0].set_title("La imagen (8 × 8)")
ejes[1].imshow(mapa, cmap="PuOr_r")
ejes[1].set_title("Con el filtro (6 × 6)")
for eje in ejes:
    eje.axis("off")
plt.show()

# %% [markdown]
# La ventana de 3 × 3 cabe en 6 × 6 posiciones de una imagen de 8 × 8. Con `padding="same"`, Keras rodea la
# imagen de ceros para que el mapa conserve su tamaño de 8 × 8. Después, el ***pooling*** resume: `MaxPooling2D(2)`
# se queda con el máximo de cada cuadro de 2 × 2 (8 × 8 pasa a 4 × 4), y `GlobalMaxPooling2D` se queda con el
# máximo de cada filtro en toda la imagen: dice *si* apareció el rasgo, no *dónde*.
#
# ## 13 · Una CNN y una red densa, con casi los mismos pesos
#
# La **red convolucional** (CNN) tiene dos capas de filtros, *pooling* y una salida **softmax**: diez
# probabilidades, una por dígito, que suman 1. La **red densa** aplana la imagen en 64 números y la trata como
# una fila de una tabla. Con diez clases, la pérdida es la entropía cruzada para varias clases.

# %% cnn
keras.utils.set_random_seed(2026)
cnn = keras.Sequential([
    keras.Input(shape=(8, 8, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),  # 16 filtros
    layers.MaxPooling2D(2),                   # 8 × 8 → 4 × 4
    layers.Conv2D(32, 3, padding="same", activation="relu"),  # 32 filtros
    layers.GlobalMaxPooling2D(),              # el máximo de cada filtro
    layers.Dense(10, activation="softmax"),   # P de cada dígito
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
h_cnn = cnn.fit(I_ent, d_ent, epochs=30, batch_size=32, verbose=0)

keras.utils.set_random_seed(2026)
densa = keras.Sequential([
    keras.Input(shape=(8, 8, 1)),
    layers.Flatten(),                         # 8 × 8 → 64 números
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
densa.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
h_densa = densa.fit(I_ent, d_ent, epochs=30, batch_size=32, verbose=0)

# %% exactitud
def exactitud(modelo, imagenes, etiquetas=d_prueba):
    """La proporción de dígitos que acierta (el de mayor probabilidad)."""
    prob = modelo.predict(imagenes, verbose=0)   # 10 probabilidades
    return float((prob.argmax(axis=1) == etiquetas).mean())

for nombre, modelo in {"CNN": cnn, "Densa": densa}.items():
    print(f"{nombre}: {modelo.count_params()} pesos ·",
          f"exactitud {exactitud(modelo, I_prueba):.3f}")

# %% [markdown]
# Con las imágenes tal como vienen, las dos aciertan casi todo. Pero en una foto real el dígito no siempre
# está centrado. ¿Qué pasa si cada imagen se **corre un píxel** a la derecha? La función `correr` lo hace: mueve
# las columnas y rellena con ceros la que queda vacía.

# %% correr
def correr(imagenes, pixeles=1):
    """Corre cada imagen hacia la derecha; entran columnas de ceros."""
    corridas = np.zeros_like(imagenes)
    corridas[:, :, pixeles:] = imagenes[:, :, :imagenes.shape[2] - pixeles]
    return corridas

I_corridas = correr(I_prueba)
fig, ejes = plt.subplots(1, 2, figsize=(4.6, 2.6))
ejes[0].imshow(I_prueba[0, :, :, 0], cmap="Purples")
ejes[0].set_title("Original")
ejes[1].imshow(I_corridas[0, :, :, 0], cmap="Purples")
ejes[1].set_title("Un píxel a la derecha")
for eje in ejes:
    eje.axis("off")
plt.show()

# %% [markdown]
# ## Ejercicio 3 · Correr un píxel
#
# Si el ejercicio 2 quedó a medias: haz clic en la celda [digitos] → **Entorno de ejecución → Ejecutar después**.
# Se detiene aquí, en el ejercicio 3.
#
# Antes de ejecutar, predice y anota en «Tus respuestas»: con las imágenes corridas, ¿cuál pierde más, la CNN o la
# densa?
#
# Calcula la exactitud de las dos redes con las imágenes de prueba **tal como vienen** (`I_prueba`) y con las
# **corridas** (`I_corridas`). Pista: la función `exactitud` de la celda [exactitud].
#
# Después responde en «Tus respuestas»: ¿cuál conserva más? ¿Qué parte de la CNN crees que la ayuda? Si te sobra tiempo,
# prueba con `correr(I_prueba, 2)`.

# %% ej3 plantilla
exactitud_cnn = ...              # la CNN, con I_prueba
exactitud_densa = ...            # la densa, con I_prueba
corrida_cnn = ...                # la CNN, con I_corridas
corrida_densa = ...              # la densa, con I_corridas
revisar_corrimiento(cnn=cnn, densa=densa,
                    exactitud_cnn=exactitud_cnn,
                    exactitud_densa=exactitud_densa,
                    corrida_cnn=corrida_cnn, corrida_densa=corrida_densa)

# %% ej3_solucion solucion
exactitud_cnn = exactitud(cnn, I_prueba)
exactitud_densa = exactitud(densa, I_prueba)
corrida_cnn = exactitud(cnn, I_corridas)
corrida_densa = exactitud(densa, I_corridas)
revisar_corrimiento(cnn=cnn, densa=densa,
                    exactitud_cnn=exactitud_cnn,
                    exactitud_densa=exactitud_densa,
                    corrida_cnn=corrida_cnn, corrida_densa=corrida_densa)

# %% [markdown]
# ## 14 · Una secuencia: el tráfico de Yarumo, hora por hora
#
# El jefe de TI quiere anticipar el tráfico de la próxima hora para planear la capacidad. Yarumo no nos dio esa
# serie, así que la **inventamos** aquí, con una fórmula a la vista (es ficticia, como todo el caso): 70 días,
# hora por hora, en Gbps (gigabits por segundo), con
#
# - un ciclo diario: más tráfico hacia las 5 p. m. y menos hacia las 5 a. m.;
# - 8 Gbps más los fines de semana (el día 0 es un lunes);
# - un ruido que arrastra el 70 % del de la hora anterior: si una hora vino alta, la siguiente suele venir alta.

# %% serie
rng = np.random.default_rng(2026)
horas = np.arange(24 * 70)                   # 70 días, hora por hora
hora, dia = horas % 24, horas // 24
ruido = np.zeros(len(horas))
for t in range(1, len(horas)):          # arrastra el de la hora anterior
    ruido[t] = 0.7 * ruido[t - 1] + rng.normal(0, 3)
trafico = (40 + 25 * np.sin(2 * np.pi * (hora - 11) / 24)  # ciclo diario
           + 8 * (dia % 7 >= 5)              # fin de semana
           + ruido)                           # Gbps
print(f"{len(trafico)} horas · de {trafico.min():.1f} a",
      f"{trafico.max():.1f} Gbps")

# %% serie_grafica
fig, ax = plt.subplots(figsize=(8.4, 3.2))
ax.plot(horas[:24 * 14] / 24, trafico[:24 * 14], color="#8A2E9E",
        linewidth=1.5)
ax.set_title("Dos semanas de tráfico (ficticio)")
ax.set_xlabel("Día (el 0 es un lunes)")
ax.set_ylabel("Gbps")
plt.show()

# %% [markdown]
# Para predecir la hora siguiente, cada ejemplo es una **ventana** con las últimas 24 horas, y la etiqueta es la
# hora que sigue. En una serie de tiempo **no se baraja**: se entrena con el pasado (el primer 80 %) y se prueba
# con el futuro (el último 20 %, unos 14 días).

# %% ventanas
V = 24                                       # la ventana: 24 horas
X_s = np.array([trafico[t - V:t] for t in range(V, len(trafico))])
y_s = trafico[V:]                            # la hora siguiente
corte = int(len(y_s) * 0.8)                  # sin barajar
Xs_ent, Xs_prueba = X_s[:corte], X_s[corte:]
ys_ent, ys_prueba = y_s[:corte], y_s[corte:]
print(f"{len(ys_ent)} ventanas para entrenar y {len(ys_prueba)}",
      "para probar")

# %% [markdown]
# Tres líneas base antes de la red: decir que la próxima hora será igual a **la última hora**; igual a **la
# misma hora de ayer**; o una **regresión lineal** con las 24 horas de la ventana. Medimos con el MAE de la
# sesión 2: cuántos Gbps se equivoca, en promedio, cada predicción.

# %% bases_serie
errores = {
    "La última hora": mean_absolute_error(ys_prueba, Xs_prueba[:, -1]),
    "La misma hora de ayer": mean_absolute_error(ys_prueba,
                                                 Xs_prueba[:, 0]),
}
recta_s = LinearRegression().fit(Xs_ent, ys_ent)   # pesos para 24 horas
errores["Regresión lineal (24 h)"] = mean_absolute_error(
    ys_prueba, recta_s.predict(Xs_prueba))
pd.DataFrame({"MAE en prueba (Gbps)": errores}).round(2)

# %% [markdown]
# ## 15 · Una red recurrente
#
# Una **red recurrente** (RNN) lee la ventana hora por hora y arrastra un **estado**: un resumen de lo que ya
# leyó, que mezcla con cada hora nueva. `SimpleRNN` es la más sencilla. Como siempre, escalamos con el
# entrenamiento.

# %% rnn
media, desv = ys_ent.mean(), ys_ent.std()    # escala del entrenamiento
keras.utils.set_random_seed(2026)
rnn = keras.Sequential([
    keras.Input(shape=(V, 1)),               # 24 horas, un valor por hora
    layers.SimpleRNN(16),                    # lee y arrastra un estado
    layers.Dense(1),                         # la hora siguiente
])
rnn.compile(optimizer="adam", loss="mse")
rnn.fit(((Xs_ent - media) / desv)[..., None], (ys_ent - media) / desv,
        epochs=40, batch_size=32, verbose=0)
pred_rnn = rnn.predict(((Xs_prueba - media) / desv)[..., None],
                       verbose=0).ravel() * desv + media
errores["Red recurrente"] = mean_absolute_error(ys_prueba, pred_rnn)
pd.DataFrame({"MAE en prueba (Gbps)": errores}).round(2)

# %% [markdown]
# La red recurrente le gana a las dos primeras líneas base y **empata** con la regresión lineal. No es un fracaso
# de la red: esta serie depende sobre todo de la última hora y de la hora del día, y eso una suma ponderada de
# las 24 horas ya lo capta. La recurrencia sirve cuando lo que importa quedó lejos en la secuencia; para eso se
# inventó la **LSTM**, una red recurrente con compuertas que deciden qué recordar y qué olvidar.
#
# ---
# # Parte 4 · Nube: los datos de Yarumo en Azure
#
# ## Ejercicio de diseño · El plan de almacenamiento y monitoreo
#
# Este ejercicio no lleva código: se hace en parejas, con las diapositivas de la sesión, y va directo a la parte
# 2 del proyecto. Para cada **activo** de Yarumo, completa las cinco columnas:
#
# 1. **Servicio y redundancia**: ¿un blob en Blob Storage o una tabla en Azure SQL Database? ¿LRS, ZRS o GRS?
# 2. **Quién lee**: el rol integrado y su ámbito (la matriz de roles de la sesión 2).
# 3. **Cómo se autoriza**: ¿Microsoft Entra ID, la clave compartida o una firma de acceso compartido (SAS)?
#    ¿Acceso anónimo?
# 4. **Qué se registra y qué alerta**: ¿el registro de actividad, una configuración de diagnóstico con
#    `StorageRead` hacia Log Analytics? ¿Qué alerta y a quién?
# 5. **Costo**: según la página oficial de servicios gratuitos de Azure, ¿cabe en lo gratuito? ¿Por cuánto tiempo?
#
# *Doble clic para escribir aquí.*
#
# | Activo | Servicio y redundancia | Quién lee (rol y ámbito) | Cómo se autoriza | Qué se registra y qué alerta | Costo |
# |---|---|---|---|---|---|
# Completa la fila de `clientes.csv` entera; en las otras dos, solo lo que cambia. Las tres van en Blob Storage, en
# contenedores distintos (si prefieres una tabla de Azure SQL Database para el riesgo de cada cliente, dilo en la
# columna 1). El riesgo de cada cliente es un dato personal: por eso va en un contenedor nuevo, `predicciones`, y no
# en `informes`, que la gerencia lee sin clientes uno a uno.
#
# | Activo | Servicio y redundancia | Quién lee (rol y ámbito) | Cómo se autoriza | Qué se registra y qué alerta | Costo |
# |---|---|---|---|---|---|
# | `clientes.csv` (contenedor `datos-clientes`; datos personales seudonimizados) | … | … | … | … | … |
# | El modelo entrenado (contenedor `modelos`) | … | … | … | … | … |
# | El riesgo de cada cliente (contenedor nuevo `predicciones`) | … | … | … | … | … |
#
# ---
# ### Tus respuestas
#
# *Doble clic para escribir aquí.*
#
# 1. Ejercicio 1: sin escalar, el F1 fue de …; con escalado, de … ¿Por qué cambia tanto? ¿Qué modelo de la
#    sesión 2 hizo lo mismo sin escalar? …
# 2. Ejercicio 2: con una tasa de *dropout* de … y una paciencia de …, la red paró en la época … (su mejor época
#    fue la …) y tuvo un F1 de … y una sensibilidad de … Con tasa 0: paró en la …, mejor época …, F1 … Con 0,8:
#    paró en la …, mejor época …, F1 … ¿Cuánto aporta el *dropout*? ¿Qué relación hay entre la época en que paró,
#    la mejor y la paciencia? ¿Cómo queda frente a la red grande, al MLP, a la logística y a XGBoost (celda
#    [comparacion])? ¿Le recomendarías a la gerente la red o XGBoost? …
# 3. Ejercicio 3: mi predicción fue … Con las imágenes corridas, la CNN acierta … y la densa, … ¿Qué parte de la
#    CNN la ayuda? …
# 4. La red recurrente empató con la regresión lineal. ¿Con qué serie esperarías que le ganara? …
#
# ### Autoevaluación (respóndela en el cierre de la sesión)
#
# ¿Verdadero o falso? Explica cada una.
#
# 1. «Con la clave compartida desactivada y sin acceso anónimo, el Propietario de la suscripción lee
#    `clientes.csv` aunque no tenga un rol de datos». …
# 2. «Con las imágenes corridas un píxel, la CNN conserva bastante más exactitud que una red densa de tamaño
#    parecido». …
# 3. «Para saber quién leyó `clientes.csv` ayer, la configuración de diagnóstico tenía que existir antes de la
#    lectura». …
# 4. «Con `restore_best_weights=True`, la parada temprana deja la red con los pesos de la última época». …

# %% [markdown]
# ---
# ## La respuesta a la gerencia
#
# **¿Una red neuronal encuentra mejor a los que van a cancelar?** No en esta tabla: el MLP iguala a la
# logística y XGBoost sigue al frente; una red grande sin regularizar memoriza y queda peor. Con la parada
# temprana y el *dropout*, la red mejora, pero no supera a XGBoost. Las redes valen cuando el dato es una
# imagen, un texto o una secuencia.
#
# **¿Dónde guardamos los datos en Azure, quién los lee y cómo nos enteramos?** En una cuenta de almacenamiento
# sin acceso anónimo y sin la clave compartida, con roles de datos en el ámbito más estrecho. Las lecturas se
# registran con una configuración de diagnóstico con `StorageRead` hacia Log Analytics, y una alerta avisa.
# Defender for Cloud evalúa la configuración y da una puntuación de seguridad. El costo se calcula con la página
# oficial de servicios gratuitos de Azure.

# %% cifras precalculo
# Cifras que las diapositivas y sus notas citan (no va al notebook)
import json
_cif = {"tiempo_notebook_s": round(time.time() - _t_inicio, 1)}
_cif["prueba"] = {"n": int(len(yc_prueba)), "cancelaron": int(yc_prueba.sum()),
                  "meta_70": int(np.ceil(0.70 * yc_prueba.sum()))}
_cif["columnas"] = int(Xc.shape[1])
_cif["entropia"] = {"canceló_p09": round(float(-np.log(0.9)), 3), "canceló_p01": round(float(-np.log(0.1)), 3)}
_cif["neurona"] = {"pesos": int(neurona.coef_.size), "z": round(float(z_ := (x @ neurona.coef_.ravel() + neurona.intercept_[0])[0]), 4),
                   "p": round(float(1 / (1 + np.exp(-z_))), 4)}
_cif["capa_oculta"] = {"mae_recta": round(float(mean_absolute_error(y_prueba, recta.predict(X_prueba))), 2),
                       "mae_capa": round(float(mean_absolute_error(y_prueba, capa.predict(X_prueba))), 2),
                       "iter_capa": int(capa[1].n_iter_)}
_cif["mlp"] = {"iter": int(mlp[1].n_iter_), "pesos": int(sum(c.size for c in mlp[1].coefs_) + sum(b.size for b in mlp[1].intercepts_))}
_cif["red_grande"] = {"pesos": int(grande.count_params()), "mejor_epoca": mejor,
                      "val_min": round(float(min(perdida_val)), 4), "val_final": round(float(perdida_val[-1]), 4),
                      "ent_final": round(float(perdida[-1]), 4), "ent_inicial": round(float(perdida[0]), 4),
                      "val_inicial": round(float(perdida_val[0]), 4), "n_ent": int(len(yr_ent)), "n_val": int(len(yr_val))}
_cif["tabla"] = {m: {k: (int(v) if k == "llamadas 7 de 10" else round(float(v), 4)) for k, v in fila.items()}
                 for m, fila in tabla.to_dict("index").items()}
_cif["encontrados_05"] = {m: int(((r >= 0.5) & (yc_prueba.to_numpy() == 1)).sum()) for m, r in riesgos.items()}
_cif["llamadas_05"] = {m: int((r >= 0.5).sum()) for m, r in riesgos.items()}
_cif["llamadas_azar"] = int(np.ceil(0.70 * len(yc_prueba)))
# Ejercicio 1
_cif["ej1"] = {"f1_sin": round(float(f1_sin), 4), "f1_con": round(float(f1_con), 4),
               "llamadas_sin": int(mlp_sin.predict(Xc_prueba).sum())}
# Ejercicio 2 (la solución: tasa 0.3, paciencia 10)
_vl = h.history["val_loss"]
_pr = red.predict(Xr_prueba, verbose=0).ravel()
_cif["ej2"] = {"tasa": tasa_dropout, "paciencia": paciencia, "epocas": len(_vl), "mejor_epoca": int(np.argmin(_vl)) + 1,
               "f1": round(float(f1_red), 4), "sens": round(float(sens_red), 4),
               "prec": round(float(precision_score(yc_prueba, pred_red)), 4),
               "llamadas_70": llamadas_para(_pr), "pesos": int(red.count_params())}
# Ejercicio 2 con tasa 0 y 0.8 (la misma paciencia): lo que el enunciado pide comparar
_cif["ej2_tasas"] = {}
for _t in (0.0, 0.8):
    keras.utils.set_random_seed(2026)
    _m = keras.Sequential([keras.Input(shape=(Xr_ent.shape[1],)), layers.Dense(256, activation="relu"),
                           layers.Dropout(_t), layers.Dense(256, activation="relu"), layers.Dropout(_t),
                           layers.Dense(1, activation="sigmoid")])
    _m.compile(optimizer="adam", loss="binary_crossentropy")
    _hh = _m.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val), epochs=100, batch_size=32, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    _p = _m.predict(Xr_prueba, verbose=0).ravel()
    _cif["ej2_tasas"][str(_t)] = {"epocas": len(_hh.history["loss"]), "mejor": int(np.argmin(_hh.history["val_loss"])) + 1,
                                  "f1": round(float(f1_score(yc_prueba, (_p >= 0.5).astype(int))), 4),
                                  "llamadas_70": llamadas_para(_p)}
# Dígitos y CNN
_cif["digitos"] = {"n": int(len(imagenes)), "ent": int(len(I_ent)), "prueba": int(len(I_prueba))}
_cif["cnn"] = {"pesos_cnn": int(cnn.count_params()), "pesos_densa": int(densa.count_params()),
               "cnn": round(exactitud_cnn, 4), "densa": round(exactitud_densa, 4),
               "cnn_corrida": round(corrida_cnn, 4), "densa_corrida": round(corrida_densa, 4),
               "cnn_corrida2": round(exactitud(cnn, correr(I_prueba, 2)), 4),
               "densa_corrida2": round(exactitud(densa, correr(I_prueba, 2)), 4),
               "pesos_capas_cnn": [int(c.count_params()) for c in cnn.layers],
               "pesos_capas_densa": [int(c.count_params()) for c in densa.layers]}
_cif["filtro"] = {"min": float(mapa.min()), "max": float(mapa.max())}
# Serie
_cif["serie"] = {"horas": int(len(trafico)), "min": round(float(trafico.min()), 1), "max": round(float(trafico.max()), 1),
                 "media": round(float(trafico.mean()), 1), "ventanas_ent": int(len(ys_ent)), "ventanas_prueba": int(len(ys_prueba)),
                 "dias_prueba": round(len(ys_prueba) / 24, 1)}
_cif["errores_serie"] = {k: round(float(v), 3) for k, v in errores.items()}
_cif["rnn_pesos"] = int(rnn.count_params())
print(json.dumps(_cif, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# %% semillas precalculo
# Las cifras de Keras cambian con la semilla y entre equipos. Aquí se repite, con 12 semillas (como 12 equipos),
# lo que los verificadores revisan y lo que el texto dice «cerca de»: la solución del ejercicio 2 con tres tasas
# y paciencias debe recibir «¡Listo!» del verificador, y la del ejercicio 3 también. Si algo falla, ejecuta se
# detiene: hay que ajustar el rango o el texto. Los rangos que resultan se citan en las notas («rangos»).
import contextlib
import io
_t_semillas = time.time()
_res, _rangos = {}, {"f1": [], "sens": [], "llamadas_red": [], "grande_mejor": [], "grande_llamadas": [],
                     "cnn": [], "densa": [], "cnn_corrida": [], "densa_corrida": [], "rnn": []}

def _salida(funcion, **argumentos):
    """Lo que imprime un verificador."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        funcion(**argumentos)
    return buffer.getvalue()

for _s in (2026, 1, 3, 7, 11, 13, 37, 42, 99, 123, 555, 1000):
    for _tasa, _pac in ((0.3, 10), (0.2, 5), (0.5, 15)):
        keras.utils.set_random_seed(_s)
        _m = keras.Sequential([keras.Input(shape=(Xr_ent.shape[1],)), layers.Dense(256, activation="relu"),
                               layers.Dropout(_tasa), layers.Dense(256, activation="relu"), layers.Dropout(_tasa),
                               layers.Dense(1, activation="sigmoid")])
        _m.compile(optimizer="adam", loss="binary_crossentropy")
        _hh = _m.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val), epochs=100, batch_size=32, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=_pac,
                                                              restore_best_weights=True)])
        _pr = _m.predict(Xr_prueba, verbose=0).ravel()
        _p = (_pr >= 0.5).astype(int)
        _f, _se = f1_score(yc_prueba, _p), recall_score(yc_prueba, _p)
        _rangos["f1"].append(_f); _rangos["sens"].append(_se); _rangos["llamadas_red"].append(llamadas_para(_pr))
        _txt = _salida(revisar_red, red=_m, historial=_hh, f1=_f, sensibilidad=_se)
        assert "¡Listo!" in _txt, ("ej2 sin ¡Listo!", _s, _tasa, _pac, _txt)
    # la red grande sin regularizar: la validación toca fondo pronto, termina muy arriba y pide más llamadas
    keras.utils.set_random_seed(_s)
    _g = keras.Sequential([keras.Input(shape=(Xr_ent.shape[1],)), layers.Dense(256, activation="relu"),
                           layers.Dense(256, activation="relu"), layers.Dense(1, activation="sigmoid")])
    _g.compile(optimizer="adam", loss="binary_crossentropy")
    _vg = _g.fit(Xr_ent, yr_ent, validation_data=(Xr_val, yr_val), epochs=40, batch_size=32,
                 verbose=0).history["val_loss"]
    _lg = llamadas_para(_g.predict(Xr_prueba, verbose=0).ravel())
    _rangos["grande_mejor"].append(int(np.argmin(_vg)) + 1); _rangos["grande_llamadas"].append(_lg)
    assert np.argmin(_vg) <= 3 and _vg[-1] > 2 * min(_vg), ("la red grande no memoriza como dice el texto", _s)
    assert _lg > max(_cif["tabla"]["Logística"]["llamadas 7 de 10"], _cif["tabla"]["MLP (scikit-learn)"]["llamadas 7 de 10"]), ("grande", _s)
    # CNN y densa: el verificador del ejercicio 3 debe decir «¡Listo!»
    keras.utils.set_random_seed(_s)
    _c = keras.models.clone_model(cnn)
    _c.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    _c.fit(I_ent, d_ent, epochs=30, batch_size=32, verbose=0)
    keras.utils.set_random_seed(_s)
    _d = keras.models.clone_model(densa)
    _d.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    _d.fit(I_ent, d_ent, epochs=30, batch_size=32, verbose=0)
    _v = [exactitud(_c, I_prueba), exactitud(_d, I_prueba), exactitud(_c, I_corridas), exactitud(_d, I_corridas)]
    for _k, _x in zip(("cnn", "densa", "cnn_corrida", "densa_corrida"), _v):
        _rangos[_k].append(_x)
    _txt = _salida(revisar_corrimiento, cnn=_c, densa=_d, exactitud_cnn=_v[0], exactitud_densa=_v[1],
                   corrida_cnn=_v[2], corrida_densa=_v[3])
    assert "¡Listo!" in _txt, ("ej3 sin ¡Listo!", _s, _txt)
    # RNN: le gana a las dos primeras líneas base y queda cerca de la lineal
    keras.utils.set_random_seed(_s)
    _r = keras.models.clone_model(rnn)
    _r.compile(optimizer="adam", loss="mse")
    _r.fit(((Xs_ent - media) / desv)[..., None], (ys_ent - media) / desv, epochs=40, batch_size=32, verbose=0)
    _er = mean_absolute_error(ys_prueba, _r.predict(((Xs_prueba - media) / desv)[..., None], verbose=0).ravel()
                              * desv + media)
    _rangos["rnn"].append(_er)
    assert _er < errores["La última hora"] and abs(_er - errores["Regresión lineal (24 h)"]) < 0.4, ("RNN", _s)
_res["rangos"] = {k: [round(float(min(v)), 3), round(float(max(v)), 3)] for k, v in _rangos.items()}
_res["casos_listo"] = {"ej2": 36, "ej3": 12}
_res["tiempo_s"] = round(time.time() - _t_semillas, 1)
print(json.dumps(_res, ensure_ascii=False))
