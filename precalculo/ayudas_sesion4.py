"""Sesión 4: estilo de las gráficas del curso y los verificadores de la práctica.

Van dentro de la celda oculta del notebook, junto con descarga_datos.py y ayudas.py. Cada
verificador recibe lo que el estudiante calculó, dice qué está bien y, en lo demás, da una
pista sin la respuesta. Nunca se cae: un «...» sin reemplazar o un tipo inesperado se avisa.

Keras se importa de forma perezosa: este archivo no lo importa (lo importa el estudiante en la
celda [keras], después de fijar el backend). Los verificadores de Keras NO reentrenan: usan la
red que el estudiante entrenó (solo predicen) y comparan lo que entregó con lo que esa red da.
Como Keras puede variar entre equipos, lo que se exige de la red son rangos, que la celda
[semillas] de celdas_sesion4.py comprueba con otras semillas al construir el material:

- ejercicio 1 (scikit-learn, determinista): los F1 del MLP de 16 y 8 neuronas, sin escalar y con
  escalado, calculados aquí con la misma semilla de la clase; tolerancia ±0,005 con escalado y ±0,02
  sin escalar (todos los F1 sin escalar son casi 0); las pistas exactas (±0,001) se prueban antes;
- ejercicio 2: el F1 de prueba de la red del estudiante en _RANGO_F1_RED y su sensibilidad en
  _RANGO_SENS_RED (rangos del barrido de 24 semillas del revisor 4: ninguna de 264 redes razonables
  queda fuera); lo entregado debe coincidir (±0,006, para aceptar el redondeo a dos decimales) con lo
  que da su propia red; se mira el historial (si paró, con qué paciencia, si volvió a los mejores
  pesos, comparando la pérdida de la red en la validación con la mínima y con la última del historial);
- ejercicio 3: con sus propias redes, la exactitud normal ≥ _MINIMO_NORMAL y, con las imágenes
  corridas un píxel, la CNN por encima de la densa en más de _MARGEN_CORRIDA.
"""

_RANGO_F1_RED = (0.38, 0.62)
_RANGO_SENS_RED = (0.25, 0.65)
_MINIMO_NORMAL = 0.93
_MARGEN_CORRIDA = 0.15


def _estilo_graficas():
    """Colores del curso y letras que se leen en Colab y proyectadas."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=["#8A2E9E", "#F46201", "#566B7F"]),
        "figure.figsize": (6.4, 3.6),
        "figure.constrained_layout.use": True,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#1E293B",
    })


_estilo_graficas()

_CACHE = {}   # lo que ya se calculó: los datos preparados y los modelos de referencia


def _clientes():
    """Las particiones de la clase (la de la sesión 2 y la validación de Keras), o None con aviso."""
    if "clientes" in _CACHE:
        return _CACHE["clientes"]
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        clientes = pd.read_csv("datos/clientes.csv")
    except FileNotFoundError:
        print("✗  No encuentro datos/clientes.csv: ejecuta primero la celda que descarga los datos "
              "(Entorno de ejecución → Ejecutar anteriores)")
        return None
    Xc = pd.get_dummies(clientes.drop(columns=["id_cliente", "cancelo"]), drop_first=True, dtype=int)
    yc = (clientes["cancelo"] == "Sí").astype(int)
    Xc_e, Xc_p, yc_e, yc_p = train_test_split(Xc, yc, test_size=0.25, random_state=2026, stratify=yc)
    Xc_e2, Xc_v, yc_e2, yc_v = train_test_split(Xc_e, yc_e, test_size=0.2, random_state=2026, stratify=yc_e)
    escala = StandardScaler().fit(Xc_e2)
    _CACHE["clientes"] = {
        "Xc_e": Xc_e, "Xc_p": Xc_p, "yc_e": yc_e, "yc_p": yc_p,
        "Xr_e": escala.transform(Xc_e2).astype("float32"), "Xr_v": escala.transform(Xc_v).astype("float32"),
        "Xr_p": escala.transform(Xc_p).astype("float32"),
        "yr_e": yc_e2.to_numpy(), "yr_v": yc_v.to_numpy(), "yr_p": yc_p.to_numpy(),
    }
    return _CACHE["clientes"]


def _digitos():
    """Las imágenes de la clase: entrenamiento y prueba, como en la celda [digitos]."""
    if "digitos" in _CACHE:
        return _CACHE["digitos"]
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    imagenes = (d.images / 16).astype("float32")[..., None]
    I_e, I_p, d_e, d_p = train_test_split(imagenes, d.target, test_size=0.25, random_state=2026,
                                          stratify=d.target)
    _CACHE["digitos"] = {"I_e": I_e, "I_p": I_p, "d_e": d_e, "d_p": d_p}
    return _CACHE["digitos"]


def _sobrantes(sueltos, otros, nombres, ejemplo):
    """Avisa de valores sin nombre o con otro nombre. True si hubo alguno."""
    if sueltos:
        print(f"✗  Pasa los valores por nombre, como en la plantilla: {ejemplo}")
        return True
    if otros:
        print(f"✗  No conozco {', '.join(repr(k) for k in otros)}: usa estos nombres, en "
              f"minúsculas y sin tildes: {', '.join(n + '=' for n in nombres)}")
        return True
    return False


def _seguro(verificador):
    """Red de seguridad: si algo imprevisto falla al revisar, se avisa sin traceback."""
    import functools

    @functools.wraps(verificador)
    def envoltura(*args, **kwargs):
        try:
            return verificador(*args, **kwargs)
        except Exception as error:   # noqa: BLE001 — el estudiante nunca ve un traceback
            print(f"✗  No pude revisar lo que entregaste ({type(error).__name__}). Revisa que cada valor "
                  "sea el resultado de un cálculo (un número) y vuelve a ejecutar la celda")
    return envoltura


def _un_numero(x):
    """(valor, None) si x es un número (o algo con un solo número); si no, (None, motivo)."""
    import math
    import numbers
    import numpy as np
    import pandas as pd

    if x is Ellipsis:
        return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo"
    if isinstance(x, pd.DataFrame):
        if x.size != 1:
            return None, f"es una tabla de {x.shape[0]} × {x.shape[1]}; hace falta un solo número"
        x = x.iloc[0, 0]
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)) or (
            hasattr(x, "shape") and hasattr(x, "numpy") and getattr(x, "shape", ()) != ()):
        arreglo = np.asarray(x, dtype=object)
        valores = list(np.ravel(arreglo))
        if len(valores) == 0:
            return None, "está vacío"
        if len(valores) > 1:
            if len(valores) in (450, 2000, 4800, 6000):
                return None, ("es una predicción por cliente o por imagen; falta resumirla en un número "
                              "(una métrica)")
            return None, f"tiene {len(valores)} valores; hace falta uno solo"
        x = valores[0]
        if x is Ellipsis:
            return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo"
    if hasattr(x, "numpy") and not isinstance(x, (numbers.Real, np.number)):
        try:
            x = x.numpy()   # un tensor de un solo número
        except Exception:   # noqa: BLE001
            pass
    if x is None:
        return None, "está vacío (None): ¿le asignaste el resultado de un print?"
    if callable(x):
        return None, "es una función o un modelo, no su resultado: ¿faltó calcular la métrica?"
    if isinstance(x, (bool, np.bool_)):
        return None, "es True o False; hace falta un número"
    if isinstance(x, str):
        return None, "es texto; escríbelo como número, sin comillas y con punto decimal"
    if not isinstance(x, (numbers.Real, np.number)) or isinstance(x, np.complexfloating):
        return None, f"no es un número real (es {type(x).__name__})"
    try:
        valor = float(x)
    except (OverflowError, TypeError, ValueError):
        return None, "no es un número que se pueda revisar"
    if math.isnan(valor):
        return None, "salió nan: revisa el cálculo"
    if math.isinf(valor):
        return None, "salió infinito: ¿dividiste entre cero?"
    return valor, None


def _cerca(a, b, tolerancia):
    return a is not None and b is not None and abs(a - b) <= tolerancia


def _coma(x, decimales=3):
    """Un número con coma decimal, para los mensajes."""
    return f"{x:.{decimales}f}".replace(".", ",").replace("-", "−")


def _leer(entregados, nombres):
    """Convierte cada valor entregado en número; imprime los problemas. Devuelve {nombre: valor}."""
    valores, problemas = {}, {}
    for nombre in nombres:
        x = entregados[nombre]
        if x is Ellipsis:
            print(f"·  {nombre}: todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo")
            continue
        valor, problema = _un_numero(x)
        if problema:
            problemas.setdefault(problema, []).append(nombre)
        elif not 0 <= valor <= 1:
            if 1 < valor <= 100:
                problemas.setdefault("es una proporción entre 0 y 1, no un porcentaje", []).append(nombre)
            else:
                problemas.setdefault("estas métricas van de 0 a 1", []).append(nombre)
        else:
            valores[nombre] = valor
    for problema, quienes in problemas.items():   # la misma pista una sola vez
        print(f"✗  {', '.join(quienes)}: {problema}")
    return valores


def _es_red(x):
    """¿Es un modelo de Keras (algo con capas y predict)?"""
    return x is not Ellipsis and hasattr(x, "predict") and hasattr(x, "layers")


def _capas(red):
    return [type(c).__name__ for c in red.layers]


def _llamadas_para(riesgo, y_real, meta=0.70):
    """Llamadas, de mayor a menor riesgo, para encontrar a la meta (la función de la clase)."""
    import numpy as np
    for corte in np.sort(np.unique(riesgo))[::-1]:
        llamados = riesgo >= corte
        if y_real[llamados].sum() >= meta * y_real.sum():
            return int(llamados.sum())


# ---------------------------------------------------------------------------------------------
# Ejercicio 1 · Escalar importa (scikit-learn: se puede calcular aquí, en una fracción de segundo)
# ---------------------------------------------------------------------------------------------

def _referencia_escalado():
    """F1, exactitud y sensibilidad de los MLP de la clase, sin escalar y con escalado."""
    if "escalado" in _CACHE:
        return _CACHE["escalado"]
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = _clientes()
    ref = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for capas in [(16, 8), (32,), (16,), (8,)]:
            for escalado in (False, True):
                mlp = MLPClassifier(hidden_layer_sizes=capas, early_stopping=True, random_state=2026)
                modelo = make_pipeline(StandardScaler(), mlp) if escalado else mlp
                modelo.fit(d["Xc_e"], d["yc_e"])
                pp, pe = modelo.predict(d["Xc_p"]), modelo.predict(d["Xc_e"])
                ref[(capas, escalado)] = {
                    "f1": f1_score(d["yc_p"], pp, zero_division=0),
                    "exactitud": accuracy_score(d["yc_p"], pp),
                    "sensibilidad": recall_score(d["yc_p"], pp, zero_division=0),
                    "precision": precision_score(d["yc_p"], pp, zero_division=0),
                    "f1_entrena": f1_score(d["yc_e"], pe, zero_division=0)}
    _CACHE["escalado"] = ref
    return ref


@_seguro
def revisar_escalado(*sueltos, f1_sin=..., f1_con=..., **otros):
    """Ejercicio 1: el F1 en prueba del MLP de 16 y 8 neuronas, sin escalar y con escalado."""
    nombres = ["f1_sin", "f1_con"]
    if _sobrantes(sueltos, otros, nombres, "revisar_escalado(f1_sin=f1_sin, f1_con=f1_con)"):
        return
    if _clientes() is None:
        return
    vals = _leer({"f1_sin": f1_sin, "f1_con": f1_con}, nombres)
    if not vals:
        return
    ref = _referencia_escalado()
    tol = 0.02
    bien = {"f1_sin": ref[((16, 8), False)], "f1_con": ref[((16, 8), True)]}
    if ("f1_sin" in vals and "f1_con" in vals and _cerca(vals["f1_sin"], bien["f1_con"]["f1"], tol)
            and _cerca(vals["f1_con"], bien["f1_sin"]["f1"], tol)
            and not _cerca(vals["f1_sin"], bien["f1_sin"]["f1"], tol)):
        print("✗  f1_sin y f1_con están cruzados: el de mlp_sin va en f1_sin y el de mlp_con, en f1_con")
        return
    ok = {}
    for nombre in nombres:
        if nombre not in vals:
            continue
        v, r = vals[nombre], bien[nombre]
        escalado = nombre == "f1_con"
        exacto = _cerca(v, r["f1"], 0.001 if escalado else 0.02)
        pista = None
        if exacto:
            pass
        elif escalado and _cerca(v, bien["f1_sin"]["f1"], 0.005):
            pista = ("salió como el del modelo sin escalar: ¿pusiste StandardScaler() como primer paso del "
                     "make_pipeline y mediste con mlp_con? Recuerda la SVM de la sesión 2 sin escalar")
        elif _cerca(v, r["exactitud"], 0.001):
            pista = "esa es la exactitud; aquí va el F1: f1_score(yc_prueba, …)"
        elif _cerca(v, r["sensibilidad"], 0.001):
            pista = "esa es la sensibilidad; aquí va el F1: f1_score(yc_prueba, …)"
        elif _cerca(v, r["precision"], 0.001):
            pista = "esa es la precisión; aquí va el F1: f1_score(yc_prueba, …)"
        elif _cerca(v, r["f1_entrena"], 0.001):
            pista = "ese es el F1 en el entrenamiento; se mide con la prueba (Xc_prueba y yc_prueba)"
        else:
            for capas in [(32,), (16,), (8,)]:
                if _cerca(v, ref[(capas, escalado)]["f1"], 0.001):
                    pista = ("es el de otra red: deja hidden_layer_sizes=(16, 8), dos capas de 16 y 8 neuronas")
                    break
        ok[nombre] = pista is None and _cerca(v, r["f1"], 0.005 if escalado else 0.02)
        if ok[nombre]:
            print(f"✓  {nombre}: bien")
            continue
        print(f"✗  {nombre}: {pista or 'no coincide; revisa que entrenes con Xc_ent, midas el F1 con Xc_prueba y dejes random_state=2026'}")
    if ok.get("f1_sin") and ok.get("f1_con"):
        s, c = bien["f1_sin"], bien["f1_con"]
        print(f"\n¡Listo! Sin escalar, el MLP tiene un F1 de {_coma(s['f1'])} y encuentra al "
              f"{_coma(s['sensibilidad'] * 100, 1)} % de los que cancelan; con escalado, el F1 sube a "
              f"{_coma(c['f1'])}. Explica en «Tus respuestas» por qué.")


# ---------------------------------------------------------------------------------------------
# Ejercicio 2 · Que no memorice (Keras: se usa la red del estudiante, sin reentrenar)
# ---------------------------------------------------------------------------------------------

def _entropia(y, p):
    """Entropía cruzada binaria media, como la calcula Keras (con el mismo recorte)."""
    import numpy as np
    p = np.clip(np.asarray(p, dtype="float64").ravel(), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype="float64").ravel()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@_seguro
def revisar_red(*sueltos, red=..., historial=..., f1=..., sensibilidad=..., **otros):
    """Ejercicio 2: la red con dropout y parada temprana; su F1 y su sensibilidad en la prueba."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    nombres = ["red", "historial", "f1", "sensibilidad"]
    if _sobrantes(sueltos, otros, nombres, "revisar_red(red=red, historial=h, f1=f1_red, "
                                           "sensibilidad=sens_red)"):
        return
    if not _es_red(red):
        print("✗  red: pasa la red de Keras que entrenaste: red=red")
        return
    vl = None
    if isinstance(historial, dict):
        hist, epocas_max = historial, None
    elif hasattr(historial, "history") and isinstance(historial.history, dict):
        hist, epocas_max = historial.history, getattr(historial, "params", {}).get("epochs")
    else:
        print("✗  historial: pasa lo que devuelve fit, el historial del entrenamiento: historial=h")
        return
    if "val_loss" not in hist:
        print("✗  historial: no tiene pérdida de validación; entrena con validation_data=(Xr_val, yr_val)")
        return
    vl = list(hist["val_loss"])
    d = _clientes()
    if d is None:
        return
    try:
        prob = np.asarray(red.predict(d["Xr_p"], verbose=0)).ravel()
        prob_v = np.asarray(red.predict(d["Xr_v"], verbose=0)).ravel()
    except Exception as error:   # noqa: BLE001
        print(f"✗  red: no pude usarla con los clientes de prueba ({type(error).__name__}): ¿la entrada es "
              "keras.Input(shape=(Xr_ent.shape[1],))?")
        return
    if prob.size != len(d["yr_p"]):
        print("✗  red: debe dar una sola probabilidad por cliente: la última capa debe ser Dense(1, activation=\"sigmoid\")")
        return
    capas = _capas(red)
    tasas = [float(c.rate) for c in red.layers if type(c).__name__ == "Dropout"]
    # 1. ¿Hay dropout y parada temprana?
    if not tasas:
        print("·  La red no tiene capas Dropout: usa la arquitectura de la plantilla")
    elif max(tasas) == 0:
        print("·  Con una tasa de dropout de 0, no se apaga ninguna neurona: prueba entre 0.2 y 0.5 (con punto, como en el código)")
    elif max(tasas) >= 0.8:
        print(f"·  Con una tasa de dropout de {_coma(max(tasas), 2)} se apaga casi toda la red en cada paso: prueba entre 0.2 "
              "y 0.5 (con punto, como en el código)")
    n, mejor = len(vl), int(np.argmin(vl))
    paro = epocas_max is None or n < epocas_max
    if not paro:
        print(f"·  La red no paró: corrió las {n} épocas. Con restore_best_weights=True vuelve igual a su mejor "
              f"época (la {mejor + 1}), pero esperó de más: baja la paciencia")
    elif n - 1 - mejor <= 1 and n > 2:
        print("·  Paró en la primera época que no mejoró (paciencia 0 o 1): el ruido de la validación puede "
              "engañarla; prueba con una paciencia de 3 a 15")
    # 2. ¿Volvió a los mejores pesos? Con restore_best_weights=True, la pérdida de la red en la validación
    # es la mínima del historial (difieren ~1e-8); sin restaurar, es la de la última época.
    actual_v = _entropia(d["yr_v"], prob_v)
    d_min, d_ult = abs(actual_v - min(vl)), abs(actual_v - vl[-1])
    restaurada = d_min <= d_ult or vl[-1] - min(vl) < 1e-3
    if min(d_min, d_ult) > 5e-3:
        restaurada = False
        print("·  El historial no coincide con esta red en la validación: ¿es el h de esta misma red, entrenada "
              "con validation_data=(Xr_val, yr_val)?")
    elif not restaurada:
        print(f"✗  Tu red se quedó con los pesos de la última época (pérdida de validación {_coma(vl[-1])}) y no "
              f"con los de la mejor ({_coma(min(vl))}): ¿pusiste restore_best_weights=True en EarlyStopping?")
    # 3. Lo entregado frente a lo que da su propia red
    vals = _leer({"f1": f1, "sensibilidad": sensibilidad}, ["f1", "sensibilidad"])
    pred = (prob >= 0.5).astype(int)
    y = d["yr_p"]
    real = {"f1": f1_score(y, pred, zero_division=0), "sensibilidad": recall_score(y, pred, zero_division=0)}
    otros_val = {"exactitud": accuracy_score(y, pred), "precisión": precision_score(y, pred, zero_division=0)}
    pred_v = (prob_v >= 0.5).astype(int)
    ok = {}
    for nombre, v in vals.items():
        pista = None
        for otro, valor in [*otros_val.items(), *[(k, real[k]) for k in real if k != nombre]]:
            if not _cerca(v, real[nombre], 0.001) and _cerca(v, valor, 0.001):
                articulo = {"exactitud": "la exactitud", "precisión": "la precisión", "f1": "el F1",
                            "sensibilidad": "la sensibilidad"}[otro]
                pista = f"ese valor es {articulo} de tu red; aquí va {'el F1' if nombre == 'f1' else 'la sensibilidad'}"
                break
        if pista is None and not _cerca(v, real[nombre], 0.001) and _cerca(
                v, (f1_score if nombre == "f1" else recall_score)(d["yr_v"], pred_v, zero_division=0), 0.001):
            pista = "lo mediste con la validación; aquí va la prueba (yc_prueba y pred_red)"
        ok[nombre] = pista is None and _cerca(v, real[nombre], 0.006)
        if ok[nombre]:
            print(f"✓  {nombre}: coincide con tu red")
            continue
        print(f"✗  {nombre}: {pista or 'no coincide con lo que da tu red en la prueba: calcula con yc_prueba y pred_red'}")
    # 4. ¿Está la red donde se espera?
    f1_real, s_real = real["f1"], real["sensibilidad"]
    dentro = (_RANGO_F1_RED[0] <= f1_real <= _RANGO_F1_RED[1]
              and _RANGO_SENS_RED[0] <= s_real <= _RANGO_SENS_RED[1])
    if not dentro:
        print(f"·  Tu red tiene en la prueba un F1 de {_coma(f1_real)} y una sensibilidad de {_coma(s_real)}, fuera de "
              f"lo que esperamos (F1 entre {_coma(_RANGO_F1_RED[0], 2)} y {_coma(_RANGO_F1_RED[1], 2)}). Las redes de Keras "
              "cambian un poco de un equipo a otro: vuelve a ejecutar la celda; si sigue igual, prueba una tasa de dropout "
              "entre 0.2 y 0.5 (con punto, como en el código) y una paciencia entre 3 y 15")
    elif all(ok.get(k) for k in ("f1", "sensibilidad")) and tasas and 0 < max(tasas) < 0.8 and restaurada:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if "llamadas_log" not in _CACHE:
            log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d["Xc_e"], d["yc_e"])
            _CACHE["llamadas_log"] = _llamadas_para(log.predict_proba(d["Xc_p"])[:, 1], y)
        llamadas = _llamadas_para(prob, y)
        print(f"\n¡Listo! Tu red {'paró en' if paro else 'corrió hasta'} la época {n} (la mejor fue la {mejor + 1}) y volvió a esos pesos. En la prueba: "
              f"F1 de {_coma(f1_real)} y sensibilidad de {_coma(s_real)}. Para encontrar a 7 de cada 10 necesita "
              f"{llamadas} llamadas; la logística, {_CACHE['llamadas_log']}. Compárala en «Tus respuestas» con la "
              "tabla de la celda [comparacion].")


# ---------------------------------------------------------------------------------------------
# Ejercicio 3 · Correr un píxel (Keras: se usan las redes del estudiante, sin reentrenar)
# ---------------------------------------------------------------------------------------------

def _correr(imagenes, pixeles=1):
    import numpy as np
    corridas = np.zeros_like(imagenes)
    corridas[:, :, pixeles:] = imagenes[:, :, :imagenes.shape[2] - pixeles]
    return corridas


def _exactitud(modelo, imagenes, etiquetas):
    import numpy as np
    return float((np.asarray(modelo.predict(imagenes, verbose=0)).argmax(axis=1) == etiquetas).mean())


@_seguro
def revisar_corrimiento(*sueltos, cnn=..., densa=..., exactitud_cnn=..., exactitud_densa=...,
                        corrida_cnn=..., corrida_densa=..., **otros):
    """Ejercicio 3: la exactitud de la CNN y de la densa, con las imágenes como vienen y corridas un píxel."""
    nombres = ["cnn", "densa", "exactitud_cnn", "exactitud_densa", "corrida_cnn", "corrida_densa"]
    if _sobrantes(sueltos, otros, nombres, "revisar_corrimiento(cnn=cnn, densa=densa, exactitud_cnn=…, "
                                           "exactitud_densa=…, corrida_cnn=…, corrida_densa=…)"):
        return
    for nombre, x in (("cnn", cnn), ("densa", densa)):
        if not _es_red(x):
            print(f"✗  {nombre}: pasa la red de Keras de la clase: {nombre}={nombre}")
            return
    if "Conv2D" not in _capas(cnn) and "Conv2D" in _capas(densa):
        print("✗  cnn y densa están cruzadas: la CNN es la que tiene capas Conv2D")
        return
    if "Conv2D" not in _capas(cnn):
        print("✗  cnn: esa red no tiene capas Conv2D; pasa la CNN de la celda [cnn]")
        return
    if "Conv2D" in _capas(densa):
        print("✗  densa: esa red tiene capas Conv2D; pasa la red densa de la celda [cnn]")
        return
    d = _digitos()
    I_p, d_p, I_e, d_e = d["I_p"], d["d_p"], d["I_e"], d["d_e"]
    corr = _correr(I_p)
    real = {"exactitud_cnn": _exactitud(cnn, I_p, d_p), "exactitud_densa": _exactitud(densa, I_p, d_p),
            "corrida_cnn": _exactitud(cnn, corr, d_p), "corrida_densa": _exactitud(densa, corr, d_p)}
    alternativas = {
        "con las imágenes corridas": {"exactitud_cnn": real["corrida_cnn"], "exactitud_densa": real["corrida_densa"]},
        "con las imágenes como vienen (I_prueba)": {"corrida_cnn": real["exactitud_cnn"],
                                                    "corrida_densa": real["exactitud_densa"]},
        "con las imágenes del entrenamiento": {"exactitud_cnn": _exactitud(cnn, I_e, d_e),
                                               "exactitud_densa": _exactitud(densa, I_e, d_e)},
        "con las imágenes corridas dos píxeles": {"corrida_cnn": _exactitud(cnn, _correr(I_p, 2), d_p),
                                                  "corrida_densa": _exactitud(densa, _correr(I_p, 2), d_p)},
    }
    vals = _leer({k: v for k, v in zip(nombres[2:], (exactitud_cnn, exactitud_densa, corrida_cnn,
                                                      corrida_densa))}, nombres[2:])
    ok = {}
    for nombre, v in vals.items():
        pista = None
        for como, valores in alternativas.items():
            if nombre in valores and _cerca(v, valores[nombre], 0.001) and not _cerca(v, real[nombre], 0.001):
                pista = f"ese valor sale de medir {como}"
                break
        if pista is None and not _cerca(v, real[nombre], 0.001):
            cruzado = nombre.replace("cnn", "X").replace("densa", "cnn").replace("X", "densa")
            if _cerca(v, real[cruzado], 0.001):
                pista = "ese valor es el de la otra red"
        ok[nombre] = pista is None and _cerca(v, real[nombre], 0.006)
        if ok[nombre]:
            print(f"✓  {nombre}: coincide con tu red")
            continue
        print(f"✗  {nombre}: {pista or 'no coincide con lo que dan tus redes: usa exactitud(red, imágenes)'}")
    if len(ok) == 4 and all(ok.values()):
        ec, ed, cc, cd = (real[k] for k in nombres[2:])
        if min(ec, ed) < _MINIMO_NORMAL or cc - cd <= _MARGEN_CORRIDA:
            print(f"·  Tus cifras están bien calculadas, pero tus redes salieron distintas de lo esperado (CNN "
                  f"{_coma(ec)} y {_coma(cc)}; densa {_coma(ed)} y {_coma(cd)}). Vuelve a ejecutar las celdas "
                  "[cnn] y [exactitud] y repite")
            return
        print(f"\n¡Listo! Con las imágenes como vienen, la CNN tiene una exactitud de {_coma(ec)} y la densa, de {_coma(ed)}. "
              f"Corridas un píxel, la CNN conserva {_coma(cc)} y la densa cae a {_coma(cd)}. Explica en «Tus respuestas» qué "
              "parte de la CNN la ayuda.")
