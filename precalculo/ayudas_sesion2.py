"""Sesión 2: estilo de las gráficas del curso y los verificadores de la práctica.

Van dentro de la celda oculta del notebook, junto con descarga_datos.py y ayudas.py. Cada
verificador recibe lo que el estudiante calculó, dice qué está bien y, en lo demás, da una
pista sin la respuesta. Nunca se cae: un «...» sin reemplazar o un tipo inesperado se avisa.

Los valores esperados se calculan aquí, en tiempo de ejecución, desde datos/clientes.csv, con
la misma preparación y la misma semilla de la clase (y se guardan para no repetir el cálculo).
Tolerancias: ejercicio 1, ±0,5 GB en el MAE y ±0,005 en cada R²; ejercicio 2, conteos (con ±2
de tolerancia por si otra versión de scikit-learn cambia un empate); ejercicio 3, el umbral exacto de la
lista y ±2 llamadas; ejercicio 4, la profundidad exacta y ±0,005 en el F1 medio.
"""


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


def _datos():
    """Las particiones de la clase (regresión y clasificación), o None con aviso."""
    if "datos" in _CACHE:
        return _CACHE["datos"]
    import pandas as pd
    from sklearn.model_selection import train_test_split

    try:
        clientes = pd.read_csv("datos/clientes.csv")
    except FileNotFoundError:
        print("✗  No encuentro datos/clientes.csv: ejecuta primero la celda que descarga los datos "
              "(Entorno de ejecución → Ejecutar anteriores)")
        return None
    X = pd.get_dummies(clientes.drop(columns=["id_cliente", "consumo_gb", "cancelo"]),
                       drop_first=True, dtype=int)
    y = clientes["consumo_gb"]
    reg = train_test_split(X, y, test_size=0.25, random_state=2026)
    Xc = pd.get_dummies(clientes.drop(columns=["id_cliente", "cancelo"]), drop_first=True, dtype=int)
    yc = (clientes["cancelo"] == "Sí").astype(int)
    clf = train_test_split(Xc, yc, test_size=0.25, random_state=2026, stratify=yc)
    _CACHE["datos"] = {"reg": reg, "clf": clf}
    return _CACHE["datos"]


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
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        arreglo = np.asarray(x, dtype=object)
        if arreglo.shape == (2, 2):
            return None, ("es la matriz de confusión entera: saca sus cuatro números con .ravel() "
                          "(en orden VN, FP, FN, VP)")
        valores = list(np.ravel(arreglo))
        if len(valores) == 0:
            return None, "está vacío"
        if len(valores) > 1:
            if len(valores) in (2000, 6000):
                return None, ("es una predicción por cliente; falta resumirla en un número "
                              "(una métrica o un conteo)")
            return None, f"tiene {len(valores)} valores; hace falta uno solo"
        x = valores[0]
        if x is Ellipsis:
            return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo"
    if x is None:
        return None, "está vacío (None): ¿le asignaste el resultado de un print?"
    if callable(x):
        return None, "es una función, no su resultado: ¿faltaron los paréntesis o los datos?"
    if isinstance(x, (bool, np.bool_)):
        return None, "es True o False; hace falta un número"
    if isinstance(x, str):
        return None, "es texto; escríbelo como número, sin comillas y con punto decimal"
    if hasattr(x, "fit") and hasattr(x, "get_params"):
        return None, "es un modelo, no un número: ¿te faltó calcular la métrica?"
    if not isinstance(x, (numbers.Real, np.number)) or isinstance(x, np.complexfloating):
        return None, f"no es un número real (es {type(x).__name__})"
    try:
        valor = float(x)
    except OverflowError:
        return None, "es un número demasiado grande para ser una métrica o un conteo"
    if math.isnan(valor):
        return None, "salió nan: revisa el cálculo"
    if math.isinf(valor):
        return None, "salió infinito: ¿dividiste entre cero?"
    return valor, None


def _cerca(a, b, tolerancia):
    return a is not None and b is not None and abs(a - b) <= tolerancia


def _niveles(p):
    """«1 nivel», «6 niveles»."""
    return f"{p} nivel" if p == 1 else f"{p} niveles"


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
        else:
            valores[nombre] = valor
    for problema, quienes in problemas.items():   # la misma pista una sola vez
        print(f"✗  {', '.join(quienes)}: {problema}")
    return valores


# ---------------------------------------------------------------------------------------------
# Ejercicio 1 · Un árbol con límite (regresión)
# ---------------------------------------------------------------------------------------------

def _referencia_arbol():
    """MAE, RMSE, MSE y R² de los árboles de 1 a 15 niveles y del sin límite, y de la clase."""
    if "arbol" in _CACHE:
        return _CACHE["arbol"]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor

    X_e, X_p, y_e, y_p = _datos()["reg"]
    ref = {}
    for p in list(range(1, 16)) + [None]:
        m = DecisionTreeRegressor(max_depth=p, random_state=2026).fit(X_e, y_e)
        pe, pp = m.predict(X_e), m.predict(X_p)
        ref[p] = {"mae": mean_absolute_error(y_p, pp), "mae_entrena": mean_absolute_error(y_e, pe),
                  "mse": mean_squared_error(y_p, pp), "rmse": mean_squared_error(y_p, pp) ** 0.5,
                  "r2_entrenamiento": r2_score(y_e, pe), "r2_prueba": r2_score(y_p, pp),
                  "r2_al_reves": r2_score(pp, y_p), "r2_entrena_al_reves": r2_score(pe, y_e)}
    _CACHE["arbol"] = ref
    return ref


@_seguro
def revisar_arbol(*sueltos, mae=..., r2_entrenamiento=..., r2_prueba=..., **otros):
    """Ejercicio 1: el MAE en prueba y el R² en entrenamiento y en prueba del árbol de 6 niveles."""
    nombres = ["mae", "r2_entrenamiento", "r2_prueba"]
    if _sobrantes(sueltos, otros, nombres, "revisar_arbol(mae=mae_prueba, r2_entrenamiento=r2_entrena6, "
                                           "r2_prueba=r2_prueba6)"):
        return
    if _datos() is None:
        return
    vals = _leer({"mae": mae, "r2_entrenamiento": r2_entrenamiento, "r2_prueba": r2_prueba}, nombres)
    if not vals:
        return
    ref = _referencia_arbol()
    bien = ref[6]
    tol = {"mae": 0.5, "r2_entrenamiento": 0.005, "r2_prueba": 0.005}
    # ¿Qué otro árbol produce los valores entregados? Se decide con todos juntos: los R² de árboles
    # vecinos se parecen, y un solo valor puede coincidir por casualidad con otra profundidad.
    fina = {"mae": 0.1, "r2_entrenamiento": 0.001, "r2_prueba": 0.001}
    aciertos = {p: sum(_cerca(vals[n], r[n], fina[n]) for n in vals) for p, r in ref.items() if p != 6}
    otra, n_otra = max(aciertos.items(), key=lambda par: par[1])
    if n_otra < min(2, len(vals)) or list(aciertos.values()).count(n_otra) > 1:
        otra = False          # ninguna profundidad (o varias) explica los valores
    bien_ahora = {}
    for nombre in nombres:
        if nombre not in vals:
            continue
        v = vals[nombre]
        # un valor «dentro de la tolerancia» que está más cerca del de otra profundidad no cuenta como bien
        de_otra = otra is not False and abs(v - ref[otra][nombre]) < abs(v - bien[nombre])
        bien_ahora[nombre] = _cerca(v, bien[nombre], tol[nombre]) and not de_otra
        if bien_ahora[nombre]:
            print(f"✓  {nombre}: bien")
            continue
        # ¿De qué otro árbol o de qué otra cuenta es?
        pista = None
        if otra is not False and not _cerca(v, bien[nombre], fina[nombre]) and _cerca(v, ref[otra][nombre], fina[nombre]):
            pista = (f"los valores son los de un árbol {'sin límite' if otra is None else 'de ' + _niveles(otra)}: "
                     "¿pusiste max_depth=6?")
        elif nombre == "mae":
            if _cerca(v, bien["mae_entrena"], 0.5):
                pista = "ese es el MAE del entrenamiento; el que importa es el de la prueba (y_prueba)"
            elif _cerca(v, bien["rmse"], 0.5):
                pista = "esa es la RMSE: el MAE promedia los errores sin elevarlos al cuadrado"
            elif _cerca(v, bien["mse"], 5):
                pista = "ese es el error cuadrático medio (en GB²): el MAE promedia los errores absolutos"
        else:
            otro = "r2_prueba" if nombre == "r2_entrenamiento" else "r2_entrenamiento"
            if _cerca(v, bien[otro], 0.001):
                pista = "está cruzado con el otro R²: uno se mide con el entrenamiento y el otro, con la prueba"
            elif 1 < v <= 100 and _cerca(v / 100, bien[nombre], 0.005):
                pista = "el R² se escribe como proporción (1 es un modelo perfecto), no como porcentaje"
            elif nombre == "r2_prueba" and _cerca(v, bien["r2_al_reves"], 0.005):
                pista = "r2_score recibe primero lo real y después lo predicho: r2_score(y_prueba, pred_prueba)"
            elif nombre == "r2_entrenamiento" and _cerca(v, bien["r2_entrena_al_reves"], 0.005):
                pista = "r2_score recibe primero lo real y después lo predicho: r2_score(y_ent, pred_entrena)"
            elif v < 1 and v == round(v, 1) and round(v, 1) == round(bien[nombre], 1):
                pista = "¿redondeaste? deja al menos dos decimales"
        if pista is None:
            clave = nombre
            posibles = [p for p, r in ref.items() if p != 6 and _cerca(v, r[clave], tol[nombre] / 5 if nombre == "mae" else 0.001)]
            if len(posibles) == 1:
                p = posibles[0]
                pista = ("ese es el valor del árbol sin límite: ¿pusiste max_depth=6?" if p is None
                         else f"ese es el valor de un árbol de {_niveles(p)}: ¿pusiste max_depth=6?")
        print(f"✗  {nombre}: {pista or 'no coincide; revisa con qué datos entrenaste y con cuáles mediste'}")
    if all(bien_ahora.get(n) for n in nombres):
        sin = ref[None]
        print(f"\n¡Listo! El árbol de 6 niveles tiene un R² de {_coma(bien['r2_entrenamiento'], 2)} en entrenamiento y "
              f"de {_coma(bien['r2_prueba'], 2)} en prueba; el sin límite, de {_coma(sin['r2_entrenamiento'], 2)} y de "
              f"{_coma(sin['r2_prueba'], 2)}. Escribe en «Tus respuestas» qué te dice esa diferencia.")


# ---------------------------------------------------------------------------------------------
# Clasificación: los modelos de referencia
# ---------------------------------------------------------------------------------------------

def _modelos_c():
    """El bosque, la logística y XGBoost de la clase, ya entrenados."""
    if "modelos_c" in _CACHE:
        return _CACHE["modelos_c"]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    Xc_e, Xc_p, yc_e, yc_p = _datos()["clf"]
    modelos = {
        "bosque": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=2026, n_jobs=-1),
        "logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=2026),
    }
    for m in modelos.values():
        m.fit(Xc_e, yc_e)
    _CACHE["modelos_c"] = modelos
    return modelos


def _matriz(y_real, pred):
    import numpy as np
    y_real, pred = np.asarray(y_real), np.asarray(pred)
    return {"vn": int(((y_real == 0) & (pred == 0)).sum()), "fp": int(((y_real == 0) & (pred == 1)).sum()),
            "fn": int(((y_real == 1) & (pred == 0)).sum()), "vp": int(((y_real == 1) & (pred == 1)).sum())}


# ---------------------------------------------------------------------------------------------
# Ejercicio 2 · La matriz de confusión del bosque
# ---------------------------------------------------------------------------------------------

@_seguro
def revisar_matriz(*sueltos, vp=..., fp=..., fn=..., vn=..., **otros):
    """Ejercicio 2: los cuatro números de la matriz de confusión del bosque, con el umbral de 0,5."""
    nombres = ["vp", "fp", "fn", "vn"]
    if _sobrantes(sueltos, otros, nombres, "revisar_matriz(vp=vp_b, fp=fp_b, fn=fn_b, vn=vn_b)"):
        return
    if _datos() is None:
        return
    entregados = {"vp": vp, "fp": fp, "fn": fn, "vn": vn}
    # si la tupla de .ravel() quedó entera en una sola variable
    import numpy as np
    for nombre, x in entregados.items():
        if isinstance(x, (np.ndarray, list, tuple)) and np.size(x) == 4:
            print(f"✗  {nombre}: tiene los cuatro números juntos; reparte .ravel() en cuatro variables, "
                  "en el orden VN, FP, FN, VP: vn_b, fp_b, fn_b, vp_b = matriz.ravel()")
            return
    vals = _leer(entregados, nombres)
    if not vals:
        return
    no_enteros = [k for k, v in vals.items() if v != int(v) or v < 0]
    if no_enteros:
        print(f"✗  {', '.join(no_enteros)}: cada casilla de la matriz es un conteo de clientes (un entero "
              "de 0 en adelante), no una proporción")
        return
    Xc_e, Xc_p, yc_e, yc_p = _datos()["clf"]
    modelos = _modelos_c()
    bien = _matriz(yc_p, modelos["bosque"].predict(Xc_p))
    tol = 2
    if all(_cerca(vals.get(k), bien[k], tol) for k in nombres):
        print("✓  vp, fp, fn y vn: bien")
        print(f"\n¡Listo! El bosque encontró a {bien['vp']} de los {bien['vp'] + bien['fn']} que cancelaron "
              f"y señaló a {bien['fp']} que no cancelaron. Escríbelo en «Tus respuestas».")
        return
    completos = len(vals) == 4
    if completos:
        total = sum(vals.values())
        if total == len(yc_e):
            print(f"✗  Los cuatro suman {int(total)}: son los clientes del entrenamiento. La matriz se calcula "
                  "con la prueba (yc_prueba y la predicción de Xc_prueba)")
            return
        if total != len(yc_p) and sum(_cerca(vals[k], bien[k], tol) for k in nombres) < 3:
            print(f"✗  Los cuatro suman {int(total)}, y la prueba tiene {len(yc_p)} clientes: revisa que la "
                  "matriz sea de yc_prueba contra la predicción de Xc_prueba")
            return
        # ¿Es la matriz de otro modelo?
        for nombre, m in modelos.items():
            if nombre != "bosque" and all(_cerca(vals[k], v, tol) for k, v in _matriz(yc_p, m.predict(Xc_p)).items()):
                print(f"✗  Esos son los números de {'la logística' if nombre == 'logística' else nombre}, no los del bosque: "
                      "predice con bosque_c")
                return
        if vals["vp"] == 0 and vals["fp"] == 0:
            print("✗  Con VP y FP en 0, el modelo dijo «No» a todos, como la línea base: predice con bosque_c")
            return
        # ¿Están los números bien, pero en otro lugar?
        if sorted(vals.values()) == sorted(bien.values()):
            if (_cerca(vals["fp"], bien["fn"], 0) and _cerca(vals["fn"], bien["fp"], 0)
                    and _cerca(vals["vp"], bien["vp"], 0) and _cerca(vals["vn"], bien["vn"], 0)):
                print("✗  fp y fn están cruzados: confusion_matrix recibe primero lo real y después lo predicho "
                      "(yc_prueba, pred_bosque)")
            else:
                print("✗  Están los cuatro números, pero no en su lugar: el orden de .ravel() es VN, FP, FN, VP")
            return
    for k in nombres:
        if k in vals:
            print(f"{'✓' if _cerca(vals[k], bien[k], tol) else '✗'}  {k}: "
                  f"{'bien' if _cerca(vals[k], bien[k], tol) else 'no coincide'}")
    malos = [k for k, v in vals.items() if not _cerca(v, bien[k], tol)]
    if not malos:
        return      # lo entregado está bien; solo falta lo que no era un número: ya se avisó arriba
    if len(vals) == 4 and len(malos) == 1:
        print(f"·  Los otros tres coinciden: revisa cómo sacaste {malos[0]}")
    elif "vp" in vals and "fn" in vals and _cerca(vals["vp"] + vals["fn"], bien["vp"] + bien["fn"], 0):
        print("·  vp + fn sí suman los que cancelaron: revisa con qué modelo y con qué umbral predijiste")
    else:
        print("·  Pista: predice con bosque_c.predict(Xc_prueba) y calcula confusion_matrix(yc_prueba, …)")


# ---------------------------------------------------------------------------------------------
# Ejercicio 3 · El umbral de la gerente
# ---------------------------------------------------------------------------------------------

_UMBRALES = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]


def _tabla_umbral(nombre_modelo):
    """Sensibilidad, llamadas y encontrados de cada umbral de la lista, con un modelo."""
    clave = f"umbral_{nombre_modelo}"
    if clave in _CACHE:
        return _CACHE[clave]
    import numpy as np
    Xc_e, Xc_p, yc_e, yc_p = _datos()["clf"]
    prob = _modelos_c()[nombre_modelo].predict_proba(Xc_p)[:, 1]
    y = np.asarray(yc_p)
    tabla = {}
    for u in _UMBRALES:
        pred = (prob >= u).astype(int)
        tabla[u] = {"sensibilidad": float(pred[y == 1].mean()), "llamadas": int(pred.sum()),
                    "encontrados": int(pred[y == 1].sum())}
    _CACHE[clave] = tabla
    return tabla


@_seguro
def revisar_umbral(*sueltos, umbral=..., llamadas=..., **otros):
    """Ejercicio 3: el umbral más alto con el que XGBoost encuentra al menos a 7 de cada 10, y sus llamadas."""
    nombres = ["umbral", "llamadas"]
    if _sobrantes(sueltos, otros, nombres, "revisar_umbral(umbral=umbral_elegido, llamadas=llamadas_elegido)"):
        return
    if _datos() is None:
        return
    vals = _leer({"umbral": umbral, "llamadas": llamadas}, nombres)
    if not vals:
        return
    tabla = _tabla_umbral("XGBoost")
    cumplen = [u for u in _UMBRALES if tabla[u]["sensibilidad"] >= 0.70]
    mejor = max(cumplen)
    u = vals.get("umbral")
    elegido = None
    if u is not None:
        if 1 < u <= 100 and any(_cerca(u / 100, x, 1e-9) for x in _UMBRALES):
            print("✗  umbral: es una probabilidad entre 0 y 1, no un porcentaje")
        elif not any(_cerca(u, x, 1e-9) for x in _UMBRALES):
            print("✗  umbral: elige uno de los umbrales del ciclo (0.50, 0.45, …, 0.15)")
        else:
            elegido = min(_UMBRALES, key=lambda x: abs(x - u))
            s = tabla[elegido]["sensibilidad"]
            if elegido == mejor:
                print("✓  umbral: bien")
            elif s < 0.70:
                print(f"✗  umbral: con {_coma(elegido, 2)}, XGBoost encuentra al {_coma(s * 100, 1)} % de los que cancelan: "
                      "no llega a la meta del 70 %. Baja el umbral")
            else:
                print(f"✗  umbral: con {_coma(elegido, 2)} sí llega a la meta, pero hay un umbral más alto que "
                      "también llega, con menos llamadas")
    n = vals.get("llamadas")
    if n is not None:
        if 0 < n < 1:
            print("✗  llamadas: es un conteo de clientes (cuántos 1 hay en pred), no una proporción")
        elif elegido is None:
            print("·  llamadas: la reviso cuando el umbral sea uno de la lista")
        else:
            fila = tabla[elegido]
            if _cerca(n, fila["llamadas"], 2):
                print(f"✓  llamadas: bien para el umbral de {_coma(elegido, 2)}")
            elif _cerca(n, fila["encontrados"], 2):
                print("✗  llamadas: esos son los encontrados (los que cancelaron); las llamadas son todos los "
                      "clientes con 1 en pred: pred.sum()")
            elif _cerca(n, 2000 - fila["llamadas"], 2):
                print("✗  llamadas: contaste los 0; las llamadas son los 1 de pred")
            elif _cerca(n, _tabla_umbral("logística")[elegido]["llamadas"], 2):
                print("✗  llamadas: son las de la logística; aquí van las de XGBoost (prob_xgb)")
            else:
                otro = [x for x in _UMBRALES if _cerca(n, tabla[x]["llamadas"], 2)]
                print("✗  llamadas: " + (f"son las del umbral de {_coma(otro[0], 2)}, no las del que elegiste"
                                         if otro else "no coinciden con las de ese umbral: cuenta los 1 de pred"))
    if elegido == mejor and n is not None and _cerca(n, tabla[mejor]["llamadas"], 2):
        f = tabla[mejor]
        print(f"\n¡Listo! Con {_coma(mejor, 2)}, XGBoost encuentra a {f['encontrados']} de los que cancelaron "
              f"({_coma(f['sensibilidad'] * 100, 1)} %) con {f['llamadas']} llamadas. Compara su exactitud con la de 0,5 "
              "en «Tus respuestas».")


# ---------------------------------------------------------------------------------------------
# Ejercicio 4 · La profundidad que elige la validación cruzada
# ---------------------------------------------------------------------------------------------

def _referencia_profundidad():
    if "profundidad" in _CACHE:
        return _CACHE["profundidad"]
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.tree import DecisionTreeClassifier

    Xc_e, Xc_p, yc_e, yc_p = _datos()["clf"]
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    ref = {}
    for p in range(1, 11):
        m = DecisionTreeClassifier(max_depth=p, random_state=2026)
        cv = cross_validate(m, Xc_e, yc_e, cv=pliegues, scoring=["f1", "accuracy", "recall"],
                            return_train_score=True)
        m.fit(Xc_e, yc_e)
        cv5 = cross_validate(DecisionTreeClassifier(max_depth=p, random_state=2026), Xc_e, yc_e, cv=5, scoring="f1")
        ref[p] = {"f1": cv["test_f1"].mean(), "f1_cv5": cv5["test_score"].mean(), "exactitud": cv["test_accuracy"].mean(),
                  "sensibilidad": cv["test_recall"].mean(), "f1_entrena": cv["train_f1"].mean(),
                  "f1_prueba": f1_score(yc_p, m.predict(Xc_p))}
    _CACHE["profundidad"] = ref
    return ref


@_seguro
def revisar_profundidad(*sueltos, profundidad=..., f1_medio=..., **otros):
    """Ejercicio 4: la profundidad con el F1 medio más alto en la validación cruzada, y ese F1."""
    nombres = ["profundidad", "f1_medio"]
    if _sobrantes(sueltos, otros, nombres, "revisar_profundidad(profundidad=mejor_profundidad, "
                                           "f1_medio=mejor_f1)"):
        return
    if _datos() is None:
        return
    vals = _leer({"profundidad": profundidad, "f1_medio": f1_medio}, nombres)
    if not vals:
        return
    ref = _referencia_profundidad()
    mejor = max(ref, key=lambda p: ref[p]["f1"])
    p = vals.get("profundidad")
    f = vals.get("f1_medio")
    mejor_cv5 = max(ref, key=lambda q: ref[q]["f1_cv5"])
    if p is not None and f is not None and p == mejor_cv5 and mejor_cv5 != mejor and \
            _cerca(f, ref[mejor_cv5]["f1_cv5"], 0.001):
        print("✗  Esos F1 salen de cv=5, que reparte los pliegues de otra manera: usa cv=pliegues (los pliegues de "
              "la clase) para que todos obtengamos lo mismo")
        return
    if p is not None:
        if p != int(p) or not 1 <= p <= 10:
            print("✗  profundidad: es un número entero de niveles, de 1 a 10")
            p = None
        else:
            p = int(p)
            if p == mejor:
                print("✓  profundidad: bien")
            elif p == max(ref, key=lambda q: ref[q]["f1_entrena"]):
                print("✗  profundidad: esa es la que mejor le va en el entrenamiento, no en la validación: "
                      "mide con cross_val_score")
            elif p == max(ref, key=lambda q: ref[q]["f1_prueba"]):
                print("✗  profundidad: esa es la que mejor le va en la prueba; aquí se elige con la validación "
                      "cruzada, sobre el entrenamiento")
            elif abs(ref[p]["f1"] - ref[mejor]["f1"]) < 0.01:
                print(f"✗  profundidad: casi: con {_niveles(p)}, el F1 medio es {_coma(ref[p]['f1'])}; hay una "
                      "profundidad un poco mejor")
            else:
                print(f"✗  profundidad: con {_niveles(p)}, el F1 medio es {_coma(ref[p]['f1'])}; busca el más alto")
    if f is not None and p is None:
        print("·  f1_medio: lo reviso cuando la profundidad sea un entero de 1 a 10")
    elif f is not None:
        objetivo = ref[p]["f1"]
        if _cerca(f, objetivo, 0.005):
            print("✓  f1_medio: bien" + ("" if p == mejor else f" para {_niveles(p)}"))
        elif 1 < f <= 100 and _cerca(f / 100, objetivo, 0.005):
            print("✗  f1_medio: el F1 se escribe como proporción, entre 0 y 1, no como porcentaje")
        elif _cerca(f, ref[p]["exactitud"], 0.005):
            print("✗  f1_medio: esa es la exactitud; pide scoring=\"f1\" en cross_val_score")
        elif _cerca(f, ref[p]["sensibilidad"], 0.005):
            print("✗  f1_medio: esa es la sensibilidad; pide scoring=\"f1\"")
        elif _cerca(f, ref[p]["f1_prueba"], 0.001):
            print("✗  f1_medio: ese es el F1 en la prueba; aquí va el promedio de los 5 pliegues")
        else:
            print("✗  f1_medio: no coincide: es el promedio (.mean()) de los 5 F1 de cross_val_score, con "
                  "cv=pliegues y scoring=\"f1\"")
    if p == mejor and f is not None and _cerca(f, ref[mejor]["f1"], 0.005):
        print(f"\n¡Listo! La validación cruzada elige {mejor} niveles. Explica en «Tus respuestas» por qué no se "
              "elige con la prueba.")
