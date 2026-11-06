"""Ayudas de la práctica de la sesión 1: verificadores con pistas y dos gráficas.

Van dentro de la celda oculta del notebook, tras descarga_datos.py y ayudas.py.

Los verificadores NUNCA se caen: si algo inesperado pasa, lo dicen con una pista y siguen.
Se llaman sin argumentos (revisar_division()) y leen las variables del notebook; así, si la
celda del ejercicio falló y la variable no existe, lo dicen en lugar de dar un NameError.
También aceptan las variables como argumentos, en el mismo orden.
Calculan lo esperado desde datos/clientes.csv en el momento de revisar: las cifras de la
respuesta no están escritas aquí. Dan pistas sin revelar la respuesta; solo cuando la
respuesta está bien muestran las cifras.
"""

import functools
import unicodedata

# Paleta del curso (validada con la skill dataviz): una serie en morado; gris para la referencia
_MORADO, _NARANJA = "#8A2E9E", "#F46201"
_TINTA, _TINTA_SUAVE, _REJILLA = "#2B2B2B", "#5F6B76", "#E3E3E0"
_SEMILLA = 2026
_FALTA = object()  # marca de «esta variable no existe en el notebook»


def _seguro(funcion):
    """Ningún verificador se cae: un error inesperado se convierte en una pista."""
    @functools.wraps(funcion)
    def envoltura(*args, **kwargs):
        try:
            return funcion(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 — a propósito: nunca romper la clase
            print(f"✗  No pude revisar tu respuesta ({type(error).__name__}: {error}).\n"
                  "   Pista: ejecuta las celdas anteriores en orden (Entorno de ejecución → "
                  "Ejecutar anteriores) y revisa que no quede ningún ... sin reemplazar.")
    return envoltura


def _variables(nombres, args, ejercicio, opcionales=()):
    """Los valores que se van a revisar: los argumentos o, si no hay, las variables del notebook.

    Devuelve None (después de explicar qué falta) si no se puede revisar todavía.
    """
    if len(args) > len(nombres):
        print(f"✗  sobran argumentos: llama a la función sin nada entre los paréntesis")
        return None
    if args:
        valores = list(args) + [_FALTA] * (len(nombres) - len(args))
    else:
        espacio = globals()  # en el notebook, son las variables de tus celdas
        valores = [espacio.get(n, _FALTA) for n in nombres]
    faltan = [n for n, v in zip(nombres, valores) if v is _FALTA and n not in opcionales]
    if faltan:
        lista = ", ".join(f"`{n}`" for n in faltan)
        verbo = "existe" if len(faltan) == 1 else "existen"
        print(f"·  todavía no {verbo} {lista}: ejecuta sin errores la celda del ejercicio "
              f"{ejercicio}. Si falla, lee el error: casi siempre es un ... sin reemplazar.")
        return None
    return valores


def _normal(texto):
    """Minúsculas, sin tildes ni espacios de más: «24 Meses » = «24 meses»."""
    texto = unicodedata.normalize("NFD", " ".join(str(texto).lower().split()))
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _decimal(x, cifras=1):
    """Número con coma decimal, para los mensajes (el texto va con coma; el código, con punto).

    Redondea «hacia arriba» en el 5 (76,45 → 76,5), como se redondea a mano, y no como el
    binario de Python, que deja 76,449999… en 76,4.
    """
    from decimal import ROUND_HALF_UP, Decimal

    exacto = Decimal(str(round(float(x), 10)))
    return str(exacto.quantize(Decimal(1).scaleb(-cifras), ROUND_HALF_UP)).replace(".", ",")


def _pct(x, cifras=1):
    return _decimal(100 * x, cifras) + " %"


def _clientes():
    import pandas as pd

    return pd.read_csv("datos/clientes.csv")


def _es_puntos(valor):
    return valor is Ellipsis or (isinstance(valor, str) and valor.strip() in ("...", "…"))


# ─────────────────────────────── Ejercicio 1 ───────────────────────────────

_CONTRATOS = ["Mes a mes", "12 meses", "24 meses"]
_FORMAS = ("se espera una serie de pandas (lo que devuelve groupby(...)[...].mean()) "
           "o un diccionario {\"Mes a mes\": ..., \"12 meses\": ..., \"24 meses\": ...}")


def _como_serie(respuesta):
    """Acepta Series, dict, DataFrame (también con reset_index) o la tabla cruzada (columna «Sí»)."""
    import pandas as pd

    if isinstance(respuesta, dict):
        return pd.Series(respuesta, dtype="object"), None
    if isinstance(respuesta, pd.DataFrame):
        esperados = {_normal(c) for c in _CONTRATOS}
        for columna in respuesta.columns:  # groupby(..., as_index=False) o .reset_index()
            if (not pd.api.types.is_numeric_dtype(respuesta[columna])
                    and esperados & {_normal(v) for v in respuesta[columna]}):
                respuesta = respuesta.drop(columns=columna).set_index(
                    respuesta[columna].to_numpy())
                break
        columnas = {_normal(c): c for c in respuesta.columns}
        if "si" in columnas:
            return respuesta[columnas["si"]], "tomé la columna «Sí» de tu tabla"
        numericas = respuesta.select_dtypes("number").columns
        if len(numericas) == 1:
            return respuesta[numericas[0]], None
        return None, ("tu resultado es una tabla con varias columnas; se espera una sola "
                      "cifra por contrato")
    if isinstance(respuesta, pd.Series):
        return respuesta, None
    return None, _FORMAS


@_seguro
def revisar_tasa_contrato(*args):
    """Revisa `tasa_contrato`: la tasa de cancelación de cada tipo de contrato."""
    import math

    import pandas as pd

    valores_nb = _variables(["tasa_contrato"], args, 1)
    if valores_nb is None:
        return
    tasa_contrato = valores_nb[0]
    if _es_puntos(tasa_contrato):
        print("·  tasa_contrato todavía es ...: reemplázalo por tu cálculo y ejecuta la celda")
        return
    serie, nota = _como_serie(tasa_contrato)
    if serie is None:
        print(f"✗  {nota}")
        return
    if isinstance(serie.index, pd.MultiIndex):
        print("✗  agrupaste por más de una columna: agrupa solo por \"contrato\"")
        return
    tuyas = {_normal(k): v for k, v in serie.items()}
    esperados = {_normal(c): c for c in _CONTRATOS}
    if not set(esperados) & set(tuyas):
        muestra = ", ".join(str(k) for k in list(serie.index)[:4])
        if isinstance(tasa_contrato, dict):
            print(f"✗  las claves de tu diccionario son {muestra}: escríbelas como en los datos, "
                  "\"Mes a mes\", \"12 meses\" y \"24 meses\"")
        else:
            print(f"✗  los grupos de tu resultado son {muestra}…: agrupa por la columna "
                  "\"contrato\" (Mes a mes, 12 meses, 24 meses)")
        return
    try:
        valores = {k: float(v) for k, v in tuyas.items() if k in esperados}
    except (TypeError, ValueError):
        print("✗  tus valores no son números: ¿promediaste una columna de texto? La columna "
              "que se promedia es la de 1 y 0 que creamos en [tasa]")
        return
    if any(math.isnan(v) for v in valores.values()):
        print("✗  hay valores vacíos (NaN) en tu resultado: revisa que promedies la columna de "
              "1 y 0 de [tasa] y que no filtres de más")
        return

    datos = _clientes()
    cancela = datos["cancelo"] == "Sí"
    grupos = {_normal(c): datos["contrato"] == c for c in _CONTRATOS}
    tasa = {k: cancela[g].mean() for k, g in grupos.items()}                 # P(cancela | contrato)
    sigue = {k: 1 - v for k, v in tasa.items()}                              # P(no cancela | contrato)
    inversa = {k: g[cancela].mean() for k, g in grupos.items()}              # P(contrato | cancela)
    conjunta = {k: (g & cancela).mean() for k, g in grupos.items()}          # P(contrato y cancela)
    canceladas = {k: (g & cancela).sum() for k, g in grupos.items()}
    clientes = {k: g.sum() for k, g in grupos.items()}

    def cerca(tuyos, referencia, tol):
        return all(abs(tuyos[k] - referencia[k]) <= tol for k in tuyos)

    if nota:
        print(f"·  {nota}")
    if max(valores.values()) > 1.5 and cerca(valores, {k: 100 * v for k, v in tasa.items()}, 0.6):
        valores = {k: v / 100 for k, v in valores.items()}
        print("·  la escribiste en porcentaje; también vale (en el curso usamos proporciones)")
    if cerca(valores, canceladas, 0.5):
        print("✗  esos son conteos: cuántos clientes cancelaron en cada contrato. La tasa es una "
              "proporción dentro de cada grupo: ¿qué operación de [tasa] da la proporción de unos?")
        return
    if cerca(valores, clientes, 0.5):
        print("✗  contaste los clientes de cada contrato. Se pide qué parte de ellos canceló: "
              "¿qué columna de 1 y 0 creamos en [tasa]?")
        return
    if cerca(valores, sigue, 0.006):
        print("✗  esa es la proporción de clientes que se QUEDARON. Revisa: en \"cancelo_num\" "
              "el 1 es «canceló»")
        return
    if cerca(valores, inversa, 0.006):
        print("✗  dividiste entre los que cancelaron: eso es P(contrato | cancela), lo que vimos "
              "en [inversa]. Se pide al revés, P(cancela | contrato): dentro de cada contrato, "
              "¿qué parte canceló?")
        return
    if cerca(valores, conjunta, 0.006):
        print("✗  dividiste entre los 8000 clientes. La tasa de cada contrato se calcula dentro "
              "de su propio grupo, como en [condicional]")
        return

    aciertos = 0
    for k, original in esperados.items():
        if k not in valores:
            print(f"·  {original}: no está en tu resultado")
            continue
        diferencia = abs(valores[k] - tasa[k])
        if diferencia <= 0.0015:
            aciertos += 1
            print(f"✓  {original}: {_pct(tasa[k])}")
        elif diferencia <= 0.006:
            aciertos += 1
            print(f"✓  {original}: {_pct(tasa[k])} (la redondeaste; para comparar, mejor sin "
                  "redondear)")
        elif diferencia <= 0.05:
            print(f"✗  {original}: se parece, pero no coincide. ¿La redondeaste demasiado? "
                  "Deja al menos tres decimales")
        else:
            print(f"✗  {original}: no coincide. ¿Promediaste la columna de 1 y 0 solo con los "
                  "clientes de ese contrato?")
    if aciertos == len(_CONTRATOS):
        veces = tasa["mes a mes"] / tasa["24 meses"]
        print(f"\nBien: las tres tasas. La de los clientes mes a mes es {_decimal(veces)} veces "
              "la de los que tienen permanencia de 24 meses.\n¿Prueba eso que el contrato CAUSA "
              "la cancelación? Respóndelo en «Para la discusión», al final del notebook.")
    else:
        print(f"\n{aciertos} de {len(_CONTRATOS)} contratos bien")


# ─────────────────────────────── Ejercicio 2 ───────────────────────────────

def _divisiones_esperadas():
    """Índices de prueba con la semilla del curso: estratificada (la buena) y sin estratificar."""
    from sklearn.model_selection import train_test_split

    datos = _clientes()
    y = (datos["cancelo"] == "Sí").astype(int)
    x = datos[["antiguedad_meses"]]
    _, con, _, y_con = train_test_split(x, y, test_size=0.25, random_state=_SEMILLA, stratify=y)
    _, sin = train_test_split(x, test_size=0.25, random_state=_SEMILLA)
    return set(con.index), set(sin.index), int(y_con.sum())


def _largo(objeto):
    try:
        return len(objeto)
    except TypeError:
        return None


def _es_texto_si_no(objeto):
    import numpy as np
    import pandas as pd

    muestra = pd.Series(np.asarray(objeto, dtype=object).ravel()[:30]).astype(str)
    return any(_normal(v) in ("si", "no") for v in muestra)


@_seguro
def revisar_division(*args):
    """Revisa X_ent, X_prueba, y_ent, y_prueba: tamaños, estratificación y semilla."""
    import numpy as np
    import pandas as pd

    nombres = ["X_ent", "X_prueba", "y_ent", "y_prueba"]
    valores = _variables(nombres, args, 2)
    if valores is None:
        return
    X_ent, X_prueba, y_ent, y_prueba = valores
    if any(_es_puntos(v) for v in valores):
        print("·  todavía hay algún ... sin reemplazar")
        return
    largos = dict(zip(nombres, (_largo(v) for v in valores)))
    if None in largos.values():
        malos = [k for k, v in largos.items() if v is None]
        sujeto = (f"`{malos[0]}` no es una tabla ni una columna" if len(malos) == 1 else
                  f"{', '.join(f'`{m}`' for m in malos)} no son tablas ni columnas")
        print(f"✗  {sujeto}: ¿ejecutaste la celda de train_test_split sin errores?")
        return
    total = largos["X_ent"] + largos["X_prueba"]
    if largos["X_ent"] != largos["y_ent"] or largos["X_prueba"] != largos["y_prueba"]:
        print("✗  `X` y `y` no tienen el mismo número de filas en cada grupo: pásalos juntos en "
              "la misma llamada, train_test_split(X, y, ...), y recibe los cuatro en este orden: "
              "X_ent, X_prueba, y_ent, y_prueba")
        return
    if total != 8000:
        print(f"✗  entre los dos grupos hay {total} clientes, no 8000: divide la `X` y la `y` "
              "completas de la celda [xy]")
        return
    if largos["X_prueba"] == 6000 and largos["X_ent"] == 2000:
        indice_ok, indice_sin, _ = _divisiones_esperadas()
        if isinstance(X_ent, pd.DataFrame) and set(X_ent.index) in (indice_ok, indice_sin):
            print("✗  los grupos quedaron al revés: train_test_split devuelve primero el de "
                  "entrenamiento y después el de prueba (X_ent, X_prueba, y_ent, y_prueba)")
        else:
            print("✗  el grupo de prueba tiene 6000 clientes, las tres cuartas partes: test_size es "
                  "la fracción que va a PRUEBA (o recibiste los cuatro grupos en otro orden)")
        return
    if largos["X_prueba"] != 2000:
        n = largos["X_prueba"]
        if n < 100:
            print(f"✗  el grupo de prueba tiene solo {n} clientes. Con un número entero, "
                  "test_size pide ESE número de filas; para una fracción escribe un decimal "
                  "con punto, entre 0 y 1")
        else:
            print(f"✗  el grupo de prueba tiene {n} clientes ({_pct(n / 8000, 0)}): se pide la "
                  "cuarta parte. Revisa test_size")
        return
    if isinstance(X_prueba, pd.DataFrame) and set(X_prueba.columns) != {"mes_a_mes", "antiguedad_meses"}:
        print("✗  X_prueba no tiene las columnas de la `X` de [xy] (mes_a_mes y antiguedad_meses): "
              "vuelve a ejecutar [xy] y divide esa `X`")
        return
    if _es_texto_si_no(y_prueba):
        print("✗  y_ent y y_prueba son texto (Sí / No). Divide la etiqueta numérica `y` de [xy], "
              "la de 1 y 0, no la columna \"cancelo\"")
        return

    indice_ok, indice_sin, cancelan_ok = _divisiones_esperadas()
    y_prueba_num = np.asarray(y_prueba, dtype=float)
    tasa_ent = float(np.mean(np.asarray(y_ent, dtype=float)))
    tasa_prueba = float(np.mean(y_prueba_num))
    if not (0 <= tasa_ent <= 1 and 0 <= tasa_prueba <= 1):
        print("✗  `y` debe ser la columna de 1 y 0 de la celda [xy]; vuelve a ejecutarla")
        return
    if int(round(y_prueba_num.sum())) != cancelan_ok:
        sin_estratificar = isinstance(X_prueba, pd.DataFrame) and set(X_prueba.index) == indice_sin
        print(f"✗  la tasa de cancelación es {_pct(tasa_ent, 2)} en entrenamiento y "
              f"{_pct(tasa_prueba, 2)} en prueba: no son iguales"
              + (" (casi, por azar, pero no exactamente)" if abs(tasa_ent - tasa_prueba) < 0.002 else "")
              + ". ¿Qué argumento le pide a train_test_split que conserve la proporción de `y` "
              "en los dos grupos?")
        if sin_estratificar:
            print("   (la semilla sí es la del curso)")
        return
    if isinstance(X_prueba, pd.DataFrame):
        if set(X_prueba.index) != indice_ok:
            print("✗  los tamaños y la tasa de cancelación están bien, pero no son los clientes de "
                  "prueba del curso. Revisa dos cosas: que random_state sea la semilla del curso y "
                  "que stratify sea la etiqueta `y`, no otra columna")
            return
        if isinstance(y_prueba, pd.Series) and not y_prueba.index.equals(X_prueba.index):
            print("✗  X_prueba y y_prueba no tienen los mismos clientes: divide `X` y `y` en la "
                  "misma llamada")
            return
    else:
        print("·  X_prueba no es un DataFrame, así que no puedo comprobar la semilla; "
              "pásale a train_test_split la `X` de [xy] tal cual")
    print(f"✓  {largos['X_ent']} clientes para entrenar y {largos['X_prueba']} para probar\n"
          f"✓  la misma tasa de cancelación en los dos grupos: {_pct(tasa_ent)}")
    if isinstance(X_prueba, pd.DataFrame):
        print("✓  la semilla del curso: todos tenemos los mismos clientes de prueba")


# ─────────────────────────────── Ejercicio 3 ───────────────────────────────

def _referencia_modelos():
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    datos = _clientes()
    y = (datos["cancelo"] == "Sí").astype(int)
    X = pd.DataFrame({"mes_a_mes": (datos["contrato"] == "Mes a mes").astype(int),
                      "antiguedad_meses": datos["antiguedad_meses"]})
    x_ent, x_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.25, random_state=_SEMILLA,
                                                  stratify=y)
    base = DummyClassifier(strategy="most_frequent").fit(x_ent, y_ent)
    arbol = DecisionTreeClassifier(max_depth=2, random_state=_SEMILLA).fit(x_ent, y_ent)
    p_base, p_arbol = base.predict(x_pru), arbol.predict(x_pru)
    return {
        "exactitud_base": (p_base == y_pru).mean(),
        "exactitud_arbol": (p_arbol == y_pru).mean(),
        "exactitud_arbol_ent": (arbol.predict(x_ent) == y_ent).mean(),
        "encontrados_base": int(((p_base == 1) & (y_pru == 1)).sum()),
        "encontrados_arbol": int(((p_arbol == 1) & (y_pru == 1)).sum()),
        "falsas_alarmas_arbol": int(((p_arbol == 1) & (y_pru == 0)).sum()),
        "marcados_arbol": int((p_arbol == 1).sum()),
        "cancelaron_prueba": int(y_pru.sum()),
        "n_ent": len(x_ent), "n_prueba": len(x_pru),
    }


def _numero(valor):
    import numpy as np

    if valor is _FALTA or _es_puntos(valor) or valor is None or isinstance(valor, (str, bool)):
        return None
    try:
        arreglo = np.asarray(valor, dtype=float)
    except (TypeError, ValueError):
        return None
    if arreglo.size != 1:
        return "varios"
    return float(arreglo.reshape(-1)[0])


def _revisar_arbol(arbol, ref):
    if _es_puntos(arbol) or not hasattr(arbol, "fit"):
        return "✗  arbol: debe ser el árbol de decisión que creaste en la celda"
    tipo = type(arbol).__name__
    if tipo != "DecisionTreeClassifier":
        return (f"✗  arbol: es un {tipo}; el primer modelo es un DecisionTreeClassifier "
                "(un árbol que clasifica), con max_depth=2")
    if not hasattr(arbol, "tree_"):
        return "✗  arbol: todavía no está entrenado; falta arbol.fit(...)"
    vistos = int(arbol.tree_.n_node_samples[0])
    if getattr(arbol, "max_depth", None) != 2:
        return (f"✗  arbol: tiene max_depth={arbol.max_depth}; el primer modelo es muy simple, "
                "con solo dos niveles de preguntas")
    if vistos == ref["n_prueba"]:
        return ("✗  arbol: lo entrenaste con los clientes de PRUEBA. Tiene que aprender con los de "
                "entrenamiento y examinarse con los de prueba, que no vio")
    if vistos == ref["n_ent"] + ref["n_prueba"]:
        return ("✗  arbol: lo entrenaste con los 8000 clientes; así ya «vio» el examen. "
                "Entrénalo solo con los de entrenamiento")
    if vistos != ref["n_ent"]:
        return (f"✗  arbol: aprendió con {vistos} clientes; deben ser los {ref['n_ent']} de "
                "entrenamiento (X_ent, y_ent)")
    return f"✓  arbol: dos niveles, entrenado con los {ref['n_ent']} clientes de entrenamiento"


def _revisar_exactitud(nombre, valor, esperado, ref):
    v = _numero(valor)
    if v is None:
        return f"·  {nombre}: todavía no es un número (¿quedó un ...?)"
    if v == "varios":
        return (f"✗  {nombre}: es una lista de valores; la exactitud es UN número: el "
                "promedio de (predicción == y_prueba)")
    if v == round(esperado * ref["n_prueba"]):
        return (f"✗  {nombre}: ese es el NÚMERO de aciertos; la exactitud es la proporción de "
                "aciertos: promedia, no sumes")
    if 1 < v <= 100:
        v /= 100
    if abs(v - esperado) < 0.0006:
        return f"✓  {nombre}: {_pct(esperado)} de aciertos en prueba"
    if abs(v - (1 - esperado)) < 0.0006:
        return (f"✗  {nombre}: esa es la proporción de ERRORES; la exactitud cuenta los "
                "aciertos (predicción == y_prueba)")
    if nombre == "exactitud_arbol" and abs(v - ref["exactitud_arbol_ent"]) < 0.0006:
        return ("✗  exactitud_arbol: esa es la exactitud en ENTRENAMIENTO; el examen es con "
                "X_prueba y y_prueba")
    if abs(v - esperado) < 0.006:
        return (f"✓  {nombre}: {_pct(esperado)} de aciertos en prueba (la redondeaste; para "
                "comparar, mejor sin redondear)")
    return f"✗  {nombre}: no coincide; compara las predicciones sobre X_prueba con y_prueba"


def _revisar_conteo(nombre, valor, esperado, ref, pista):
    v = _numero(valor)
    total = ref["cancelaron_prueba"]
    if v is None:
        return f"·  {nombre}: todavía no es un número (¿quedó un ...?)"
    if v == "varios":
        return f"✗  {nombre}: es una lista; súmala con .sum() para tener un solo número"
    if v == esperado:
        return None
    if 0 < v < 1:
        return f"✗  {nombre}: se pide cuántos CLIENTES (un conteo), no una proporción: suma, no promedies"
    return pista(v, total)


@_seguro
def revisar_modelos(*args):
    """Revisa el ejercicio 3: el árbol, la exactitud en prueba y a cuántos encontró cada modelo."""
    nombres = ["arbol", "exactitud_base", "exactitud_arbol", "encontrados_base",
               "encontrados_arbol", "falsas_alarmas_arbol"]
    valores = _variables(nombres, args, 3, opcionales=("falsas_alarmas_arbol",))
    if valores is None:
        return
    arbol, exactitud_base, exactitud_arbol, encontrados_base, encontrados_arbol, falsas = valores
    ref = _referencia_modelos()
    total = ref["cancelaron_prueba"]

    def pista_base(v, total):
        return ("✗  encontrados_base: no coincide; cuenta los clientes con predicción 1 Y "
                "y_prueba 1, con las dos condiciones unidas por &")

    def pista_arbol(v, total):
        if v == total:
            return (f"✗  encontrados_arbol: {total} son todos los que cancelaron en prueba; cuenta "
                    "solo a los que, además, el árbol marcó como «cancela» (las dos condiciones con &)")
        if v == ref["marcados_arbol"]:
            return ("✗  encontrados_arbol: contaste a todos los que el árbol marcó como «cancela», "
                    "también a los que no cancelaron. Exige las dos condiciones a la vez, con &")
        return ("✗  encontrados_arbol: no coincide; cuenta como en encontrados_base, pero con "
                "pred_arbol")

    def pista_falsas(v, total):
        if v == ref["encontrados_arbol"]:
            return ("✗  falsas_alarmas_arbol: esos son los encontrados. Una falsa alarma es un "
                    "cliente que el árbol marcó como «cancela» (1) y que NO canceló (y_prueba == 0)")
        if v == ref["marcados_arbol"]:
            return ("✗  falsas_alarmas_arbol: contaste a todos los marcados; quédate solo con los "
                    "que NO cancelaron (y_prueba == 0)")
        if v == total - ref["encontrados_arbol"]:
            return ("✗  falsas_alarmas_arbol: esos son los que el árbol NO encontró (predijo 0 y sí "
                    "cancelaron). Una falsa alarma es al revés: predijo 1 y no canceló")
        return ("✗  falsas_alarmas_arbol: no coincide; cuenta los clientes con pred_arbol == 1 Y "
                "y_prueba == 0")

    mensajes = [
        _revisar_arbol(arbol, ref),
        _revisar_exactitud("exactitud_base", exactitud_base, ref["exactitud_base"], ref),
        _revisar_exactitud("exactitud_arbol", exactitud_arbol, ref["exactitud_arbol"], ref),
        _revisar_conteo("encontrados_base", encontrados_base, ref["encontrados_base"], ref,
                        pista_base)
        or f"✓  encontrados_base: encontró a {ref['encontrados_base']} de los {total} clientes "
           "de prueba que cancelaron",
        _revisar_conteo("encontrados_arbol", encontrados_arbol, ref["encontrados_arbol"], ref,
                        pista_arbol)
        or f"✓  encontrados_arbol: encontró a {ref['encontrados_arbol']} de los {total} clientes "
           "de prueba que cancelaron",
    ]
    pistas = sum(not m.startswith("✓") for m in mensajes)
    if falsas is _FALTA or _es_puntos(falsas):
        mensajes.append("·  falsas_alarmas_arbol (opcional): todavía no la calculas")
    else:
        mensajes.append(
            _revisar_conteo("falsas_alarmas_arbol", falsas, ref["falsas_alarmas_arbol"], ref,
                            pista_falsas)
            or f"✓  falsas_alarmas_arbol: marcó como «cancela» a {ref['falsas_alarmas_arbol']} "
               "clientes que no cancelaron")
        pistas += mensajes[-1].startswith("✗")
    for m in mensajes:
        print(m)
    if pistas == 0:
        print("\nBien. Mira las parejas de cifras: ¿cuál le sirve a la gerencia para decidir? "
              "Responde en «Para la discusión», al final del notebook.")


# ──────────────────────────────── Gráficas ─────────────────────────────────

def _estilo():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 13.5, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 13.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.titlelocation": "left", "axes.edgecolor": _REJILLA, "axes.labelcolor": _TINTA_SUAVE,
        "xtick.color": _TINTA_SUAVE, "ytick.color": _TINTA_SUAVE, "text.color": _TINTA,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": _REJILLA, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.formatter.use_locale": False,
    })


def _eje_porcentaje(eje, maximo):
    from matplotlib.ticker import FuncFormatter

    eje.set_ylim(0, maximo)
    eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))


@_seguro
def grafica_por_mes(tasa_por_mes, tasa_general):
    """Tasa de cancelación según los meses de antigüedad, con la tasa de todos como referencia."""
    import matplotlib.pyplot as plt

    _estilo()
    meses = [int(m) for m in tasa_por_mes.index]
    figura, eje = plt.subplots(figsize=(9.6, 4.2))
    eje.plot(meses, tasa_por_mes.values, color=_MORADO, linewidth=2.2, marker="o", markersize=5.5,
             markeredgecolor="white", markeredgewidth=1.2, solid_capstyle="round")
    eje.axhline(tasa_general, color=_TINTA_SUAVE, linewidth=1.2, linestyle=(0, (4, 3)))
    eje.text(meses[-1], tasa_general + 0.012, f"todos los clientes: {_pct(tasa_general)}",
             ha="right", va="bottom", color=_TINTA_SUAVE, fontsize=12.5)
    _eje_porcentaje(eje, 0.5)
    eje.set_xlim(0.3, meses[-1] + 0.7)
    eje.set_xticks([1, 6, 12, 18, 24, 30, 36])
    eje.set_xlabel("Meses como cliente (antigüedad)")
    eje.set_title("Tasa de cancelación según la antigüedad del cliente")
    figura.tight_layout()
    plt.show()


@_seguro
def grafica_trampa(*args):
    """Dos paneles con el mismo par de modelos: la exactitud se parece; a quién encuentran, no."""
    import matplotlib.pyplot as plt

    nombres = ["exactitud_base", "exactitud_arbol", "encontrados_base", "encontrados_arbol",
               "y_prueba"]
    valores = _variables(nombres, args, 3)
    if valores is None:
        return
    numeros = [_numero(v) for v in valores[:4]]
    if any(n is None or n == "varios" for n in numeros):
        print("·  primero completa el ejercicio 3: su revisión tiene que salir sin ✗")
        return
    exactitud_base, exactitud_arbol, encontrados_base, encontrados_arbol = numeros
    cancelaron = int(sum(valores[4])) if not isinstance(valores[4], (int, float)) else int(valores[4])
    _estilo()
    etiquetas = ["Línea base\n(siempre «No»)", "Árbol de\ndos niveles"]
    colores = [_TINTA_SUAVE, _MORADO]
    figura, (izq, der) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    exactitudes = [exactitud_base, exactitud_arbol]
    izq.bar(etiquetas, exactitudes, color=colores, width=0.45)
    for i, v in enumerate(exactitudes):
        izq.text(i, v + 0.02, _pct(v), ha="center", va="bottom", fontsize=14, fontweight="bold")
    _eje_porcentaje(izq, 1.0)
    izq.set_title("Exactitud en prueba")
    encontrados = [int(encontrados_base), int(encontrados_arbol)]
    der.bar(etiquetas, encontrados, color=colores, width=0.45)
    for i, v in enumerate(encontrados):
        der.text(i, v + cancelaron * 0.02, f"{v} de {cancelaron}", ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
    der.set_ylim(0, cancelaron)
    der.set_title(f"De los {cancelaron} que cancelaron,\n¿a cuántos encontró?")
    for eje in (izq, der):
        eje.tick_params(axis="x", length=0, labelsize=13, labelcolor=_TINTA)
    figura.tight_layout(w_pad=3)
    plt.show()
