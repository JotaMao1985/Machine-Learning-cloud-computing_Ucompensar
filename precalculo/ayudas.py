"""Ayudas comunes a todos los notebooks del curso (van en la celda oculta, tras descarga_datos.py).

Las de cada sesión están en ayudas_sesionN.py.
"""


def _aviso_version():
    """Las diapositivas se ejecutaron con pandas 2.2 y scikit-learn 1.6; con otras, algo cambia."""
    import pandas as pd
    import sklearn

    if int(pd.__version__.split(".")[0]) >= 3:
        print(f"Aviso: este Colab trae pandas {pd.__version__} y las diapositivas usan pandas 2.2. "
              "Algunas salidas se verán un poco distintas (por ejemplo, el texto sale como str "
              "y no como object).")
    if sklearn.__version__.split(".")[:2] != ["1", "6"]:
        print(f"Aviso: este Colab trae scikit-learn {sklearn.__version__} y las diapositivas usan "
              "la 1.6. Algunas cifras de los modelos pueden cambiar en los últimos decimales.")


_aviso_version()
