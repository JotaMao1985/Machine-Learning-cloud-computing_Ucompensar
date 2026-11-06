"""Ayudas comunes a todos los notebooks del curso (van en la celda oculta, tras descarga_datos.py).

Las de cada sesión están en ayudas_sesionN.py.
"""


def _aviso_version():
    """Las diapositivas se ejecutaron con pandas 2.2 y scikit-learn 1.6: si Colab trae otras, se dice."""
    import pandas as pd
    import sklearn

    if pd.__version__.split(".")[:2] != ["2", "2"] or sklearn.__version__.split(".")[:2] != ["1", "6"]:
        print(f"Nota: este Colab trae pandas {pd.__version__} y scikit-learn {sklearn.__version__}; "
              "las diapositivas se hicieron con pandas 2.2 y scikit-learn 1.6. Si alguna salida "
              "se ve un poco distinta, es por eso.")


_aviso_version()
