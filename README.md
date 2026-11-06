# Machine Learning y Seguridad en la Nube

Material de clase del curso *Machine Learning y Seguridad en la Nube* (Especialización en Big Data,
UCompensar). Son cinco sesiones sincrónicas, con Python en Google Colab y Microsoft Azure, que siguen un
mismo caso: **Yarumo Fibra S.A.S.**, un operador regional de internet fijo **ficticio**. Sus datos se
generan con una semilla fija y no corresponden a personas ni a empresas reales.

**Material en línea:** <https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/>

| Sesión | Fecha | Tema | Diapositivas | Notebook |
|---|---|---|---|---|
| 1 | viernes 6 de noviembre de 2026 · 4 h | Del dato al modelo, del equipo a la nube: diagnóstico, computación en la nube (IaaS, PaaS, SaaS, responsabilidad compartida), qué es aprender de datos, entrenamiento y prueba, línea base | [Ver](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-1-del-dato-al-modelo.html) | [Abrir en Colab](https://colab.research.google.com/github/JotaMao1985/Machine-Learning-cloud-computing_Ucompensar/blob/main/notebooks/sesion-1-del-dato-al-modelo.ipynb) |
| 2 | sábado 7 de noviembre de 2026 · 6 h | Aprender con etiqueta: regresión y clasificación, métricas, sobreajuste y validación cruzada; identidades y acceso en Azure (Entra ID, RBAC, Azure Policy) | [Ver](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-2-aprender-con-etiqueta.html) | [Abrir en Colab](https://colab.research.google.com/github/JotaMao1985/Machine-Learning-cloud-computing_Ucompensar/blob/main/notebooks/sesion-2-aprender-con-etiqueta.ipynb) |
| 3 | viernes 13 de noviembre de 2026 · 4 h | Grupos sin etiqueta | Próximamente | |
| 4 | sábado 14 de noviembre de 2026 · 6 h | Redes y datos en Azure | Próximamente | |
| 5 | viernes 20 de noviembre de 2026 · 4 h | Aprender por ensayo y error, y sustentación | Próximamente | |

**Proyecto final:** un plan de adopción segura de Azure para un modelo de Yarumo, en parejas.
[Enunciado y rúbrica](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/proyecto.html).

Cada sesión tiene:

- **diapositivas de clase** en HTML: las flechas avanzan, `F` pone la pantalla completa, `M` abre el
  índice y `P`, la vista del presentador;
- **un cuaderno (*notebook*) de Google Colab** con las mismas celdas que se proyectan y una práctica
  guiada cuyos ejercicios se revisan solos. Después de abrirlo, hay que guardar una copia en Drive para
  conservar los cambios. La primera celda descarga `datos/clientes.csv` del repositorio y comprueba su
  huella SHA-256, así que no hay que subir nada.

## Cómo se construye

Las diapositivas y los notebooks salen de `fuente/` y `precalculo/`; no se editan a mano en `Htmls/` ni en
`notebooks/`. Las cifras que se proyectan salen de ejecutar el notebook en un entorno con las mismas
versiones de Colab (pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.4.1).

Material diseñado por Javier Mauricio Sierra.
