import pandas as pd
import streamlit as st

from pdf_generator import generar_pdf
from perceptron import Perceptron, fmt

st.set_page_config(page_title="Perceptrón monocapa", layout="wide")

ENTRADAS = [(0, 0), (0, 1), (1, 0), (1, 1)]
COMPUERTAS = {"AND": [0, 0, 0, 1]}


def p(v):
    """Número entre paréntesis si es negativo."""
    s = fmt(v)
    return f"({s})" if s.startswith("-") else s


def esquema(w1, w2, b, x1=None, x2=None, z=None, y=None):
    """Esquema del perceptrón (SVG). Si hay x1, x2, z, y muestra la evaluación."""
    sub = lambda c, n: f'{c}<tspan baseline-shift="sub" font-size="11">{n}</tspan>'
    v1, v2 = (fmt(x1), fmt(x2)) if x1 is not None else (sub("x", 1), sub("x", 2))
    col = {1: "#2e8b57", 0: "#c0392b"}.get(y, "currentColor")
    circ = lambda cx, cy, r, f, ln="currentColor": (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{f}" stroke="{ln}" stroke-width="2"/>')
    txt = lambda x, y_, t, s=14, fill="currentColor", w="normal": (
        f'<text x="{x}" y="{y_}" font-size="{s}" text-anchor="middle" fill="{fill}" font-weight="{w}">{t}</text>')
    svg = f"""<svg viewBox="0 0 760 290" xmlns="http://www.w3.org/2000/svg"
 style="width:100%;max-width:760px;color:inherit;font-family:sans-serif">
<g stroke="currentColor" stroke-width="2" fill="none">
 <line x1="108" y1="95" x2="292" y2="141"/><line x1="108" y1="235" x2="292" y2="189"/>
 <line x1="330" y1="76" x2="330" y2="108" stroke-dasharray="5 4"/>
 <line x1="372" y1="165" x2="470" y2="165"/><line x1="570" y1="165" x2="650" y2="165"/>
 <polygon points="650,159 662,165 650,171" fill="currentColor"/></g>
{circ(80, 95, 28, "rgba(52,152,219,.2)")}{circ(80, 235, 28, "rgba(52,152,219,.2)")}
{circ(330, 56, 20, "rgba(241,196,15,.25)")}{circ(330, 165, 42, "rgba(155,89,182,.22)")}
<rect x="470" y="125" width="100" height="80" rx="8" fill="rgba(230,126,34,.2)" stroke="currentColor" stroke-width="2"/>
<polyline points="485,183 520,183 520,147 555,147" fill="none" stroke="currentColor" stroke-width="3"/>
{circ(700, 165, 32, "rgba(46,204,113,.2)", col)}
{txt(80, 101, v1, 18)}{txt(80, 241, v2, 18)}{txt(330, 62, 1, 16)}{txt(330, 178, "Σ", 36)}
{txt(700, 173, y if y is not None else "y", 22, col, "bold")}
{txt(168, 84, f"{sub('W', 1)} = {fmt(w1)}")}{txt(168, 262, f"{sub('W', 2)} = {fmt(w2)}")}
{txt(330, 20, f"b = {fmt(b)}")}{txt(330, 240, f"Z = {fmt(z)}" if z is not None else "Z", 13)}
{txt(520, 224, "Escalón", 12)}{txt(520, 241, "f(Z)=1 si Z ≥ 0", 11)}{txt(700, 215, "Salida", 12)}
</svg>"""
    st.markdown(f"<div>{svg}</div>", unsafe_allow_html=True)


def latex_muestra(s, eta):
    e, et = p(s.e), p(eta)
    return rf"""\begin{{aligned}}
Z &= (x_1 \cdot W_1) + (x_2 \cdot W_2) + b = ({fmt(s.x1)} \cdot {p(s.w1_prev)}) + ({fmt(s.x2)} \cdot {p(s.w2_prev)}) + {p(s.b_prev)} = {fmt(s.z)} \\
y &= f(Z) = f({fmt(s.z)}) = {s.y} \qquad (Z \ge 0 \Rightarrow 1,\; Z < 0 \Rightarrow 0) \\
e &= t - y = {s.t} - {s.y} = {s.e} \\
W_1(n) &= W_1(\text{{prev}}) + \eta \cdot e \cdot x_1 = {p(s.w1_prev)} + {et} \cdot {e} \cdot {fmt(s.x1)} = {fmt(s.w1_new)} \\
W_2(n) &= W_2(\text{{prev}}) + \eta \cdot e \cdot x_2 = {p(s.w2_prev)} + {et} \cdot {e} \cdot {fmt(s.x2)} = {fmt(s.w2_new)} \\
b(n) &= b(\text{{prev}}) + \eta \cdot e = {p(s.b_prev)} + {et} \cdot {e} = {fmt(s.b_new)}
\end{{aligned}}"""


st.title("Perceptrón monocapa — simulador paso a paso")

# ------------------- Configuración en la barra lateral izquierda (REQ 2) -------------------
with st.sidebar:
    st.subheader("Configuración")
    gate = st.selectbox("Compuerta predefinida", list(COMPUERTAS))
    st.caption("Edita la columna t para definir tu propia tabla de verdad.")
    tabla = st.data_editor(
        pd.DataFrame({"x1": [a for a, _ in ENTRADAS], "x2": [b for _, b in ENTRADAS],
                      "t": COMPUERTAS[gate]}),
        key=f"tabla_{gate}", hide_index=True, disabled=["x1", "x2"], use_container_width=True,
        column_config={"t": st.column_config.NumberColumn("t (objetivo)", min_value=0, max_value=1, step=1)})
    objetivos = [int(t) for t in tabla["t"]]
    nombre = gate if objetivos == COMPUERTAS[gate] else "Personalizada"
    st.markdown(f"**Compuerta actual:** {nombre}")

    st.markdown("**Parámetros iniciales**")
    a, b = st.columns(2)
    w1 = a.number_input("w₁", value=0.1, step=0.1, format="%.3f")
    w2 = b.number_input("w₂", value=0.3, step=0.1, format="%.3f")
    bias = a.number_input("Sesgo b", value=0.2, step=0.1, format="%.3f")
    eta = b.number_input("Tasa η", value=0.1, min_value=0.001, step=0.05, format="%.3f")
    max_ep = st.number_input("Límite de épocas", value=20, min_value=1, max_value=1000)

    if st.button("Entrenar", type="primary"):
        dataset = [(x, t) for x, t in zip(ENTRADAS, objetivos)]
        modelo = Perceptron(w1, w2, bias, eta)
        res = modelo.train(int(max_ep), dataset)
        pdf = generar_pdf(res.steps, res.epochs, w1_inicial=w1, w2_inicial=w2, b_inicial=bias,
                          eta=eta, max_epocas=int(max_ep), convergio=res.convergio, compuerta=nombre)
        st.session_state["run"] = dict(modelo=modelo, res=res, pdf=pdf, eta=eta, nombre=nombre)

run = st.session_state.get("run")

# ---------------------------- Entrenamiento (REQ 4 y 5) ----------------------------
st.subheader("Entrenamiento")
if not run:
    st.info("Configura los parámetros y pulsa **Entrenar**.")
else:
    res = run["res"]
    resumen = f"W₁ = {fmt(res.w1)}, W₂ = {fmt(res.w2)}, b = {fmt(res.b)}"
    if res.convergio:
        st.success(f"Convergió en {res.epocas_ejecutadas} época(s). {resumen}")
    else:
        st.warning(f"No convergió en {res.epocas_ejecutadas} época(s). {resumen}")
    st.dataframe(pd.DataFrame([{"Época": e.epoca, "W₁": fmt(e.w1), "W₂": fmt(e.w2),
                                "Bias (b)": fmt(e.b), "Errores": e.errores}
                               for e in res.epochs]),
                 hide_index=True, use_container_width=True,
                 height=min(38 * (len(res.epochs) + 1) + 3, 600))
    st.download_button("Descargar informe PDF", run["pdf"],
                       file_name=f"informe_perceptron_{run['nombre']}.pdf",
                       mime="application/pdf")

# ---------------------------- Cálculos por época (REQ 3) ----------------------------
st.subheader("Cálculos por época")
if not run:
    st.caption("Aquí aparecerá el procedimiento de cada época.")
else:
    for e in run["res"].epochs:
        with st.expander(f"Época {e.epoca} — errores: {e.errores}", expanded=(e.epoca == 1)):
            for s in (s for s in run["res"].steps if s.epoca == e.epoca):
                st.markdown(f"**Muestra {s.muestra}:** $(x_1, x_2) = ({fmt(s.x1)}, {fmt(s.x2)})$, $t = {s.t}$")
                st.latex(latex_muestra(s, run["eta"]))
            st.markdown(f"Al terminar: $W_1 = {fmt(e.w1)}$, $W_2 = {fmt(e.w2)}$, $b = {fmt(e.b)}$")

# ---------------------------- Esquema y prueba (REQ 1 y 6) ----------------------------
st.divider()
izq, der = st.columns([2, 1], gap="large")
with der:
    st.subheader("Módulo de inferencia")
    if not run:
        st.info("Entrena el perceptrón para habilitar la prueba.")
    else:
        m = run["modelo"]
        a, b = st.columns(2)
        x1 = a.number_input("x₁", value=1.0, step=1.0, key="x1_test")
        x2 = b.number_input("x₂", value=1.0, step=1.0, key="x2_test")
        z, y = m.predict(x1, x2)
        st.latex(rf"Z = ({fmt(x1)} \cdot {p(m.w1)}) + ({fmt(x2)} \cdot {p(m.w2)}) + {p(m.b)} = {fmt(z)}"
                 rf"\;\Rightarrow\; y = f({fmt(z)}) = {y}")
with izq:
    st.subheader("Esquema del perceptrón y prueba")
    if not run:
        st.caption("Pesos iniciales configurados.")
        esquema(w1, w2, bias)
    else:
        st.caption("Pesos finales tras el entrenamiento y evaluación de la entrada de prueba.")
        esquema(m.w1, m.w2, m.b, x1, x2, z, y)

