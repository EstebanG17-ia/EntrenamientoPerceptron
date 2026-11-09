from io import BytesIO
from datetime import datetime
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from perceptron import EpochRecord, StepRecord, fmt

ETA = "eta"  # las fuentes estándar no incluyen la letra griega; se escribe "eta"
AZUL = colors.HexColor("#1f4e79")
GRIS = colors.HexColor("#f2f2f2")


def _estilos():
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("titulo", parent=base["Title"], textColor=AZUL, fontSize=20),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], textColor=AZUL, fontSize=14,
                             spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=AZUL, fontSize=11.5,
                             spaceBefore=8, spaceAfter=4),
        "normal": ParagraphStyle("normal", parent=base["Normal"], fontSize=10, leading=14),
        "formula": ParagraphStyle("formula", parent=base["Normal"], fontSize=9.5, leading=14,
                                  leftIndent=14),
        "pequeno": ParagraphStyle("pequeno", parent=base["Normal"], fontSize=8.5,
                                  textColor=colors.grey),
    }


def _p(v: float) -> str:
    """Número entre paréntesis si es negativo, para mostrar operaciones claras."""
    s = fmt(v)
    return f"({s})" if s.startswith("-") else s


def _bloque_muestra(s: StepRecord, eta: float, est) -> KeepTogether:
    """Desglose matemático completo de una muestra (sin truncar datos)."""
    f = est["formula"]
    eta_s = _p(eta)
    e_s = _p(s.e)

    filas = [
        Paragraph(f"<b>Muestra {s.muestra}</b>: (x<sub>1</sub>, x<sub>2</sub>) = "
                  f"({fmt(s.x1)}, {fmt(s.x2)}), &nbsp; t = {s.t}", est["normal"]),
        Paragraph(
            f"Z = (x<sub>1</sub>&middot;W<sub>1</sub>) + (x<sub>2</sub>&middot;W<sub>2</sub>) + b "
            f"= ({fmt(s.x1)}&middot;{_p(s.w1_prev)}) + ({fmt(s.x2)}&middot;{_p(s.w2_prev)}) "
            f"+ {_p(s.b_prev)} = <b>{fmt(s.z)}</b>", f),
        Paragraph(
            f"y = f(Z) = f({fmt(s.z)}) = <b>{s.y}</b> &nbsp; (f(Z) = 1 si Z &gt;= 0; "
            f"f(Z) = 0 si Z &lt; 0)", f),
        Paragraph(f"e = t - y = {s.t} - {s.y} = <b>{s.e}</b>", f),
        Paragraph(
            f"W<sub>1</sub>(n) = W<sub>1</sub>(prev) + {ETA}&middot;e&middot;x<sub>1</sub> "
            f"= {_p(s.w1_prev)} + {eta_s}&middot;{e_s}&middot;{fmt(s.x1)} = <b>{fmt(s.w1_new)}</b>", f),
        Paragraph(
            f"W<sub>2</sub>(n) = W<sub>2</sub>(prev) + {ETA}&middot;e&middot;x<sub>2</sub> "
            f"= {_p(s.w2_prev)} + {eta_s}&middot;{e_s}&middot;{fmt(s.x2)} = <b>{fmt(s.w2_new)}</b>", f),
        Paragraph(
            f"b(n) = b(prev) + {ETA}&middot;e "
            f"= {_p(s.b_prev)} + {eta_s}&middot;{e_s} = <b>{fmt(s.b_new)}</b>", f),
        Spacer(1, 6),
    ]
    return KeepTogether(filas)


def _tabla_epocas(epochs: List[EpochRecord]) -> Table:
    datos = [["Época", "W1", "W2", "Bias (b)", "Errores"]]
    for ep in epochs:
        datos.append([str(ep.epoca), fmt(ep.w1), fmt(ep.w2), fmt(ep.b), str(ep.errores)])

    tabla = Table(datos, repeatRows=1, hAlign="CENTER",
                  colWidths=[2.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 2.4 * cm])
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tabla


def generar_pdf(steps: List[StepRecord], epochs: List[EpochRecord], *,
                w1_inicial: float, w2_inicial: float, b_inicial: float,
                eta: float, max_epocas: int, convergio: bool, compuerta: str = "AND") -> bytes:
    """Construye el informe y devuelve el PDF como bytes (listo para descargar)."""
    est = _estilos()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Informe del Perceptrón Monocapa - Compuerta {compuerta}",
    )
    story = []

    # --- Portada / resumen ------------------------------------------------
    story.append(Paragraph("Informe del Perceptrón Monocapa", est["titulo"]))
    story.append(Paragraph(f"Aprendizaje de la compuerta lógica {compuerta}", est["h2"]))
    story.append(Paragraph(f"Generado el {datetime.now():%d/%m/%Y %H:%M}", est["pequeno"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Parámetros iniciales", est["h1"]))
    story.append(Paragraph(
        f"W<sub>1</sub> = {fmt(w1_inicial)} &nbsp;|&nbsp; W<sub>2</sub> = {fmt(w2_inicial)} "
        f"&nbsp;|&nbsp; b = {fmt(b_inicial)} &nbsp;|&nbsp; {ETA} = {fmt(eta)} "
        f"&nbsp;|&nbsp; Épocas máximas = {max_epocas}", est["normal"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Modelo: Z = (x<sub>1</sub>&middot;W<sub>1</sub>) + (x<sub>2</sub>&middot;W<sub>2</sub>) + b; "
        f"y = f(Z) con función escalón (1 si Z &gt;= 0, 0 en otro caso); e = t - y.",
        est["normal"]))

    ultimo = epochs[-1]
    estado = ("Convergió: la última época no cometió errores."
              if convergio else "No convergió dentro del límite de épocas.")
    story.append(Paragraph("2. Resultado final", est["h1"]))
    story.append(Paragraph(
        f"W<sub>1</sub> = <b>{fmt(ultimo.w1)}</b>, W<sub>2</sub> = <b>{fmt(ultimo.w2)}</b>, "
        f"b = <b>{fmt(ultimo.b)}</b> tras {len(epochs)} época(s). {estado}", est["normal"]))

    # --- Tabla por épocas -------------------------------------------------
    story.append(Paragraph("3. Tabla resumen por épocas", est["h1"]))
    story.append(_tabla_epocas(epochs))

    # --- Procedimiento paso a paso ---------------------------------------
    story.append(Paragraph("4. Procedimiento paso a paso", est["h1"]))
    story.append(Paragraph(
        "Para cada muestra se calcula la suma ponderada Z, la salida y = f(Z), el error "
        f"e = t - y y se aplica la regla delta: W<sub>i</sub>(n) = W<sub>i</sub>(prev) + "
        f"{ETA}&middot;e&middot;x<sub>i</sub>; b(n) = b(prev) + {ETA}&middot;e.", est["normal"]))

    por_epoca = {}
    for s in steps:
        por_epoca.setdefault(s.epoca, []).append(s)

    for ep in epochs:
        story.append(Paragraph(
            f"Época {ep.epoca} &mdash; errores: {ep.errores}", est["h2"]))
        for s in por_epoca.get(ep.epoca, []):
            story.append(_bloque_muestra(s, eta, est))
        story.append(Paragraph(
            f"<i>Al terminar la época {ep.epoca}: W<sub>1</sub> = {fmt(ep.w1)}, "
            f"W<sub>2</sub> = {fmt(ep.w2)}, b = {fmt(ep.b)}</i>", est["normal"]))

    def _pie(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_pie, onLaterPages=_pie)
    return buffer.getvalue()
