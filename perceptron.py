from dataclasses import dataclass, field
from typing import List, Tuple

# Tabla de verdad de la compuerta AND: ((x1, x2), t)
AND_DATASET: List[Tuple[Tuple[int, int], int]] = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

# Se redondea para evitar ruido de punto flotante (p. ej. -1e-17 en lugar de 0),
# lo cual podría alterar la comparación z >= 0 del escalón.
_DECIMALES = 10


def _r(valor: float) -> float:
    return round(valor + 0.0, _DECIMALES)  # "+ 0.0" convierte -0.0 en 0.0


def fmt(valor: float) -> str:
    """Formatea un número sin truncar información útil (sin ceros sobrantes)."""
    texto = f"{_r(valor):.10f}".rstrip("0").rstrip(".")
    return "0" if texto in ("", "-0") else texto


@dataclass
class StepRecord:
    """Registro completo de una muestra evaluada dentro de una época."""
    epoca: int
    muestra: int          # 1..4
    x1: float
    x2: float
    t: int                # salida deseada
    w1_prev: float
    w2_prev: float
    b_prev: float
    z: float
    y: int
    e: int
    w1_new: float
    w2_new: float
    b_new: float


@dataclass
class EpochRecord:
    """Resumen de una época: pesos/bias al terminar y errores cometidos."""
    epoca: int
    w1: float
    w2: float
    b: float
    errores: int


@dataclass
class TrainingResult:
    w1: float
    w2: float
    b: float
    convergio: bool
    epocas_ejecutadas: int
    steps: List[StepRecord] = field(default_factory=list)
    epochs: List[EpochRecord] = field(default_factory=list)


class Perceptron:
    """Perceptrón de 2 entradas con función de activación escalón."""

    def __init__(self, w1: float, w2: float, b: float, eta: float):
        self.w1 = _r(w1)
        self.w2 = _r(w2)
        self.b = _r(b)
        self.eta = eta
        self.w1_inicial = self.w1
        self.w2_inicial = self.w2
        self.b_inicial = self.b
        self.history_steps: List[StepRecord] = []
        self.history_epochs: List[EpochRecord] = []

    # ------------------------------------------------------------------ #
    # Operaciones básicas
    # ------------------------------------------------------------------ #
    @staticmethod
    def activacion(z: float) -> int:
        """Función escalón: 1 si z >= 0, de lo contrario 0."""
        return 1 if _r(z) >= 0 else 0

    def suma_ponderada(self, x1: float, x2: float) -> float:
        """z = (x1 * w1) + (x2 * w2) + b"""
        return _r(x1 * self.w1 + x2 * self.w2 + self.b)

    def predict(self, x1: float, x2: float) -> Tuple[float, int]:
        """Devuelve (z, y) para una entrada cualquiera."""
        z = self.suma_ponderada(x1, x2)
        return z, self.activacion(z)

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def train(self, max_epocas: int, dataset=None) -> TrainingResult:
        """
        Entrena con la regla delta. Se detiene al llegar a una época sin
        errores (convergencia) o al alcanzar `max_epocas`.
        """
        dataset = dataset or AND_DATASET
        self.history_steps = []
        self.history_epochs = []
        convergio = False

        for epoca in range(1, max_epocas + 1):
            errores = 0
            for idx, ((x1, x2), t) in enumerate(dataset, start=1):
                w1_prev, w2_prev, b_prev = self.w1, self.w2, self.b

                z = self.suma_ponderada(x1, x2)
                y = self.activacion(z)
                e = t - y
                if e != 0:
                    errores += 1

                self.w1 = _r(w1_prev + self.eta * e * x1)
                self.w2 = _r(w2_prev + self.eta * e * x2)
                self.b = _r(b_prev + self.eta * e)

                self.history_steps.append(StepRecord(
                    epoca=epoca, muestra=idx, x1=x1, x2=x2, t=t,
                    w1_prev=w1_prev, w2_prev=w2_prev, b_prev=b_prev,
                    z=z, y=y, e=e,
                    w1_new=self.w1, w2_new=self.w2, b_new=self.b,
                ))

            self.history_epochs.append(EpochRecord(
                epoca=epoca, w1=self.w1, w2=self.w2, b=self.b, errores=errores,
            ))
            if errores == 0:
                convergio = True
                break

        return TrainingResult(
            w1=self.w1, w2=self.w2, b=self.b,
            convergio=convergio,
            epocas_ejecutadas=len(self.history_epochs),
            steps=self.history_steps,
            epochs=self.history_epochs,
        )
