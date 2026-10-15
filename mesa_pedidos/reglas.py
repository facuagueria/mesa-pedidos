"""Las reglas comerciales que usa `validar_pedido`, cada una con el artículo de donde sale.

Son números y cuentas, así que van en Python y no en el LLM (Clase 6: los LLM no
son calculadoras confiables). Las respuestas a consultas sí salen del RAG: estas
constantes son solo las que el sistema necesita para decidir sobre un pedido.
"""

# Condiciones Comerciales, art. 3: pedido mínimo facturable, neto sin IVA
PEDIDO_MINIMO = {"A": 250_000, "B": 250_000, "C": 120_000}

# Art. 4: escala de descuento por volumen sobre el neto de lista (desde, porcentaje)
ESCALA_VOLUMEN = [(6_000_000, 12.0), (3_000_000, 8.0), (1_000_000, 5.0)]

# Art. 5: tope del descuento total, y el tope más bajo para ciertas marcas de
# herramienta eléctrica por acuerdo con el fabricante
TOPE_DESCUENTO = 20.0
TOPE_DESCUENTO_MARCAS = 15.0
MARCAS_CON_TOPE = {"Dewalt", "Makita", "Bosch"}
RUBRO_CON_TOPE = "Herramienta electrica"

# Art. 7: días de mora a partir de los cuales cambia lo que se puede hacer
MORA_REQUIERE_AUTORIZACION = 30
MORA_SUSPENDE = 60


def descuento_por_volumen(neto_lista: float) -> float:
    """El porcentaje de la escala del art. 4 que corresponde a ese neto."""
    for desde, porcentaje in ESCALA_VOLUMEN:
        if neto_lista >= desde:
            return porcentaje
    return 0.0


def tope_descuento(producto: dict) -> float:
    """El descuento total máximo para un producto (art. 5)."""
    if producto["rubro"] == RUBRO_CON_TOPE and producto["marca"] in MARCAS_CON_TOPE:
        return TOPE_DESCUENTO_MARCAS
    return TOPE_DESCUENTO


def forma_pago_efectiva(forma_pago: str, cliente: dict) -> str:
    """Si el mensaje no dice cómo paga, se asume la condición habitual del cliente (art. 6)."""
    if forma_pago != "no_indica":
        return forma_pago
    return "cuenta_corriente" if cliente["dias_plazo"] > 0 else "anticipado"
