"""El estado que viaja por el grafo y las salidas estructuradas del LLM (Clases 2 y 7).

`EstadoPedido` es el único canal entre nodos: lo que no quedó escrito acá no se
puede auditar después. Cada nodo devuelve solo los campos que cambió.

Los modelos de Pydantic son el contrato con el LLM: `with_structured_output`
garantiza la forma de la respuesta, no la verdad. Por eso después `validar_pedido`
igual contrasta todo contra los CSV.
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# --- Salidas estructuradas del LLM -------------------------------------------------


class Clasificacion(BaseModel):
    """Qué tipo de mensaje mandó el cliente."""

    intencion: Literal["consulta", "pedido", "derivar"] = Field(
        description=(
            "consulta: pregunta algo que se responde con las políticas de la empresa "
            "(precios, fletes, plazos, garantías, devoluciones, productos). "
            "pedido: quiere comprar productos concretos. "
            "derivar: reclamos por entregas o facturas, propuestas comerciales, pedidos "
            "de asesoramiento o llamado, o cualquier cosa que necesite a una persona."
        )
    )
    motivo: str = Field(description="En una oración, por qué se eligió esa intención")


class ItemPedido(BaseModel):
    sku: str = Field(description="Código del producto tal como lo escribió el cliente")
    # ge=1 y no gt=0: Gemini documenta `minimum` en sus esquemas pero no `exclusiveMinimum`
    cantidad: int = Field(ge=1, description="Cantidad pedida, en unidades del catálogo")
    descripcion: str = Field(description="Cómo nombró el cliente al producto")


class PedidoExtraido(BaseModel):
    """Los datos del pedido tal como aparecen en el mensaje, sin completar nada."""

    items: list[ItemPedido] = Field(description="Un renglón por cada producto pedido")
    forma_pago: Literal["cuenta_corriente", "anticipado", "no_indica"] = Field(
        description=(
            "cuenta_corriente si pide plazo (30 días, 45 días, 'a cuenta corriente'). "
            "anticipado si paga antes del envío (transferencia, contado anticipado). "
            "no_indica si el mensaje no lo dice."
        )
    )
    plazo_dias: int | None = Field(
        default=None, description="Días de plazo que pide el cliente, si los menciona"
    )
    descuento_solicitado: float | None = Field(
        default=None,
        description="Porcentaje de descuento total que pide el cliente, si lo pide (25 = 25%)",
    )
    destino: str | None = Field(default=None, description="Lugar de entrega, si lo menciona")


# --- El estado del grafo -----------------------------------------------------------


class EstadoPedido(TypedDict, total=False):
    # Entrada: el mensaje tal como viene de bandeja.jsonl
    mensaje: dict
    # cargar_cliente: la fila de clientes.csv, o None si el CUIT no existe
    cliente: dict | None
    # clasificar
    intencion: str
    motivo_clasificacion: str
    # responder_consulta: respuesta, citas y contexto_suficiente
    respuesta: dict
    # extraer_pedido: el PedidoExtraido como dict
    pedido: dict
    # validar_pedido: decisión, renglones, totales y alertas
    validacion: dict
    # aprobacion_humana: lo que contestó la persona
    decision_humana: dict
    # Hojas: cómo terminó el mensaje (respondido, confirmado, rechazado, derivado)
    resultado: str
    # Con el reducer `add` cada nodo agrega sus líneas en vez de pisar las anteriores
    log: Annotated[list[str], add]
