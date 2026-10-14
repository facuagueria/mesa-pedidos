"""Las dos tools que consultan los datos de la empresa (Clase 6).

Primero están como funciones comunes (`buscar_producto`, `consultar_credito`), que se
pueden probar sin LangChain. Después se envuelven con `@tool` y un `args_schema` de
Pydantic: eso es lo que las convierte en herramientas con un contrato declarado.

En este proyecto no las llama un agente que decide solo: las llama `validar_pedido`,
en el orden que definimos nosotros. Esa es la diferencia entre la Clase 6 y la 7.
"""

import csv
from difflib import get_close_matches
from functools import cache
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from mesa_pedidos import config

# --- Lectura de los CSV ------------------------------------------------------------


def _leer_csv(ruta: Path) -> list[dict[str, str]]:
    # utf-8-sig descarta el BOM que agrega Excel; sin esto la primera columna
    # se llamaría "﻿sku" en vez de "sku".
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


@cache
def cargar_catalogo() -> dict[str, dict]:
    """El catálogo indexado por SKU, con los números ya convertidos."""
    return {
        fila["sku"]: {
            "sku": fila["sku"],
            "nombre": fila["nombre"],
            "rubro": fila["rubro"],
            "marca": fila["marca"],
            "unidad": fila["unidad"],
            "precio_lista": float(fila["precio_lista"]),
            "alicuota_iva": float(fila["alicuota_iva"]),
            "stock": int(fila["stock"]),
            "deposito": fila["deposito"],
        }
        for fila in _leer_csv(config.CATALOGO_CSV)
    }


@cache
def cargar_clientes() -> dict[str, dict]:
    """Los clientes indexados por CUIT (solo dígitos), con los números ya convertidos."""
    return {
        normalizar_cuit(fila["cuit"]): {
            "cuit": fila["cuit"],
            "razon_social": fila["razon_social"],
            "provincia": fila["provincia"],
            "localidad": fila["localidad"],
            "condicion_iva": fila["condicion_iva"],
            "categoria": fila["categoria"],
            "limite_credito": float(fila["limite_credito"]),
            "saldo_actual": float(fila["saldo_actual"]),
            "dias_plazo": int(fila["dias_plazo"]),
            "descuento_habitual": float(fila["descuento_habitual"]),
            "estado": fila["estado"],
            "mora_dias": int(fila["mora_dias"]),
        }
        for fila in _leer_csv(config.CLIENTES_CSV)
    }


def normalizar_cuit(cuit: str) -> str:
    """'30-71455982-2' y '30714559822' son el mismo CUIT: nos quedamos con los dígitos."""
    return "".join(c for c in cuit if c.isdigit())


# --- Funciones comunes -------------------------------------------------------------


def buscar_producto(sku: str) -> dict:
    """Devuelve los datos del producto, o un error con SKUs parecidos para sugerir."""
    catalogo = cargar_catalogo()
    sku = sku.strip().upper()
    producto = catalogo.get(sku)
    if producto:
        return {"encontrado": True, **producto}
    # Un error accionable (Clase 6): no solo "no existe", sino qué se puede probar.
    return {
        "encontrado": False,
        "sku": sku,
        "error": f"No existe el SKU {sku} en el catálogo.",
        "sugerencias": get_close_matches(sku, list(catalogo), n=3, cutoff=0.6),
    }


def consultar_credito(cuit: str, monto_neto: float = 0.0) -> dict:
    """Devuelve la situación crediticia del cliente y si `monto_neto` entra en su crédito.

    Crédito disponible = límite − saldo actual (Condiciones Comerciales, art. 7).
    La tool solo informa los datos; qué hacer con la mora o el estado lo decide
    `validar_pedido`, que es donde viven las reglas.
    """
    cliente = cargar_clientes().get(normalizar_cuit(cuit))
    if cliente is None:
        return {
            "encontrado": False,
            "cuit": cuit,
            "error": f"No hay ningún cliente con CUIT {cuit}.",
        }
    credito_disponible = cliente["limite_credito"] - cliente["saldo_actual"]
    return {
        "encontrado": True,
        **cliente,
        "credito_disponible": credito_disponible,
        "monto_neto": monto_neto,
        "alcanza_credito": monto_neto <= credito_disponible,
    }


# --- Las mismas funciones como tools -----------------------------------------------


class ConsultarProductoArgs(BaseModel):
    sku: str = Field(description="Código del producto, por ejemplo 'AB-1020'")


class VerificarCreditoArgs(BaseModel):
    cuit: str = Field(
        description="CUIT del cliente, con o sin guiones, por ejemplo '30-71455982-2'"
    )
    monto_neto: float = Field(
        default=0.0,
        ge=0,
        description="Monto neto del pedido sin IVA, en pesos, para saber si entra en el crédito",
    )


@tool(args_schema=ConsultarProductoArgs)
def consultar_producto(sku: str) -> dict:
    """Busca un producto del catálogo por SKU.

    Devuelve nombre, marca, rubro, precio de lista sin IVA, alícuota de IVA, stock
    disponible y depósito. Si el SKU no existe, devuelve encontrado=False y
    sugerencias de SKUs parecidos.
    """
    return buscar_producto(sku)


@tool(args_schema=VerificarCreditoArgs)
def verificar_credito(cuit: str, monto_neto: float = 0.0) -> dict:
    """Consulta la cuenta corriente de un cliente por CUIT.

    Devuelve razón social, categoría, límite de crédito, saldo actual, crédito
    disponible, días de mora, estado de la cuenta, descuento habitual y si el
    monto neto indicado entra en el crédito disponible.
    """
    return consultar_credito(cuit, monto_neto)
