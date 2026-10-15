"""Los nodos sin LLM, probados con un estado armado a mano (consigna, paso 3)."""

from mesa_pedidos import reglas
from mesa_pedidos.nodos import (
    cargar_cliente,
    ruta_por_validacion,
    ruta_post_aprobacion,
    validar_pedido,
)


def estado_con_pedido(cuit, items, forma_pago="no_indica", plazo=None, descuento=None):
    estado = {"mensaje": {"id": "TEST", "cuit": cuit}}
    estado.update(cargar_cliente(estado))
    estado["pedido"] = {
        "items": [{"sku": sku, "cantidad": cantidad, "descripcion": ""} for sku, cantidad in items],
        "forma_pago": forma_pago,
        "plazo_dias": plazo,
        "descuento_solicitado": descuento,
        "destino": None,
    }
    return estado


def test_escala_de_descuento_por_volumen():
    assert reglas.descuento_por_volumen(999_999) == 0
    assert reglas.descuento_por_volumen(1_000_000) == 5
    assert reglas.descuento_por_volumen(6_500_000) == 12


def test_cargar_cliente_con_cuit_desconocido_deja_none():
    resultado = cargar_cliente({"mensaje": {"cuit": "20-00000000-0"}})
    assert resultado["cliente"] is None
    assert len(resultado["log"]) == 1


def test_pedido_rutinario_se_confirma_con_los_totales_calculados():
    # El Tornillo Feliz: 12% habitual, neto de lista 1.152.500 → +5% por volumen
    estado = estado_con_pedido("30-71455982-2", [("AB-1020", 250), ("AB-1120", 50)])
    validacion = validar_pedido(estado)["validacion"]

    assert validacion["decision"] == "confirmar"
    assert validacion["descuento_volumen"] == 5
    assert validacion["renglones"][0]["descuento"] == 17
    assert validacion["neto"] == round(1_152_500 * 0.83, 2)
    assert validacion["total"] == round(validacion["neto"] * 1.21, 2)


def test_las_marcas_con_acuerdo_tienen_tope_de_15():
    # Talleres Patagonia: 15% habitual + 5% por volumen = 20%, pero Dewalt topea en 15%
    estado = estado_con_pedido("30-69874512-2", [("HE-4110", 4)])
    assert validar_pedido(estado)["validacion"]["renglones"][0]["descuento"] == 15


def test_si_supera_el_credito_va_a_aprobacion_humana():
    # Metalúrgica San Jorge tiene 1.550.000 disponibles
    estado = estado_con_pedido("30-70988341-7", [("HE-4180", 3)], "cuenta_corriente")
    validacion = validar_pedido(estado)["validacion"]

    assert validacion["decision"] == "humano"
    assert "crédito disponible" in validacion["alertas"][0]


def test_descuento_pedido_por_encima_del_tope_va_a_aprobacion():
    estado = estado_con_pedido("30-71455982-2", [("AB-1020", 100)], descuento=25)
    assert validar_pedido(estado)["validacion"]["decision"] == "humano"


def test_cliente_suspendido_en_cuenta_corriente_se_rechaza():
    estado = estado_con_pedido("30-71788450-3", [("SE-5090", 10)], "cuenta_corriente", 30)
    assert validar_pedido(estado)["validacion"]["decision"] == "rechazar"


def test_cliente_suspendido_que_paga_anticipado_no_se_rechaza():
    estado = estado_con_pedido("30-71788450-3", [("SE-5090", 10)], "anticipado")
    assert validar_pedido(estado)["validacion"]["decision"] == "confirmar"


def test_un_sku_inexistente_se_deriva_con_sugerencias():
    estado = estado_con_pedido("30-71455982-2", [("HE-4331", 1)])
    validacion = validar_pedido(estado)["validacion"]

    assert validacion["decision"] == "derivar"
    assert "HE-4330" in validacion["alertas"][0]


def test_los_routers_solo_traducen_la_decision():
    assert ruta_por_validacion({"validacion": {"decision": "humano"}}) == "humano"
    assert ruta_post_aprobacion({"decision_humana": {"aprobado": False}}) == "rechaza"
