"""Tests de las tools contra los CSV reales."""

from mesa_pedidos.tools import (
    buscar_producto,
    consultar_credito,
    consultar_producto,
    verificar_credito,
)


def test_buscar_producto_devuelve_los_datos_del_catalogo():
    producto = buscar_producto("AB-1020")

    assert producto["encontrado"] is True
    assert producto["precio_lista"] == 3250.0
    assert producto["stock"] == 1840
    assert producto["deposito"] == "Cordoba"


def test_buscar_producto_acepta_minusculas_y_espacios():
    assert buscar_producto(" he-4330 ")["sku"] == "HE-4330"


def test_un_sku_inexistente_trae_sugerencias():
    resultado = buscar_producto("HE-4331")

    assert resultado["encontrado"] is False
    assert "HE-4330" in resultado["sugerencias"]


def test_consultar_credito_calcula_el_disponible():
    # El Tornillo Feliz: límite 4.500.000, saldo 1.280.000
    credito = consultar_credito("30-71455982-2", monto_neto=1_000_000)

    assert credito["razon_social"] == "Ferreteria El Tornillo Feliz S.R.L."
    assert credito["credito_disponible"] == 3_220_000
    assert credito["alcanza_credito"] is True


def test_un_pedido_que_supera_el_disponible_no_alcanza():
    # Metalúrgica San Jorge: límite 9.000.000, saldo 7.450.000 → disponible 1.550.000
    assert consultar_credito("30-70988341-7", monto_neto=2_000_000)["alcanza_credito"] is False


def test_el_cuit_se_reconoce_con_o_sin_guiones():
    assert consultar_credito("30714559822")["encontrado"] is True


def test_un_cuit_desconocido_devuelve_error_y_no_una_excepcion():
    resultado = consultar_credito("20-00000000-0")

    assert resultado["encontrado"] is False
    assert "error" in resultado


def test_las_tools_exponen_su_esquema_y_se_pueden_invocar():
    assert consultar_producto.name == "consultar_producto"
    assert set(verificar_credito.args) == {"cuit", "monto_neto"}

    resultado = verificar_credito.invoke({"cuit": "30-71455982-2", "monto_neto": 500_000})
    assert resultado["alcanza_credito"] is True
