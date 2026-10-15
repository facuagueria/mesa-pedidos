"""La comparación contra el golden set, con casos armados a mano."""

from evaluar import comparar, resumir


def obtenido(**cambios):
    base = {"intencion": "consulta", "ruta": "responder_consulta", "respuesta": "", "citas": []}
    base.update({"items": [], **cambios})
    return base


def test_el_dato_clave_se_busca_sin_importar_tildes_ni_mayusculas():
    esperado = {
        "intencion": "consulta",
        "ruta": "responder_consulta",
        "debe_contener": ["Garantía"],
    }
    comparacion = comparar(obtenido(respuesta="La garantia es de 12 meses"), esperado)

    assert comparacion["ruta_ok"]
    assert comparacion["dato_clave_ok"]


def test_precision_y_recall_de_los_items():
    esperado = {
        "intencion": "pedido",
        "ruta": "confirmar_pedido",
        "items": [{"sku": "AB-1020", "cantidad": 200}, {"sku": "AB-1120", "cantidad": 50}],
    }
    extraidos = [{"sku": "AB-1020", "cantidad": 200}]
    comparacion = comparar(obtenido(intencion="pedido", items=extraidos), esperado)

    assert comparacion["items_precision"] == 1.0
    assert comparacion["items_recall"] == 0.5
    assert not comparacion["ruta_ok"]


def test_los_casos_con_error_no_cuentan_en_las_metricas():
    casos = [
        {
            "id": "A",
            "esperado": {"intencion": "derivar"},
            "comparacion": {"ruta_ok": True, "intencion_ok": True},
        },
        {"id": "B", "esperado": {"intencion": "consulta"}, "error": "429"},
    ]
    metricas = resumir(casos)

    assert metricas["casos_evaluados"] == 1
    assert metricas["casos_con_error"] == 1
    assert metricas["ruta_correcta"] == 1
