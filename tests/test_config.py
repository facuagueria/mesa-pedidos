from mesa_pedidos import config


def test_existen_los_datos_de_entrada():
    for ruta in (
        config.CATALOGO_CSV,
        config.CLIENTES_CSV,
        config.BANDEJA_JSONL,
        config.GOLDEN_SET_JSON,
    ):
        assert ruta.is_file(), f"Falta {ruta.name} en datos/"


def test_estan_los_cuatro_pdf_de_politicas():
    assert len(list(config.DIR_POLITICAS.glob("*.pdf"))) == 4
