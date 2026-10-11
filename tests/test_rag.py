"""Tests de las piezas del RAG que no necesitan llamar al LLM."""

from langchain_core.documents import Document

from mesa_pedidos.rag import Cita, RespuestaConsulta, filtrar_citas, formatear_contexto

CHUNKS = [
    Document(
        page_content="Zona 5 - Patagonia: flete $175.000 por envio.",
        metadata={"documento": "logistica_fletes_2026.pdf", "pagina": 1},
    ),
    Document(
        page_content="Herramienta electrica: 12 meses de garantia.",
        metadata={"documento": "politica_devoluciones_2026.pdf", "pagina": 1},
    ),
]


def test_el_contexto_lleva_documento_y_pagina_delante_de_cada_chunk():
    contexto = formatear_contexto(CHUNKS)

    assert "[logistica_fletes_2026.pdf, pág. 1]\nZona 5" in contexto
    assert "[politica_devoluciones_2026.pdf, pág. 1]" in contexto


def test_se_descartan_las_citas_a_paginas_que_no_se_recuperaron():
    respuesta = RespuestaConsulta(
        respuesta="El flete a Neuquén sale $175.000.",
        citas=[
            Cita(documento="logistica_fletes_2026.pdf", pagina=1),
            Cita(documento="logistica_fletes_2026.pdf", pagina=9),  # inventada
        ],
        contexto_suficiente=True,
    )

    filtrada = filtrar_citas(respuesta, CHUNKS)

    assert filtrada.citas == [Cita(documento="logistica_fletes_2026.pdf", pagina=1)]
