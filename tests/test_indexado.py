"""Tests del indexado que no llaman a ninguna API: usan embeddings falsos."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from mesa_pedidos import indexado


def test_cada_pagina_trae_documento_y_numero_de_pagina():
    paginas = indexado.cargar_paginas()

    assert len(paginas) == 10  # 3 + 3 + 2 + 2 páginas
    primera = paginas[0]
    assert primera.metadata == {"documento": "condiciones_comerciales_2026.pdf", "pagina": 1}


def test_los_chunks_conservan_la_metadata_y_respetan_el_tamanio():
    chunks = indexado.partir_en_chunks(indexado.cargar_paginas())

    assert len(chunks) > 10
    for chunk in chunks:
        assert {"documento", "pagina"} <= chunk.metadata.keys()
        assert len(chunk.page_content) <= indexado.TAMANIO_CHUNK


def test_el_indice_se_guarda_y_se_vuelve_a_cargar(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)

    creado = indexado.construir_indice(embeddings, dir_indice=tmp_path)
    cargado = indexado.cargar_indice(embeddings, dir_indice=tmp_path)

    assert (tmp_path / "index.faiss").is_file()
    assert cargado.index.ntotal == creado.index.ntotal
