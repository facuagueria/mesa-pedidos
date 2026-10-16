"""El retriever híbrido, sin embeddings reales."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from mesa_pedidos.indexado import cargar_paginas, partir_en_chunks
from mesa_pedidos.retriever import RetrieverHibrido, crear_bm25, fusionar_rrf, tokenizar


def doc(texto, pagina=1):
    return Document(texto, metadata={"documento": "x.pdf", "pagina": pagina})


class RetrieverFijo(BaseRetriever):
    documentos: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.documentos


def test_tokenizar_deja_los_skus_enteros_y_saca_tildes():
    assert tokenizar("¿El HE-4330 trae batería?") == ["el", "he-4330", "trae", "bateria"]


def test_rrf_premia_lo_que_aparece_arriba_en_las_dos_listas():
    a, b, c = doc("a"), doc("b"), doc("c")
    fusion = fusionar_rrf([[a, b], [b, c]])

    assert fusion[0] == b
    assert set(d.page_content for d in fusion) == {"a", "b", "c"}


def test_bm25_encuentra_el_sku_exacto_en_las_fichas():
    # El caso del MSG-004: para un embedding HE-4330 y HE-4335 son casi lo mismo
    bm25 = crear_bm25(partir_en_chunks(cargar_paginas()), cantidad=1)
    primero = bm25.invoke("el atornillador HE-4330 viene con bateria?")[0]

    assert "HE-4330" in primero.page_content


def test_el_hibrido_mezcla_las_dos_busquedas_y_corta_en_la_cantidad():
    denso = RetrieverFijo(documentos=[doc("a"), doc("b")])
    bm25 = RetrieverFijo(documentos=[doc("c"), doc("a")])
    resultado = RetrieverHibrido(denso=denso, bm25=bm25, cantidad=2).invoke("pregunta")

    assert [d.page_content for d in resultado] == ["a", "c"]
