"""Retriever híbrido: búsqueda densa + BM25, fusionadas con Reciprocal Rank Fusion (Clase 5).

- **Densa** (FAISS + embeddings): encuentra textos que *significan* algo parecido,
  aunque no compartan palabras. Falla con códigos: para un embedding, HE-4330 y
  HE-4335 son casi el mismo punto.
- **BM25**: cuenta palabras exactas, pesadas por cuán raras son. Encuentra el
  `HE-4330` literal, pero no entiende sinónimos.

RRF junta los dos rankings sin tener que comparar sus puntajes (que están en escalas
distintas): cada documento suma 1 / (k + posición) por cada lista en la que aparece.
Lo que está arriba en las dos listas gana.
"""

import re
import unicodedata

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from mesa_pedidos.indexado import cargar_indice, cargar_paginas, partir_en_chunks

# Valor estándar de RRF: amortigua la diferencia entre el puesto 1 y el 2
K_RRF = 60


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin tildes, y los códigos como HE-4330 quedan enteros."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", sin_tildes.lower())


def clave(documento: Document) -> tuple:
    """Identifica un chunk sin importar de qué retriever vino."""
    return (documento.metadata["documento"], documento.metadata["pagina"], documento.page_content)


def fusionar_rrf(rankings: list[list[Document]], k: int = K_RRF) -> list[Document]:
    puntajes: dict[tuple, float] = {}
    documentos: dict[tuple, Document] = {}
    for ranking in rankings:
        for posicion, documento in enumerate(ranking, start=1):
            c = clave(documento)
            puntajes[c] = puntajes.get(c, 0.0) + 1 / (k + posicion)
            documentos[c] = documento
    orden = sorted(puntajes, key=puntajes.get, reverse=True)
    return [documentos[c] for c in orden]


class RetrieverHibrido(BaseRetriever):
    denso: BaseRetriever
    bm25: BaseRetriever
    cantidad: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        rankings = [self.denso.invoke(query), self.bm25.invoke(query)]
        return fusionar_rrf(rankings)[: self.cantidad]


def crear_bm25(chunks: list[Document], cantidad: int) -> BM25Retriever:
    return BM25Retriever.from_documents(chunks, k=cantidad, preprocess_func=tokenizar)


def crear_retriever_hibrido(cantidad: int = 4) -> RetrieverHibrido:
    # Cada retriever trae el doble de candidatos y la fusión se queda con los mejores.
    # BM25 no necesita embeddings: se arma en memoria con los mismos chunks del índice.
    candidatos = cantidad * 2
    denso = cargar_indice().as_retriever(search_kwargs={"k": candidatos})
    bm25 = crear_bm25(partir_en_chunks(cargar_paginas()), candidatos)
    return RetrieverHibrido(denso=denso, bm25=bm25, cantidad=cantidad)
