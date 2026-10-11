"""Indexado de los PDF de políticas (Clase 4).

Fase 1 del RAG: PDF → páginas → chunks → embeddings → índice FAISS en disco.
Se corre una sola vez (o cuando cambian los PDF o el modelo de embeddings):

    uv run python -m mesa_pedidos.indexado

Cada chunk guarda en su metadata el `documento` y la `pagina` de donde salió,
que es lo que después permite citar la fuente en la respuesta.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from mesa_pedidos import config

# ~800 caracteres alcanzan para un artículo corto de las políticas; el overlap de
# 120 (15 %) evita cortar una regla a la mitad entre dos chunks (Clase 4).
TAMANIO_CHUNK = 800
SOLAPAMIENTO_CHUNK = 120

# Archivo donde se anota con qué modelo se generó el índice. Si alguien cambia
# MODELO_EMBEDDINGS, el índice viejo no sirve: sus vectores viven en otro espacio.
ARCHIVO_MODELO = "modelo_embeddings.txt"


def cargar_paginas(dir_politicas: Path = config.DIR_POLITICAS) -> list[Document]:
    """Lee cada PDF y devuelve un Document por página, con documento y página en la metadata."""
    paginas = []
    for ruta_pdf in sorted(dir_politicas.glob("*.pdf")):
        lector = PdfReader(ruta_pdf)
        for numero, pagina in enumerate(lector.pages, start=1):
            texto = (pagina.extract_text() or "").strip()
            if not texto:
                continue
            paginas.append(
                Document(
                    page_content=texto,
                    metadata={"documento": ruta_pdf.name, "pagina": numero},
                )
            )
    return paginas


def partir_en_chunks(paginas: list[Document]) -> list[Document]:
    """Divide las páginas en chunks. Cada chunk hereda la metadata de su página."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TAMANIO_CHUNK,
        chunk_overlap=SOLAPAMIENTO_CHUNK,
    )
    return splitter.split_documents(paginas)


def crear_embeddings() -> Embeddings:
    """El modelo de embeddings. Indexado y consultas TIENEN que usar el mismo (Clase 4)."""
    return GoogleGenerativeAIEmbeddings(model=config.MODELO_EMBEDDINGS)


def construir_indice(
    embeddings: Embeddings | None = None,
    dir_politicas: Path = config.DIR_POLITICAS,
    dir_indice: Path = config.DIR_INDICE,
) -> FAISS:
    """Indexa los PDF desde cero y guarda el índice en disco."""
    embeddings = embeddings or crear_embeddings()
    chunks = partir_en_chunks(cargar_paginas(dir_politicas))
    indice = FAISS.from_documents(chunks, embeddings)
    indice.save_local(str(dir_indice))
    (dir_indice / ARCHIVO_MODELO).write_text(config.MODELO_EMBEDDINGS, encoding="utf-8")
    return indice


def cargar_indice(
    embeddings: Embeddings | None = None,
    dir_indice: Path = config.DIR_INDICE,
) -> FAISS:
    """Carga el índice persistido; si no existe o es de otro modelo, lo construye."""
    embeddings = embeddings or crear_embeddings()
    archivo_modelo = dir_indice / ARCHIVO_MODELO
    if archivo_modelo.is_file() and archivo_modelo.read_text(encoding="utf-8") == (
        config.MODELO_EMBEDDINGS
    ):
        # El índice lo generamos nosotros, así que deserializarlo es seguro.
        return FAISS.load_local(str(dir_indice), embeddings, allow_dangerous_deserialization=True)
    return construir_indice(embeddings, dir_indice=dir_indice)


if __name__ == "__main__":
    indice = construir_indice()
    print(f"Índice guardado en {config.DIR_INDICE} con {indice.index.ntotal} chunks.")
