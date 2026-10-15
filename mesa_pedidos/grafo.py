"""El cableado: nodos, aristas, routers y checkpointer (Clase 7).

Acá no hay lógica de negocio. Este archivo solo dice qué nodo va después de cuál,
y en las bifurcaciones qué string del router lleva a qué nodo. Así el ruteo queda
declarado en el grafo, se puede dibujar y se puede evaluar.

Para ver el dibujo:

    uv run python -m mesa_pedidos.grafo
"""

import sqlite3
from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from mesa_pedidos import config, nodos
from mesa_pedidos.estado import EstadoPedido
from mesa_pedidos.rag import crear_llm, crear_retriever

# Los nodos que llaman al LLM reintentan solo ante errores pasajeros: cuota agotada
# del free tier (429), servicio no disponible (503) o timeout. Un error de otro tipo
# (una clave inválida, un esquema que Gemini no acepta) no se arregla reintentando,
# así que sale enseguida y se ve.
ERRORES_TRANSITORIOS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "timed out", "Timeout")


def es_error_transitorio(error: Exception) -> bool:
    texto = f"{type(error).__name__}: {error}"
    return any(marca in texto for marca in ERRORES_TRANSITORIOS)


REINTENTOS_LLM = RetryPolicy(
    max_attempts=3, initial_interval=5.0, backoff_factor=2.0, retry_on=es_error_transitorio
)


def abrir_checkpointer() -> SqliteSaver:
    """Checkpointer sobre archivo: sobrevive a cerrar la terminal (InMemorySaver no)."""
    config.DIR_ARTEFACTOS.mkdir(exist_ok=True)
    conexion = sqlite3.connect(config.CHECKPOINTS_DB, check_same_thread=False)
    return SqliteSaver(conexion)


def construir_grafo(
    llm: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    llm = llm or crear_llm()
    retriever = retriever or crear_retriever()

    grafo = StateGraph(EstadoPedido)

    grafo.add_node("cargar_cliente", nodos.cargar_cliente)
    grafo.add_node("clasificar", partial(nodos.clasificar, llm=llm), retry_policy=REINTENTOS_LLM)
    grafo.add_node(
        "responder_consulta",
        partial(nodos.responder_consulta, llm=llm, retriever=retriever),
        retry_policy=REINTENTOS_LLM,
    )
    grafo.add_node(
        "extraer_pedido", partial(nodos.extraer_pedido, llm=llm), retry_policy=REINTENTOS_LLM
    )
    grafo.add_node("validar_pedido", nodos.validar_pedido)
    grafo.add_node("aprobacion_humana", nodos.aprobacion_humana)
    grafo.add_node("confirmar_pedido", nodos.confirmar_pedido)
    grafo.add_node("rechazar_pedido", nodos.rechazar_pedido)
    grafo.add_node("derivar_humano", nodos.derivar_humano)

    grafo.add_edge(START, "cargar_cliente")
    grafo.add_edge("cargar_cliente", "clasificar")

    # Los tres routers: función que devuelve un string + mapa de string a nodo
    grafo.add_conditional_edges(
        "clasificar",
        nodos.ruta_por_intencion,
        {
            "consulta": "responder_consulta",
            "pedido": "extraer_pedido",
            "derivar": "derivar_humano",
        },
    )
    grafo.add_edge("extraer_pedido", "validar_pedido")
    grafo.add_conditional_edges(
        "validar_pedido",
        nodos.ruta_por_validacion,
        {
            "confirmar": "confirmar_pedido",
            "humano": "aprobacion_humana",
            "rechazar": "rechazar_pedido",
            "derivar": "derivar_humano",
        },
    )
    grafo.add_conditional_edges(
        "aprobacion_humana",
        nodos.ruta_post_aprobacion,
        {"aprueba": "confirmar_pedido", "rechaza": "rechazar_pedido"},
    )

    # Las hojas
    for hoja in ("responder_consulta", "confirmar_pedido", "rechazar_pedido", "derivar_humano"):
        grafo.add_edge(hoja, END)

    return grafo.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.vectorstores import InMemoryVectorStore

    # Para dibujarlo no hace falta ni el LLM ni el índice de verdad
    retriever_falso = InMemoryVectorStore(DeterministicFakeEmbedding(size=8)).as_retriever()
    grafo = construir_grafo(llm=FakeListChatModel(responses=[""]), retriever=retriever_falso)
    print(grafo.get_graph().draw_mermaid())
