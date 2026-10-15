"""El grafo de punta a punta con un LLM falso: ruteo, interrupt y retomar."""

import sqlite3

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from mesa_pedidos.estado import Clasificacion, ItemPedido, PedidoExtraido
from mesa_pedidos.grafo import construir_grafo, es_error_transitorio
from mesa_pedidos.rag import Cita, RespuestaConsulta


class LLMFalso:
    """Devuelve una respuesta fija por cada modelo de Pydantic que se le pide."""

    def __init__(self, *respuestas):
        self.respuestas = {type(r): r for r in respuestas}

    def with_structured_output(self, esquema):
        return RunnableLambda(lambda _: self.respuestas[esquema])


class RetrieverFalso(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        texto = "Zona 5 - Patagonia: flete $175.000"
        return [Document(texto, metadata={"documento": "logistica.pdf", "pagina": 1})]


def mensaje(cuit, texto="..."):
    return {"mensaje": {"id": "TEST", "cuit": cuit, "texto": texto}, "log": []}


def pedido(*items, forma_pago="cuenta_corriente"):
    return PedidoExtraido(
        items=[ItemPedido(sku=sku, cantidad=cantidad, descripcion="") for sku, cantidad in items],
        forma_pago=forma_pago,
    )


CONFIG = {"configurable": {"thread_id": "TEST"}}


def test_una_consulta_se_responde_con_cita():
    llm = LLMFalso(
        Clasificacion(intencion="consulta", motivo="pregunta por el flete"),
        RespuestaConsulta(
            respuesta="$175.000",
            citas=[Cita(documento="logistica.pdf", pagina=1)],
            contexto_suficiente=True,
        ),
    )
    grafo = construir_grafo(llm=llm, retriever=RetrieverFalso())

    final = grafo.invoke(mensaje("30-69874512-2"))

    assert final["resultado"] == "respondido"
    assert final["respuesta"]["citas"] == [{"documento": "logistica.pdf", "pagina": 1}]
    # Con el reducer, el log tiene una línea por nodo y no solo la última
    assert [linea.split(":")[0] for linea in final["log"]] == [
        "cargar_cliente",
        "clasificar",
        "responder_consulta",
    ]


def test_un_reclamo_se_deriva_sin_pasar_por_el_pedido():
    llm = LLMFalso(Clasificacion(intencion="derivar", motivo="reclamo por faltante"))
    final = construir_grafo(llm=llm, retriever=RetrieverFalso()).invoke(mensaje("30-71455982-2"))

    assert final["resultado"] == "derivado"
    assert "pedido" not in final


def test_un_pedido_rutinario_se_confirma_sin_frenar():
    llm = LLMFalso(
        Clasificacion(intencion="pedido", motivo="pide discos"), pedido(("AB-1020", 100))
    )
    grafo = construir_grafo(llm=llm, retriever=RetrieverFalso(), checkpointer=InMemorySaver())

    final = grafo.invoke(mensaje("30-71455982-2"), CONFIG)

    assert final["resultado"] == "confirmado"
    assert grafo.get_state(CONFIG).interrupts == ()


def test_un_pedido_sin_credito_frena_y_se_retoma_desde_otro_proceso(tmp_path):
    llm = LLMFalso(
        Clasificacion(intencion="pedido", motivo="pide amoladoras"), pedido(("HE-4180", 3))
    )
    archivo = tmp_path / "checkpoints.sqlite"

    # Primera "terminal": corre hasta el interrupt y se cierra
    with sqlite3.connect(archivo, check_same_thread=False) as conexion:
        grafo = construir_grafo(llm, RetrieverFalso(), SqliteSaver(conexion))
        grafo.invoke(mensaje("30-70988341-7"), CONFIG)
        payload = grafo.get_state(CONFIG).interrupts[0].value

    assert payload["razon_social"] == "Metalurgica San Jorge S.A."
    assert payload["credito_disponible"] == 1_550_000
    assert payload["renglones"][0]["sku"] == "HE-4180"
    assert payload["alertas"]

    # Segunda "terminal": otro grafo, otra conexión, mismo archivo y mismo thread_id
    with sqlite3.connect(archivo, check_same_thread=False) as conexion:
        grafo = construir_grafo(llm, RetrieverFalso(), SqliteSaver(conexion))
        final = grafo.invoke(Command(resume={"aprobado": False, "comentario": "sin cupo"}), CONFIG)

    assert final["resultado"] == "rechazado"
    assert final["log"][-1] == "rechazar_pedido: Gerencia lo rechazó: sin cupo"


def test_si_gerencia_aprueba_se_confirma():
    llm = LLMFalso(
        Clasificacion(intencion="pedido", motivo="pide amoladoras"), pedido(("HE-4180", 3))
    )
    grafo = construir_grafo(llm=llm, retriever=RetrieverFalso(), checkpointer=InMemorySaver())

    grafo.invoke(mensaje("30-70988341-7"), CONFIG)
    final = grafo.invoke(Command(resume={"aprobado": True, "comentario": "ok"}), CONFIG)

    assert final["resultado"] == "confirmado"


def test_solo_se_reintentan_los_errores_pasajeros():
    assert es_error_transitorio(Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert not es_error_transitorio(Exception("400 INVALID_ARGUMENT: schema"))
