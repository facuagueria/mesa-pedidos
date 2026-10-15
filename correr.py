"""Procesa un mensaje de la bandeja y maneja la aprobación humana.

    uv run python correr.py MSG-009                          procesa el mensaje
    uv run python correr.py MSG-013 --aprobar "cliente histórico"   retoma y aprueba
    uv run python correr.py MSG-013 --rechazar "sin crédito"        retoma y rechaza
    uv run python correr.py MSG-009 --de-nuevo               borra el checkpoint y reprocesa

El `thread_id` es el id del mensaje. Como el checkpointer es SQLite sobre archivo, un
pedido que quedó esperando aprobación se puede retomar desde otra terminal u otro día.
"""

import argparse
import json
import sys

from langgraph.types import Command

from mesa_pedidos import config
from mesa_pedidos.grafo import abrir_checkpointer, construir_grafo


def cargar_mensaje(mensaje_id: str) -> dict:
    with config.BANDEJA_JSONL.open(encoding="utf-8") as archivo:
        for linea in archivo:
            mensaje = json.loads(linea)
            if mensaje["id"] == mensaje_id:
                return mensaje
    sys.exit(f"No existe el mensaje {mensaje_id} en {config.BANDEJA_JSONL.name}")


def ejecutar(grafo, entrada, configuracion: dict) -> None:
    """Corre el grafo mostrando cada nodo apenas termina, para ver por dónde va."""
    for actualizacion in grafo.stream(entrada, configuracion, stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            if nodo == "__interrupt__":
                continue
            for linea in (cambios or {}).get("log", []):
                print(linea, flush=True)


def mostrar(grafo, configuracion: dict) -> None:
    estado = grafo.get_state(configuracion)
    if estado.interrupts:
        print("\n⏸  Esperando aprobación de Gerencia Comercial:")
        print(json.dumps(estado.interrupts[0].value, indent=2, ensure_ascii=False))
        return
    valores = estado.values
    print(f"\nResultado: {valores.get('resultado')}")
    if respuesta := valores.get("respuesta"):
        print(respuesta["respuesta"])
        for cita in respuesta["citas"]:
            print(f"  - {cita['documento']}, pág. {cita['pagina']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mensaje_id", help="Id del mensaje en la bandeja, por ejemplo MSG-009")
    accion = parser.add_mutually_exclusive_group()
    accion.add_argument("--aprobar", metavar="COMENTARIO", help="Retoma el pedido y lo aprueba")
    accion.add_argument("--rechazar", metavar="COMENTARIO", help="Retoma el pedido y lo rechaza")
    accion.add_argument("--de-nuevo", action="store_true", help="Borra el checkpoint y reprocesa")
    args = parser.parse_args()

    checkpointer = abrir_checkpointer()
    grafo = construir_grafo(checkpointer=checkpointer)
    configuracion = {"configurable": {"thread_id": args.mensaje_id}}
    estado = grafo.get_state(configuracion)

    if args.aprobar is not None or args.rechazar is not None:
        if not estado.interrupts:
            sys.exit(f"{args.mensaje_id} no está esperando aprobación.")
        aprobado = args.aprobar is not None
        comentario = args.aprobar if aprobado else args.rechazar
        ejecutar(
            grafo, Command(resume={"aprobado": aprobado, "comentario": comentario}), configuracion
        )
    elif args.de_nuevo or not estado.values:
        checkpointer.delete_thread(args.mensaje_id)
        ejecutar(grafo, {"mensaje": cargar_mensaje(args.mensaje_id), "log": []}, configuracion)
    elif estado.next and not estado.interrupts:
        # Se cortó a mitad de camino (Ctrl+C, un error): el checkpoint guarda hasta el
        # último nodo que terminó, y con entrada None el grafo sigue desde ahí.
        print("\n".join(estado.values.get("log", [])))
        print(f"(quedó a medias, retomo desde {', '.join(estado.next)})")
        ejecutar(grafo, None, configuracion)
    elif estado.interrupts:
        print(f"{args.mensaje_id} sigue esperando aprobación:\n")
        print("\n".join(estado.values.get("log", [])))
    else:
        print(f"{args.mensaje_id} ya se procesó (usá --de-nuevo para reprocesarlo):\n")
        print("\n".join(estado.values.get("log", [])))

    mostrar(grafo, configuracion)


if __name__ == "__main__":
    main()
