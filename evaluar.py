"""Corre los 20 mensajes de la bandeja contra el golden set y saca los dos números.

    uv run python evaluar.py                  corre lo que falte y escribe evaluacion.json
    uv run python evaluar.py --de-nuevo       vuelve a correr los 20 casos
    uv run python evaluar.py --pausa 10       espera 10 s entre mensajes (cuota del free tier)

Los dos números que pide la consigna:
1. cuántos de los 20 mensajes salieron por la rama correcta, y
2. en cuántas de las 8 consultas aparece el dato clave.

Si un caso falla (por ejemplo, por cuota agotada) queda anotado con su error y la
próxima corrida lo reintenta sin repetir los que ya salieron bien.
"""

import argparse
import json
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from mesa_pedidos import config
from mesa_pedidos.grafo import construir_grafo

SALIDA = config.RAIZ / "evaluacion.json"

# Hoja en la que terminó el grafo, según el `resultado` que escribe cada una
RUTA_POR_RESULTADO = {
    "respondido": "responder_consulta",
    "confirmado": "confirmar_pedido",
    "rechazado": "rechazar_pedido",
    "derivado": "derivar_humano",
}


def leer_bandeja() -> list[dict]:
    with config.BANDEJA_JSONL.open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que "Garantía" y "garantia" cuenten igual."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower()


def correr_caso(grafo, mensaje: dict) -> dict:
    """Procesa un mensaje y devuelve qué hizo el sistema. No mira el golden set."""
    configuracion = {"configurable": {"thread_id": mensaje["id"]}}
    grafo.invoke({"mensaje": mensaje, "log": []}, configuracion)
    estado = grafo.get_state(configuracion)
    valores = estado.values
    # Si quedó frenado en el interrupt, la rama es la de aprobación humana
    ruta = "aprobacion_humana" if estado.interrupts else RUTA_POR_RESULTADO[valores["resultado"]]
    respuesta = valores.get("respuesta") or {}
    pedido = valores.get("pedido") or {}
    return {
        "intencion": valores.get("intencion"),
        "ruta": ruta,
        "respuesta": respuesta.get("respuesta"),
        "citas": respuesta.get("citas", []),
        "items": [{"sku": i["sku"], "cantidad": i["cantidad"]} for i in pedido.get("items", [])],
        "alertas": (valores.get("validacion") or {}).get("alertas", []),
        "log": valores.get("log", []),
    }


def comparar(obtenido: dict, esperado: dict) -> dict:
    """Compara un caso contra su etiqueta del golden set."""
    comparacion = {
        "intencion_ok": obtenido["intencion"] == esperado["intencion"],
        "ruta_ok": obtenido["ruta"] == esperado["ruta"],
    }
    if esperado["intencion"] == "consulta":
        texto = normalizar(obtenido["respuesta"] or "")
        faltan = [d for d in esperado.get("debe_contener", []) if normalizar(d) not in texto]
        comparacion["dato_clave_ok"] = not faltan
        comparacion["datos_faltantes"] = faltan
        documentos = {c["documento"] for c in obtenido["citas"]}
        comparacion["documento_ok"] = esperado.get("documento") in documentos
    if esperado.get("items"):
        # Extra de la consigna: precisión y recall de los renglones extraídos
        extraidos = {(i["sku"].upper(), i["cantidad"]) for i in obtenido["items"]}
        reales = {(i["sku"].upper(), i["cantidad"]) for i in esperado["items"]}
        correctos = len(extraidos & reales)
        comparacion["items_precision"] = correctos / len(extraidos) if extraidos else 0.0
        comparacion["items_recall"] = correctos / len(reales)
    return comparacion


def resumir(casos: list[dict]) -> dict:
    evaluados = [c for c in casos if "error" not in c]
    consultas = [c for c in evaluados if c["esperado"]["intencion"] == "consulta"]
    con_items = [c for c in evaluados if "items_recall" in c["comparacion"]]

    def promedio(clave: str) -> float | None:
        if not con_items:
            return None
        return round(sum(c["comparacion"][clave] for c in con_items) / len(con_items), 3)

    return {
        "casos_evaluados": len(evaluados),
        "casos_con_error": len(casos) - len(evaluados),
        "ruta_correcta": sum(c["comparacion"]["ruta_ok"] for c in evaluados),
        "intencion_correcta": sum(c["comparacion"]["intencion_ok"] for c in evaluados),
        "consultas_evaluadas": len(consultas),
        "consultas_con_dato_clave": sum(c["comparacion"]["dato_clave_ok"] for c in consultas),
        "consultas_con_documento_citado": sum(c["comparacion"]["documento_ok"] for c in consultas),
        "items_precision_promedio": promedio("items_precision"),
        "items_recall_promedio": promedio("items_recall"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--de-nuevo", action="store_true", help="Corre los 20 casos desde cero")
    parser.add_argument("--pausa", type=float, default=5.0, help="Segundos entre mensajes")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="Archivo de resultados")
    args = parser.parse_args()

    golden = {c["id"]: c for c in json.loads(config.GOLDEN_SET_JSON.read_text("utf-8"))["casos"]}
    anteriores = {}
    if args.salida.exists() and not args.de_nuevo:
        previo = json.loads(args.salida.read_text("utf-8"))
        anteriores = {c["id"]: c for c in previo["casos"] if "error" not in c}

    # InMemorySaver alcanza: acá no se retoma nada, solo hace falta para el interrupt
    grafo = construir_grafo(checkpointer=InMemorySaver())
    casos = []
    for mensaje in leer_bandeja():
        if mensaje["id"] in anteriores:
            casos.append(anteriores[mensaje["id"]])
            continue
        esperado = golden[mensaje["id"]]
        print(f"{mensaje['id']} ...", end=" ", flush=True)
        try:
            obtenido = correr_caso(grafo, mensaje)
        except Exception as error:  # noqa: BLE001 - un caso roto no frena la evaluación
            print(f"ERROR {type(error).__name__}")
            casos.append({"id": mensaje["id"], "esperado": esperado, "error": str(error)[:500]})
            continue
        comparacion = comparar(obtenido, esperado)
        print(f"{obtenido['ruta']} {'✓' if comparacion['ruta_ok'] else '✗'}")
        casos.append(
            {
                "id": mensaje["id"],
                "esperado": esperado,
                "obtenido": obtenido,
                "comparacion": comparacion,
            }
        )
        time.sleep(args.pausa)

    metricas = resumir(casos)
    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo_llm": config.MODELO_LLM,
        "modelo_embeddings": config.MODELO_EMBEDDINGS,
        "retriever": "denso",
        "metricas": metricas,
        "casos": casos,
    }
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nRama correcta: {metricas['ruta_correcta']}/{metricas['casos_evaluados']}")
    print(
        f"Dato clave en consultas: {metricas['consultas_con_dato_clave']}"
        f"/{metricas['consultas_evaluadas']}"
    )
    if metricas["casos_con_error"]:
        print(
            f"{metricas['casos_con_error']} casos con error: volvé a correrlo para reintentarlos."
        )
    print(f"Detalle en {args.salida.name}")


if __name__ == "__main__":
    main()
