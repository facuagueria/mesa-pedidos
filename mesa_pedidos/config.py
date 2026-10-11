"""Rutas y configuración compartidas por todo el paquete.

Todo lo que dependa del entorno (claves, modelos) se lee de variables de entorno,
y todas las rutas se arman desde la raíz del repo para que los scripts funcionen
sin importar desde qué carpeta se ejecuten.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent

load_dotenv(RAIZ / ".env")

# Datos de entrada, tal como los entrega la cátedra
DIR_DATOS = RAIZ / "datos"
DIR_POLITICAS = DIR_DATOS / "politicas"
CATALOGO_CSV = DIR_DATOS / "catalogo.csv"
CLIENTES_CSV = DIR_DATOS / "clientes.csv"
BANDEJA_JSONL = DIR_DATOS / "bandeja.jsonl"
GOLDEN_SET_JSON = DIR_DATOS / "golden_set.json"

# Artefactos generados (no se versionan)
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_INDICE = DIR_ARTEFACTOS / "indice_faiss"
CHECKPOINTS_DB = DIR_ARTEFACTOS / "checkpoints.sqlite"

# Modelos
MODELO_LLM = os.getenv("MODELO_LLM", "gemini-3.6-flash")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "gemini-embedding-001")
