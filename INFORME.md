# Informe · Mesa de Pedidos

## 1. Qué resuelve

El sistema procesa los mensajes que llegan a la Mesa de Pedidos de Ferretería Industrial Andina por WhatsApp o mail. Para cada mensaje decide si es una **consulta**, un **pedido** o algo para **derivar** a una persona. A las consultas les responde con las políticas de la empresa, citando documento y página. Los pedidos los arma contra el catálogo y la cuenta corriente reales y aplica las Condiciones Comerciales. Según el caso, el pedido se confirma, se rechaza, se deriva o **se frena hasta que Gerencia Comercial lo apruebe**. No es un chatbot: cada mensaje es un procesamiento independiente, con su propio `thread_id`.

## 2. Arquitectura

Un `StateGraph` de LangGraph con 9 nodos, 3 routers y 5 hojas, igual al diagrama de la consigna (ver README).

- **Estado.** `EstadoPedido` es un `TypedDict` con un campo por cosa que produce cada nodo (`cliente`, `intencion`, `pedido`, `validacion`, `decision_humana`, `resultado`). Tiene además un `log` con reducer `add`, para que cada nodo agregue su línea en vez de pisar las anteriores. Los nodos devuelven solo los campos que cambian.
- **LLM solo para lenguaje.** El LLM se usa en tres lugares: `clasificar` y `extraer_pedido`, con structured output sobre modelos de Pydantic, y `responder_consulta`, con RAG. En los dos prompts el texto del cliente va delimitado y se aclara que es un dato y no una instrucción.
- **Decisiones sin LLM.** `validar_pedido` llama a las dos tools (`consultar_producto` y `verificar_credito`) en un orden fijo y aplica las reglas en Python, cada una con su artículo:
  - escala de descuento por volumen (art. 4) y tope del 20 %, o 15 % para Dewalt, Makita y Bosch (art. 5);
  - pedido mínimo (art. 3), crédito disponible y mora (arts. 6 y 7) y stock (art. 9).

  Cada regla que se dispara deja una alerta, y gana la decisión más grave: rechazar, derivar, humano o confirmar. Las cuentas de plata tienen que ser deterministas y auditables, y un LLM no es una calculadora confiable.
- **Routers declarados.** Los tres routers son funciones de una línea que leen el estado y devuelven un string. Traducen una decisión que ya tomó un nodo, así que el ruteo se ve en el grafo y se puede evaluar.
- **Aprobación humana.** `aprobacion_humana` llama a `interrupt()` con todo lo que Gerencia necesita para decidir: razón social, CUIT, categoría, mora, crédito disponible, neto, total, forma de pago, alertas y renglones. El estado queda en un `SqliteSaver` sobre archivo y el proceso termina. Se retoma desde otra terminal con `Command(resume={"aprobado": ..., "comentario": ...})`. Hay un test que frena con una conexión a SQLite, la cierra y retoma con otra.
- **RAG.** Los 4 PDF se cortan por página (800 caracteres con 120 de solapamiento) y se indexan una sola vez en FAISS, que queda persistido. Cada chunk guarda `documento` y `pagina`. El retriever es **híbrido**: búsqueda densa y BM25 fusionadas con Reciprocal Rank Fusion. Las citas que devuelve el LLM se filtran contra los chunks recuperados, porque el structured output garantiza la forma, no la verdad.
- **Robustez.** Los nodos con LLM tienen una `RetryPolicy` que reintenta solo errores pasajeros (cuota, 503, timeout), y el cliente del LLM tiene timeout. Cualquier otro error sale enseguida.

## 3. Resultados

Corrí los 20 mensajes de la bandeja con `evaluar.py` contra `golden_set.json`, con `gemini-3.5-flash-lite`. El detalle de cada caso está en `evaluacion.json` (híbrida) y `evaluacion_densa.json`.

| Configuración | Rama correcta | Intención | Dato clave en consultas | Documento citado | Renglones (precisión / recall) |
|---|---|---|---|---|---|
| Densa, primera corrida | 17/20 | 17/20 | 6/8 | 6/8 | 0,89 / 0,89 |
| Densa, versión final | 20/20 | 20/20 | 8/8 | 8/8 | 1,0 / 1,0 |
| Híbrida (BM25 + densa) | 20/20 | 20/20 | 8/8 | 8/8 | 1,0 / 1,0 |

**Qué falló en la primera corrida.** Los tres errores eran del clasificador, no del RAG ni de las reglas. Mandó a `derivar` dos consultas (devolución de zapatos de seguridad y percepción de Ingresos Brutos) y un pedido con SKUs que pedía 25 % de descuento. La descripción de la intención `derivar` terminaba en "o cualquier cosa que necesite a una persona", y el modelo la usaba de comodín. Acoté las tres definiciones: una pregunta sobre políticas es consulta aunque la respuesta sea que no, y un pedido con códigos sigue siendo pedido aunque pida condiciones especiales, porque esas las valida `validar_pedido`. Como el ajuste se hizo mirando el golden set, el 20/20 es optimista.

**Lo que el número no mide.** El chequeo del dato clave busca substrings, y eso deja pasar respuestas malas. MSG-001 contaba como acierto aunque se contradecía a mitad de texto ("bonificado al 100 %... Espere, el monto es de $1.200.000"). El modelo no tenía dónde razonar y hacía las cuentas adentro de la respuesta al cliente. Agregué al structured output un campo `razonamiento` antes de `respuesta`: el LLM genera los campos en orden, así que primero hace las cuentas ahí. Al hacerlo apareció otro bug: MSG-005 ("¿sigue valiendo el presupuesto del 10 de septiembre?") pasó a responder "no tengo esa información", porque el modelo nunca sabía la fecha del mensaje. Ahora la recibe en el prompt. Por todo esto, además del número, las 8 respuestas se revisaron a mano.

**Densa contra híbrida.** Con este golden set no hay diferencia medible. El corpus son 4 PDF de una o dos páginas, y con 4 chunks la búsqueda densa ya trae la página correcta en las 8 consultas, incluida la del SKU HE-4330, que es el caso donde BM25 debería ganar. El híbrido queda igual por defecto porque no empeora nada y protege el caso de códigos parecidos (HE-4330 contra HE-4335) cuando el catálogo crezca. Para ver la diferencia haría falta medir el retriever solo (por ejemplo, si la página correcta sale primera) o un corpus más grande.
