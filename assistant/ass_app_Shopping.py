import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai
import logging

# Configurar logging (solo errores)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def extraer_parametros(query: str) -> dict:
    prompt = f"""
Analiza la siguiente pregunta y extrae los parámetros relevantes para realizar una búsqueda de productos.
Devuelve un JSON con las siguientes claves:
- message (obligatorio): el término principal.
- color (opcional): el color, si se menciona.
- model (opcional): la marca o modelo, si se menciona.
- min_price (opcional): el precio mínimo, si se menciona.
- max_price (opcional): el precio máximo, si se menciona.
- num_results (opcional): el número de resultados deseados (entre 1 y 5).

Pregunta: "{query}"

Ejemplo de respuesta JSON:
{{
  "message": "celular",
  "color": "negro",
  "model": "Samsung",
  "min_price": "200",
  "max_price": "400",
  "num_results": "4"
}}

Si algún parámetro no se menciona, deja el valor como una cadena vacía.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en extraer parámetros de búsqueda de una pregunta."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0
        )
        respuesta_texto = response.choices[0].message["content"].strip()
        if not respuesta_texto:
            logger.error("Respuesta del LLM para extraer parámetros vacía.")
            return {"message": query, "color": "", "model": "", "min_price": "", "max_price": "", "num_results": ""}
        if respuesta_texto.startswith("```"):
            lines = respuesta_texto.splitlines()
            if len(lines) >= 2:
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                respuesta_texto = "\n".join(lines).strip()
        parametros = json.loads(respuesta_texto)
        for key in ["message", "color", "model", "min_price", "max_price", "num_results"]:
            if key not in parametros:
                parametros[key] = ""
        return parametros
    except Exception as e:
        logger.error("Error al extraer parámetros: %s", e)
        return {"message": query, "color": "", "model": "", "min_price": "", "max_price": "", "num_results": ""}

def build_search_query(params: dict) -> str:
    query_parts = []
    if params.get("message"):
        query_parts.append(params["message"])
    if params.get("color"):
        query_parts.append(params["color"])
    if params.get("model"):
        query_parts.append(params["model"])
    if params.get("min_price") and params.get("max_price"):
        query_parts.append(f"${params['min_price']}-{params['max_price']}")
    elif params.get("min_price"):
        query_parts.append(f"${params['min_price']}+")
    elif params.get("max_price"):
        query_parts.append(f"up to ${params['max_price']}")
    query_parts.append("Ecuador")
    return " ".join(query_parts)

def buscar_productos(query: str, num_results: int):
    search_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google_shopping",
        "api_key": SERPAPI_KEY,
        "google_domain": "google.com.ec",
        "hl": "es",
        "gl": "ec",
        "num": num_results
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        productos = []
        if "shopping_results" in data:
            for item in data["shopping_results"]:
                price_str = item.get("price", "")
                try:
                    price = float(price_str.replace("$", "").strip()) if price_str else None
                except Exception:
                    price = None
                producto = {
                    "Nombre Artículo": item.get("title", "N/A"),
                    "Comercio": item.get("source", "N/A"),
                    "Precio en USD": price,
                    "Web del anuncio": item.get("link", "N/A")
                }
                productos.append(producto)
        return productos
    except Exception as e:
        logger.error("Error en búsqueda de productos: %s", e)
        return []

def recomendar_productos(productos: list, query_original: str, final_count: int) -> list:
    prompt = (
        "Eres un asesor de compras experto en el mercado ecuatoriano. "
        "Analiza la siguiente lista de productos y la consulta del usuario, y selecciona un top de productos recomendados. "
        "Para cada producto, si el precio se encuentra, devuelve el valor numérico en el campo 'Precio' y marca 'TipoPrecio' como 'original'; "
        "si no se proporciona precio, estima un valor numérico razonable, devuelve ese valor en 'Precio' y marca 'TipoPrecio' como 'estimado'. "
        "Además, agrega un campo 'Ranking' indicando 'Top 1', 'Top 2', etc., y un breve 'Comentario' para cada producto.\n\n"
        "Consulta del usuario: " + query_original + "\n\n"
        "Lista de productos (en formato JSON):\n" + json.dumps(productos, indent=2) + "\n\n"
        f"Devuelve la respuesta en formato JSON, que sea una lista de {final_count} productos, donde cada producto tenga las claves: "
        "\"Nombre Artículo\", \"Comercio\", \"Precio\", \"TipoPrecio\", \"Ranking\" y \"Comentario\"."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asesor de compras experto que analiza productos y precios."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.0
        )
        respuesta_texto = response.choices[0].message["content"].strip()
        if respuesta_texto.startswith("```"):
            lines = respuesta_texto.splitlines()
            if len(lines) >= 2:
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                respuesta_texto = "\n".join(lines).strip()
        return json.loads(respuesta_texto)
    except Exception as e:
        logger.error("Error al recomendar productos: %s", e)
        fallback = []
        for idx, prod in enumerate(productos[:final_count], start=1):
            fallback.append({
                "Nombre Artículo": prod.get("Nombre Artículo", "N/A"),
                "Comercio": prod.get("Comercio", "N/A"),
                "Precio": prod.get("Precio en USD") if prod.get("Precio en USD") is not None else 0,
                "TipoPrecio": "original" if prod.get("Precio en USD") is not None else "estimado",
                "Ranking": f"Top {idx}",
                "Comentario": "Sin comentario"
            })
        return fallback

@app.route('/assistant/shopping-advisor', methods=['POST'])
def shopping_advisor():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No se proporcionó 'message' en la solicitud"}), 400

    parametros = extraer_parametros(data["message"])
    query = build_search_query(parametros)
    
    try:
        num_results = int(parametros.get("num_results", ""))
        num_results = max(1, min(num_results, 5))
    except Exception:
        num_results = 3
    
    productos = buscar_productos(query, num_results)
    if not productos:
        fallback_params = parametros.copy()
        fallback_params["min_price"] = ""
        fallback_params["max_price"] = ""
        fallback_query = build_search_query(fallback_params)
        productos = buscar_productos(fallback_query, num_results)
    
    if not productos:
        return jsonify({"error": "No se encontraron productos para el término de búsqueda."}), 404
    
    final_count = num_results if num_results > 0 else 3
    recomendaciones = recomendar_productos(productos, data["message"], final_count)
    
    return jsonify({"resultados": recomendaciones}), 200

if __name__ == '__main__':
    app.run(port=5003, debug=True)
