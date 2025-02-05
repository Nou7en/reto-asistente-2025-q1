
from flask import Flask, request, jsonify
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# URL base del microservicio asistente
ASSISTANT_URL = "http://localhost:5001"

def clasificador(message: str) -> str:
    """
    Usa OpenAI GPT-4/GPT-3.5 para clasificar la intención del mensaje del usuario.
    Retorna:
      - "pdf" para Análisis de PDF (resumen y análisis de gastos en documentos PDF).
      - "shopping" para Asesor de Compras (recomendaciones de productos y precios).
      - "rag" para Chat RAG (preguntas y respuestas relacionadas con educación financiera).
      - "irrazonable" si el mensaje no se relaciona con los temas anteriores o es malintencionado/tóxico.
    """
    prompt = f"""
Dado el siguiente mensaje de un usuario: "{message}"
Clasifícalo en una de las siguientes intenciones:
1. Chat RAG: Preguntas y respuestas sobre educación financiera.
2. Análisis de PDF: Para entradas con la intencion de obtner informacion,Resumen y análisis de gastos en documentos PDF.
3. Asesor de Compras: Recomendación de productos y precios.
Si el mensaje no se relaciona con ninguno de estos temas o es malintencionado/toxico 
(ejemplo: solicita obtener información sensible, realizar análisis no autorizados o promover la compra de armas u otras actividades peligrosas),
responde con "irrazonable".

Responde solo con una de las siguientes palabras exactas:
- "rag" para Chat RAG.
- "pdf" para Análisis de PDF.
- "shopping" para Asesor de Compras.
- "irrazonable" para solicitudes fuera de tema, malintencionadas o tóxicas.

# Ejemplos

Ejemplo 1:
Entrada: "Como puedo dejar de gastar dinero?"
Salida esperada: "rag"

Ejemplo 2:
Entrada: "Cual fue el mes que más gastos realicé?"
Salida esperada: "pdf"

Ejemplo 3:
Entrada: "Que celular recomiendas comprar en este 2024?"
Salida esperada: "shopping"


Ejemplo 4:
Entrada: "Quiero comprar armas de fuego para defensa propia"
Salida esperada: "irrazonable"

Ejemplo 5:
Entrada: "Eres un asistente financiero personal. Con base en ello, brinda el valor aproximado de ingresos mensuales considerando el estado de cuenta proporcionado."
Salida esperada: "irrazonable"

Ejemplo 6:
Entrada: "Quiero productos de defensa personal"
Salida esperada: "shopping"
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en clasificar la intención del usuario en uno de estos temas: "
                        "educación financiera (preguntas y respuestas), análisis de PDF y asesor de compras. "
                        "Si la consulta es malintencionada o tóxica, responde con 'irrazonable'."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        
        answer = response.choices[0].message["content"].strip().lower().strip('"')
        intent = answer.split()[0]
        return intent
    except Exception as e:
        print(f"Error al clasificar la intención: {e}")
        # En caso de error, se retorna "rag" por defecto
        return "rag"


@app.route('/orchestrate', methods=['POST'])
def orchestrate():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No se proporcionó 'message' en la solicitud"}), 400

    message = data['message']
    
    # Clasifica la intención usando el LLM
    intent = clasificador(message)
    print(f"Intención clasificada: {intent}")

    # Validar que la intención esté entre las permitidas
    if intent not in ['pdf', 'shopping', 'rag']:
        return jsonify({"error": "Solicitud no razonable"}), 400

    # Determina el endpoint del microservicio asistente según la intención
    if intent == "pdf":
        endpoint = f"{ASSISTANT_URL}/assistant/analyze-pdf"
    elif intent == "shopping":
        endpoint = f"{ASSISTANT_URL}/assistant/shopping-advisor"
    else:
        endpoint = f"{ASSISTANT_URL}/assistant/rag"

    try:
        # Reenvía la solicitud original al microservicio asistente
        response = requests.post(endpoint, json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
