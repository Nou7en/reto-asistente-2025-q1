
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
      - "pdf" para Análisis de PDF
      - "shopping" para Asesor de Compras
      - "rag" para Chat RAG acerca de educacion financiera
    """
    prompt = f"""
Dado el siguiente mensaje de un usuario: "{message}"
Clasifícalo en una de las siguientes intenciones:
1. Chat RAG (preguntas y respuestas).
2. Análisis de PDF (resumen de gastos en documentos).
3. Asesor de Compras (recomendación de productos y precios).
Responde solo con una de las siguientes palabras:
"rag" para Chat RAG,
"pdf" para Análisis de PDF,
"shopping" para Asesor de Compras.

#Ejemplos

Ejemplo 1
Entrada : "Como puedo dejar de gastar dinero?"
Salida esperada: "rag"

Ejemplo 2 
Entrada : "Cual fue el mes que mas gastos realice?"
Salida esperada: "pdf"

Ejemplo 3 
Entrada : "Que celular recomiendas comprar en este 2024?"
Salida esperada: "shopping"

"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "Eres un experto en clasificar la intencion del usuario en estos 3 temas Educacion Financiera de preguntas y respuestas, Informacion de pdf y recomendador de compras"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,  
        )
        
        answer = response.choices[0].message["content"].strip().lower()
        
        intent = answer.split()[0]
        return intent
    except Exception as e:
        print(f"Error al clasificar la intención: {e}")
        # Si falla la clasificación, se retorna "rag" por defecto
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
