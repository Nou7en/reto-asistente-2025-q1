from flask import Flask, request, jsonify
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Definir las URLs base de cada microservicio (en este ejemplo, se usan puertos distintos)
ASSISTANT_RAG_URL = "http://localhost:5002"         # Servicio Rag
ASSISTANT_PDF_URL = "http://localhost:5001"         # Servicio para análisis de PDF
ASSISTANT_SHOPPING_URL = "http://localhost:5003"    # Servicio para asesor de compras

def clasificador(message: str) -> str:
    """
    Clasifica la intención del mensaje del usuario.
    Retorna:
      - "rag" para Chat RAG.
      - "pdf" para Análisis de PDF.
      - "shopping" para Asesor de Compras.
      - "irrazonable" para solicitudes fuera de tema o malintencionadas.
    """
    prompt = f"""
Dado el siguiente mensaje de un usuario: "{message}"
Clasifícalo en una de las siguientes intenciones:
1. Chat RAG: Preguntas y respuestas sobre educación financiera.
2. Análisis de PDF: Para obtener información, resumen y análisis de gastos en documentos PDF.
3. Asesor de Compras: Recomendación de productos y precios.
Si el mensaje no se relaciona con estos temas o es malintencionado/toxico,
responde con "irrazonable".

Responde solo con una de las siguientes palabras exactas:
- "rag"
- "pdf"
- "shopping"
- "irrazonable"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Eres un experto en clasificar la intención del usuario en uno de estos temas: "
                    "educación financiera (preguntas y respuestas), análisis de PDF y asesor de compras. "
                    "Si la consulta es malintencionada o tóxica, responde con 'irrazonable'."
                )},
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
        return "rag"  # Valor por defecto

@app.route('/orchestrate', methods=['POST'])
def orchestrate():
    """
    Orquestador:
      - Si se envía un archivo PDF (campo "file" presente y no vacío), se redirige al análisis de PDF.
      - Si no se envía un archivo válido, se procesa la solicitud como JSON para clasificar la intención
        y redirigirla al endpoint correspondiente (rag o shopping).
    """
    # Verificar si se envió el campo "file" y que tenga un nombre no vacío
    file_present = ('file' in request.files and request.files['file'] and 
                    request.files['file'].filename.strip() != "")
    
    if file_present:
        # Se asume que se quiere analizar un PDF.
        file = request.files['file']
        question = request.form.get("question")  # Opcional
        files = {"file": (file.filename, file, file.mimetype)}
        data = {}
        if question:
            data["question"] = question
        endpoint = f"{ASSISTANT_PDF_URL}/assistant/analyze-pdf"
        try:
            response = requests.post(endpoint, files=files, data=data)
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Procesar la solicitud como JSON
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No se proporcionó 'message' en la solicitud"}), 400
        message = data['message']
        intent = clasificador(message)
        print(f"Intención clasificada: {intent}")
        
        # Si la intención es "pdf" pero no se envió un archivo, se rechaza la solicitud.
        if intent == "pdf":
            return jsonify({"error": "Para análisis de PDF se requiere enviar el archivo en 'file'"}), 400
        elif intent == "shopping":
            endpoint = f"{ASSISTANT_SHOPPING_URL}/assistant/shopping-advisor"
        else:  # "rag"
            endpoint = f"{ASSISTANT_RAG_URL}/assistant/rag"
        
        try:
            response = requests.post(endpoint, json=data)
            return jsonify(response.json()), response.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
