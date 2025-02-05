import PyPDF2
from flask import Flask, request, jsonify
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def lectura_contenido():
    """
    Funcion para unir y leer los documentos PDF en la carpeta 'data'
    """
    context = ""
    data_path = os.path.join(os.getcwd(), "data")

    if not os.path.exists(data_path):
        return "No se encontro la carpeta"
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            context += text + "\n"
            except Exception as e:
                print(f"Error al leer {filename}: {e}")
    
    return context

@app.route('/assistant/rag', methods=['POST'])
def assistant_rag():
    """
    Endpoint: /assistant/rag
    Función: Realizar preguntas y respuestas basadas en recuperación aumentada (RAG).
    Se espera recibir un JSON con la clave "message" que contiene la consulta del usuario.
    
    La función:
      1. Recupera el contenido de los documentos PDF ubicados en la carpeta 'data'.
      2. Construye un prompt que incluya dicho contexto y la pregunta.
      3. Llama al LLM para generar una respuesta basada en esa información.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No se proporcionó 'message' en la solicitud"}), 400

    user_message = data.get("message")
    context = lectura_contenido()
    
    prompt = (
        "En base a la informacion acerca de la educacion financiera de la informacion de los documentos razona y responde la pregunta del usuario de forma clara y precisa\n\n"
        "Contexto:\n"
        f"{context}\n"
        "Pregunta: " + user_message + "\n\n"
        "Proporciona una respuesta detallada basada en la información anterior:"
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ajusta el nombre del modelo según el que uses (por ejemplo, "gpt-3.5-turbo")
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente financiero que responde preguntas utilizando información extraída de documentos en formato PDF."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0,
        )
        
        respuesta = response.choices[0].message["content"].strip()
        return jsonify({"respuesta": respuesta}), 200
    
    except Exception as e:
        print(f"Error en /assistant/rag: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # El microservicio assistant se ejecuta en el puerto 5001
    app.run(port=5001, debug=True)
