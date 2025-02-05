import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai

# Importaciones de LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # Importa la clase para modelos de chat

# Cargar variables de entorno y configurar la API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)



class LLM:
    @staticmethod
    def get_llm():
        """
        Retorna una instancia del LLM de LangChain usando ChatOpenAI.
        Se utiliza para modelos de chat como gpt-3.5-turbo.
        """
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

class VectorStore:
    @staticmethod
    def obtencion_vectores():
        """
        Crea un vector store a partir de los documentos PDF ubicados en la carpeta "data".
        Se usan OpenAIEmbeddings (modelo "text-embedding-ada-002") y FAISS para indexar los documentos.
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        documents = []
        data_path = os.path.join(os.getcwd(), "../data")
        if not os.path.exists(data_path):
            print("No se encontró la carpeta 'data'.")
        else:
            for filename in os.listdir(data_path):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(data_path, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error al cargar {filename}: {e}")
        if not documents:
            print("No se encontraron documentos en la carpeta 'data'.")
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore



def contruccion_cadena():
    """
    Construye una cadena RetrievalQA utilizando:
      - El vector store obtenido de los documentos PDF.
      - El LLM configurado.
      - Un retriever basado en similitud (k=3 documentos).
    """
    vectorstore = VectorStore.obtencion_vectores()
    llm = LLM.get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Concatena los documentos recuperados para generar la respuesta
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False,
    )
    return qa_chain

# Construir la cadena al iniciar el servicio
qa_chain = contruccion_cadena()


@app.route('/assistant/rag', methods=['POST'])
def assistant_rag():
    """
    Endpoint: /assistant/rag
    Función: Responde preguntas de educación financiera utilizando RAG.
    Se espera recibir un JSON con la clave "message" que contenga la consulta del usuario.
    
    La cadena RetrievalQA se encarga de:
      1. Recuperar los documentos (o fragmentos) relevantes a la pregunta.
      2. Generar una respuesta basada en dicha información.
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No se proporcionó 'message' en la solicitud"}), 400

    pregunta = data["message"]
    try:
        # Usar invoke() en lugar de run() para evitar la advertencia de deprecación
        result = qa_chain.invoke(pregunta)
        return jsonify({"respuesta": result}), 200
    except Exception as e:
        print(f"Error en /assistant/rag: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Ejecutar el microservicio
# ------------------------------

if __name__ == '__main__':
    app.run(port=5001, debug=True)
