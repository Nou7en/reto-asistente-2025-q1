import os
import re
from flask import Flask, request, jsonify
import PyPDF2
import openai
from dotenv import load_dotenv

# Cargar variables de entorno (por ejemplo, OPENAI_API_KEY)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/assistant/analyze-pdf', methods=['POST'])
def analyze_pdf():
    """
    Endpoint: /assistant/analyze-pdf
    Función: Analiza un estado de cuenta PDF subido por el usuario para:
      - Extraer transacciones y calcular el Top 5 de mayores gastos.
      - Agrupar y contabilizar gastos recurrentes (por establecimiento) y obtener el Top 3.
      - Usar un modelo de lenguaje para responder una pregunta basada en el análisis realizado.
    
    Se espera:
      - Un archivo PDF enviado bajo la clave "file" (multipart/form-data).
      - Opcionalmente, un parámetro "question" (en form-data o JSON) que contenga la consulta a responder.
         Si no se envía, se usará una pregunta por defecto.
    """
    # Verificar que se haya enviado el archivo PDF
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó el archivo PDF en 'file'"}), 400

    pdf_file = request.files['file']

    # Obtener la pregunta: primero se intenta en form-data y luego en JSON
    question = request.form.get("question")
    if not question:
        data = request.get_json(silent=True)
        if data and "question" in data:
            question = data["question"]
    if not question:
        question = "Resume el análisis de gastos y proporciona recomendaciones para ahorrar dinero."

    # Extraer el texto del PDF
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        return jsonify({"error": f"Error al leer el PDF: {e}"}), 500

    # Procesar el texto para extraer transacciones
    lines = text.splitlines()
    transactions = []
    
    # Esta expresión regular asume que cada línea contiene el nombre del establecimiento
    # seguido por el monto (con punto o coma decimal) al final de la línea.
    pattern = re.compile(r'^(?P<establishment>.+?)\s+(?P<amount>\d+[.,]\d{2})\s*$')
    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            establishment = match.group("establishment").strip()
            amount_str = match.group("amount").replace(',', '.')
            try:
                amount = float(amount_str)
                transactions.append({"establishment": establishment, "amount": amount})
            except ValueError:
                continue

    if not transactions:
        return jsonify({"error": "No se encontraron transacciones en el PDF."}), 404

    # Calcular el Top 5 de mayores gastos
    top5 = sorted(transactions, key=lambda x: x["amount"], reverse=True)[:5]

    # Agrupar gastos recurrentes por establecimiento
    establishment_totals = {}
    for txn in transactions:
        est = txn["establishment"]
        amount = txn["amount"]
        if est in establishment_totals:
            establishment_totals[est]["total"] += amount
            establishment_totals[est]["count"] += 1
        else:
            establishment_totals[est] = {"total": amount, "count": 1}

    recurrent_expenses = [
        {"establishment": est, "total": data["total"], "count": data["count"]}
        for est, data in establishment_totals.items()
    ]
    top3_recurrentes = sorted(recurrent_expenses, key=lambda x: x["total"], reverse=True)[:3]

    # Construir un resumen en texto del análisis
    summary_text = "Análisis del estado de cuenta:\n\n"
    summary_text += "Top 5 de mayores gastos:\n"
    for i, txn in enumerate(top5, start=1):
        summary_text += f"{i}. {txn['establishment']}: ${txn['amount']:.2f}\n"
    summary_text += "\nTop 3 de gastos recurrentes:\n"
    for i, rec in enumerate(top3_recurrentes, start=1):
        summary_text += f"{i}. {rec['establishment']}: Total ${rec['total']:.2f} en {rec['count']} transacciones\n"

    # Construir el prompt para el modelo
    prompt = (
        summary_text + "\n" +
        "Usa únicamente la información anterior para responder la siguiente pregunta:\n" +
        f"{question}\n"
    )

    try:
        # Usar el endpoint de chat completions para generar la respuesta
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Se puede ajustar el modelo según necesidad
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de gastos y educación financiera."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0,
        )
        answer = response.choices[0].message["content"].strip()
        # Devolver tanto el análisis como la respuesta generada
        return jsonify({
            "analysis": summary_text,
            "respuesta": answer
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error al generar respuesta: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
