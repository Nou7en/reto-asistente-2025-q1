import os
import re
import pdfplumber
from flask import Flask, request, jsonify
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
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó el archivo PDF en 'file'"}), 400

    pdf_file = request.files['file']
    
    question = request.form.get("question")
    if not question:
        data = request.get_json(silent=True)
        if data and "question" in data:
            question = data["question"]
    if not question:
        question = "Resume el análisis de gastos y proporciona recomendaciones para ahorrar dinero."

    # Extraer el texto del PDF con pdfplumber
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return jsonify({"error": f"Error al leer el PDF: {e}"}), 500
    
    # Procesar el texto para extraer transacciones
    lines = text.splitlines()
    transactions = []
    pattern = re.compile(r'^(?P<establishment>.+?)\s+(?P<amount>\d+[.,]\d{2})\s*$')
    
    for line in lines:
        line = line.strip()
        match = pattern.search(line)
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
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de gastos y educación financiera."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = response.choices[0].message["content"].strip()
        return jsonify({
            "analysis": summary_text,
            "respuesta": answer
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error al generar respuesta: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)