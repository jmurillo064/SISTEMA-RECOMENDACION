from flask import Flask, render_template, request, jsonify
import time, json
from models.modelo import procesarAbstract

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', mostrar=False)

@app.route('/procesar', methods=['POST'])
def procesar():
    abstract = request.form['abstract']
    # Obtiene el resultado de procesarAbstract
    datos = procesarAbstract(abstract)
    return render_template('index.html', abstract=abstract, datos=datos, mostrar=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
