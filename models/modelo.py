import spacy, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

def procesarAbstract(abstract):
    # Cargar modelo de lenguaje
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Taxonomía IEEE (ejemplo reducido)
    taxonomy = ["Computer Vision", "Machine Learning", "Neural Networks", "Image Processing", "Agriculture", "Optimization"]

    # Extraer palabras clave usando TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([abstract])
    feature_names = vectorizer.get_feature_names_out()

    # Obtener términos clave con TF-IDF alto
    tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
    keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:10]  # Top 10 términos clave
    keywords = [kw[0] for kw in keywords]

    # Embeddings para comparación semántica
    abstract_embeddings = model.encode(keywords)
    taxonomy_embeddings = model.encode(taxonomy)

    # Calcular similitud coseno y mostrar resultados detallados
    print("\nResultados de clasificación con similitud coseno:\n")

    resultados = {}  # Inicializa resultados como un diccionario vacío

    for i, keyword in enumerate(keywords):
        # Calcular similitudes
        similarities = util.pytorch_cos_sim(abstract_embeddings[i], taxonomy_embeddings).numpy().flatten()
        
        # Convertir similitudes a float (serializable por JSON)
        similarities = similarities.astype(float)
        
        best_match_idx = np.argmax(similarities)
        best_category = taxonomy[best_match_idx]
        
        # Crear el resultado para este keyword
        keyword_result = {
            "keyword": keyword,
            "similarities": [
                {"category": taxonomy[j], "similarity": round(similarities[j], 4)} for j in range(len(taxonomy))
            ],
            "best_category": best_category
        }

        # Agregar el resultado al diccionario usando la keyword como clave
        resultados[keyword] = keyword_result

        # Mostrar la similitud de este término con todas las categorías
        print(f"Término clave: {keyword}")
        for j, category in enumerate(taxonomy):
            print(f"   {category}: {similarities[j]:.4f}")
        print(f"   -> Mejor categoría asignada: {best_category}\n")

    # Convertir la lista de resultados a JSON
    #json_resultado = json.dumps(resultados, indent=4)

    # Mostrar el JSON generado
    #print(json_resultado)
    #return json_resultado
    print(resultados)
    return resultados
