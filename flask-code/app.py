from flask import Flask, render_template, request
import cohere
import pinecone

# Get COHERE and PINECONE keys
COHERE_KEY = "8mcow1CGJUuyu9k8fPNJS3mwKKUvcvza2aDWlbnH"
PINECONE_KEY = "6c18c917-804e-4852-881a-bc547f8d9ab5"
PINECONE_ENV = 'us-west1-gcp'
PINCEONE_INDEX = 'cohere-pinecone-bible'

pinecone.init(PINECONE_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINCEONE_INDEX)
co = cohere.Client(COHERE_KEY)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/response", methods=['POST', 'GET'])
def response():
    if request.method == 'POST':
        query = str(request.form['guidance'])
        name = request.form['name']
        number_verses = int(request.form['number_verses'])
        # query = "I am feeling lonely"

        # create the query embedding
        query_embedding = co.embed(
            texts=[query],
            model='large',
            truncate='LEFT'
        ).embeddings

        # Get the top most similar results
        res = index.query(query_embedding, top_k=number_verses,
                          include_metadata=True)

        # for match in res['matches']:
        #     print(f"Score: {match['score']:.2f}, Proverbs {match['metadata']['meta']} says {match['metadata']['verse']}")
        results = []
        for result in res['matches']:
            verse = result['metadata']
            results.append(verse['meta'] + ', ' + verse['verse'])

        return render_template('response.html', results=results, name=name)
