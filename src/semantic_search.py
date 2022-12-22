
import cohere
import pinecone
from settings import COHERE_KEY, COHERE_MODEL, PINECONE_KEY, PINECONE_ENV, PINCEONE_INDEX
import streamlit as st

# Get COHERE and PINECONE keys
cohere_key = st.secrets[COHERE_KEY]
pinecone_key = st.secrets[PINECONE_KEY]

pinecone.init(pinecone_key, environment=PINECONE_ENV)
index = pinecone.Index(PINCEONE_INDEX)
co = cohere.Client(cohere_key)

query = "I am feeling lonely"

def get_most_similars(query, model=COHERE_MODEL):
  # create the query embedding
  query_embedding = co.embed(
      texts=[query],
      model=model,
      truncate='LEFT'
  ).embeddings

  # Get the top 5 most similar results
  res = index.query(query_embedding, top_k=5, include_metadata=True)

  # for match in res['matches']:
  #     print(f"Score: {match['score']:.2f}, Proverbs {match['metadata']['meta']} says {match['metadata']['verse']}")
  results = []
  for result in res['matches']:
    verse = result['metadata']
    results.append(verse['meta'] + ':' + verse['verse'])
  
  return results



