COHERE_KEY = ""
PINECONE_KEY = ""

import pinecone

pinecone.init(PINECONE_KEY, environment='us-west1-gcp')

index_name = 'cohere-pinecone-bible'

# if the index does not exist, we create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=shape[1],
        metric='cosine'
    )

# connect to index
index = pinecone.Index(index_name)


import cohere

co = cohere.Client(COHERE_KEY)


query = "I am feeling down lately"

# create the query embedding
xq = co.embed(
    texts=[query],
    model='large',
    truncate='LEFT'
).embeddings

#print(np.array(xq).shape)

# query, returning the top 10 most similar results
res = index.query(xq, top_k=10, include_metadata=True)
#res

for match in res['matches']:
    print(f"Score: {match['score']:.2f}, Proverbs {match['metadata']['meta']} says {match['metadata']['verse']}")