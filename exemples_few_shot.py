import pickle

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


example_few_shot = [
    {
        "input": "Bonjour",
        "query": 'SELECT * categories '
    },
    {
        "input": 'Quel sont les marques de produits que vous avez?',
        "query": 'SELECT nom FROM marques',
    },
    {
        "input": 'Avez vous un telephone?',
        "query": 'SELECT * FROM produits WHERE description LIKE "%smartphone %";'
    },
    {
        "input": "Avez un produit de la marque nike?",
        "query": 'SELECT produits.nom AS product_name, produits.description AS product_description, produits.prix,'
                 'marques.nom AS brand_name FROM  supermarche.produits JOIN supermarche.marques ON '
                 'produits.marque_id = marques.marque_id WHERE marques.nom = "nike";'
    },
    {
        "input": "lequel de chocolat est le moi chère ",
        "query": 'SELECT nom, prix FROM supermarche.produits WHERE description LIKE "%chocolat%" ORDER BY prix ASC LIMIT 1'
    },
    {
        "input": "dites moi son prix",
        "query": 'SELECT prix FROM supermarche.produits WHERE nom LIKE "%nom_du_produit%" or description LIKE "%nom_produit%"'
    },
    {
        "input": "Je vais en acheter 3",
        "query": 'SELECT prix * 3, nom FROM supermarche.produits WHERE nom LIKE "%nom_du_produit%" or description LIKE "%nom_produit%"'
    },
    {
        "input": "J'ai faim",
        "query": 'SELECT nom, description, prix FROM supermarche.produits WHERE categorie_id = (SELECT categorie_id FROM supermarche.categories WHERE nom = "Alimentaire")'
    },
]

embedding = HuggingFaceEmbeddings()
to_vectorize = [" ".join(examples.values()) for examples in example_few_shot]
vectorstore = FAISS.from_texts(to_vectorize, embedding, metadatas=example_few_shot)

with open("./file/vector_db.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
