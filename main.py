from flask import Flask, render_template, request, Response
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, SystemMessagePromptTemplate, \
    PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

db = SQLDatabase.from_uri("mysql+pymysql://root:fazili@localhost:3306/db_stock_supermarche")

example_few_shot = [
        {
            "Question": "Je cherche un appareil électroménager",
            "SQL query": "SELECT * FROM produits WHERE categorie_id = 4;"
        },
        {
            "Question": "Avez vous de televiseur",
            "SQL query": 'SELECT * FROM produits WHERE description LIKE "%Tel%";'
        },
        {
            "Question": "Je cherche du fromage",
            "SQL query": 'SELECT * FROM produits WHERE description = "fromage" OR nom like "%fromage%";'
         },
        {
            "Question": "Je cherche un four",
            "SQL query": 'SELECT * FROM produits WHERE description = "four";'
         },
        {
            "Question": "j'ai faim, j'aimerais mangé un truc vite fait, pourriez-vous vous me proposez un truc avec un "
                        "boison qui va avec biensur",
            "SQL query": 'SELECT nom, prix, description, nutriments FROM db_stock_supermarche.produits WHERE '
                         'categorie_id = 1 OR categorie_id=2;'
        },
        {
            "Question": "j'ai faim, j'aimerais mangé un bon plat complet, avez vous qlq chose à me proposer comme du "
                        "legume et de la viande?",
            "SQL query": 'SELECT nom, prix, description, nutriments FROM db_stock_supermarche.produits WHERE '
                         'description = "viande" OR description = "Legume";'},
        {
            "Question": "Je vais prendre 2 produits",
            "SQL query": 'SELECT nom,prix, quantite FROM db_stock_supermarche.produits WHERE description LIKE '
                         '"%produits%" or nom LIKE "%produits%";'
        },
        {
            "Question": "j'ai pas beaucoup d'argent, propose moi votre television la moi chère",
            "SQL query": 'SELECT * FROM db_stock_supermarche.produits where description LIKE "%tel%";'
        },
        {
            "Question": "Avez vous un produit",
            "SQL query": 'SELECT * from produits WHERE description LIKE "%produit%" or nom "%produit%"'
        },
        {
            "Question": "je vais acheter du pain complet, quel fromage pourrait allez avec ce pain?",
            "SQL query": 'SELECT * FROM db_stock_supermarche.produits where description = "pain" or description ='
                         ' "fromage";'
        },
        {
            "Question": "Pouvez me donnez les noms et les prix de ce differents produits",
            "SQL query": 'SELECT nom, prix FROM produits WHERE description LIKE %produits% or nom LIKE "%produits%'
        },
        {
            "Question": "C'est possible d'acheter 6 de ce produits?",
            "SQL query": 'SELECT quantite FROM produits WHERE nom LIKE "%produits%" or nom LIKE "%produits%"'
        }

    ]
embedding = HuggingFaceEmbeddings()
to_vectorize = [" ".join(examples.values()) for examples in example_few_shot]
vector_store = FAISS.from_texts(to_vectorize, embedding, metadatas=example_few_shot)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prompt", methods=["POST"])
def prompt():
    messages = request.json["messages"]
    conversation = build_conversation_dict(messages)
    return Response(sql_to_response_chain(db, conversation), mimetype="text/event-stream")


def build_conversation_dict(messages: list) -> list:
    return [
        HumanMessage(content=message) if i % 2 == 0 else AIMessage(content=message)
        for i, message in enumerate(messages)
    ]


def text_to_sql_chain(conversation: list, base_de_donne: SQLDatabase) -> RunnableSerializable:
    global vector_store
    llm_groq = ChatGroq(model="llama-3.1-70b-versatile", temperature=1)
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

    example_selectors = SemanticSimilarityExampleSelector(vectorstore=vector_store, k=2, input_keys=["Question"])

    template_prefix = """
            You are a MySQL expert of the schema below. Given an input question, create a syntactically correct MySQL 
            query  that answer to the input question. You can order the results to return the most informative data in 
            the database.Never query for all columns from a table. You must query only the columns that are needed to
            answer the question.Pay attention to use only the column names you can see in the tables below. Be careful 
            to not query for columns that do not exist, use only the columns of the table, don't try to create on. 
            Also, pay attention to which column is in which table.Pay 
            attention to use CURDATE() function to get the current date, if the question 
            involves "today".
            Schema:{schema}
            You will answer the questions by referring to the conversation history.
            Conversation History: {chat_history}
            Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not 
            even backticks.
            for exemple:
        """
    template_suffix = """
                    if the users question is a greeting, just do like the following exemple:
                    Question: Bonjour
                    SQL query: SELECT * FROM categories;
                    your turn:
                    Question: {Question}
                    SQL query: 
                    """
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQL query"],
        template="\nQuestion: {Question} \n SQL query: {SQL query}\n"
    )
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selectors,
        example_prompt=example_prompt,
        prefix=template_prefix,
        suffix=template_suffix,
        input_variables=["Question", "schema", "chat_history"]
    )
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{Question}")
        ]
    )

    def info_bd(_):
        base_de_donne.get_table_info()

    return (
            RunnablePassthrough.assign(schema=info_bd)
            | full_prompt
            | llm_groq
            | StrOutputParser()
    )


def sql_to_response_chain(base_de_donne: SQLDatabase, conversation: list) -> str:
    sql_chain = text_to_sql_chain(conversation=conversation, base_de_donne=base_de_donne)
    llm_groq = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    question = ""
    for questions in conversation:
        if isinstance(questions, HumanMessage):
            question = questions.content
    template = """
            Tu es un assistant qui charger d'assister les clients qui veulent acheter des articles dans dans le 
            supermarche Ruvunda qui se trouve a Goma, ce supermarche vend de tout, produits alimentaires, 
            electromenagers, comestique et tout ces qui se trouve dans la base de donnees, tu interagis avec des 
            clients  qui cherche certains articles ou information concernant ces articles, assures toi de toujours etre
            poli, ne donnes pas trop de details sur les produits qu'un client peut te demander,
            a moins que ce client exige plus de details sur le produit en question. C'est basant sur la table schema 
            ci dessous, question du client, query et le reponse sql. Reponds au client de maniere la plus courtoise et 
            claire possible de telle maniere que peu import l'age ou la comprehension du client, qu'il puisse te 
            comprendre, tu n'ecriras pas de code SQL ou n'importe quel autre code, ton role sera juste d'excecuter la 
            requetes pour trouver ce que le client cherche. Si le client te salut, accueil les clients en lui parlant 
            de nos categories de produits

            <SCHEMA>{schema}</SCHEMA>
            Conversation History: {chat_history}
            SQL Query: <SQL>{query}</SQL>
            Client Question: {Question}
            SQL Response: {response}
    """
    full_prompt = ChatPromptTemplate.from_template(template)
    chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda _: db.run(_["query"])
            )
            | full_prompt
            | llm_gemini
            | StrOutputParser()
    )
    response_stream = chain.stream(
        {
            "Question": question,
            "chat_history": conversation,
        }
    )
    for text in response_stream:
        yield text


if __name__ == '__main__':
    app.run(debug=True)




