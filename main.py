import os
import pickle
from operator import itemgetter
from typing import Iterator

from dotenv import load_dotenv

from flask import Flask, render_template, request, Response
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase

from langchain_core.prompts import (ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable


from setup_dynamic_table import select_table

load_dotenv()

app = Flask(__name__)

def initialisation_db():
    return SQLDatabase.from_uri(os.getenv("BD_URL"))




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prompt", methods=["POST"])
def prompt():
    """
    Handles the POST request to the /prompt route and processes a conversation.

    :return: A server-sent event stream response containing the conversation chain.
    """
    messages = request.json["messages"]
    conversation = build_conversation_list(messages)
    return Response(get_chain(conversation), mimetype="text/event-stream")


def build_conversation_list(messages: list) -> list:
    """
    :param messages: List of messages to be alternated between human and AI.
    :return: List of `HumanMessage` and `AIMessage` objects, alternating based on the order of input messages.
    """
    return [
        HumanMessage(content=message) if i % 2 == 0 else AIMessage(content=message)
        for i, message in enumerate(messages)
    ]


def get_chain(conversation: list) -> Iterator[str]:
    """
    :param conversation: List of messages in the conversation.
    :return: Iterator that yields responses based on the conversation context.
    """
    question = ""
    for message in conversation:
        if isinstance(message, HumanMessage):
            question = message.content


   # on recupere la base de donne
    db = initialisation_db()
    # on initialise le LLM
    llm_groq = ChatGroq(model="llama-3.1-70b-versatile")
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    with open("./file/vector_db.pkl", "rb") as f:
        vectorstore = pickle.load(f)

    example_selectors = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
        input_keys=["input"]
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selectors,
        input_variables=["input", "top_k"],
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are a MySQL expert of the schema below. Given an input question, create a syntactically correct MySQL 
                query  that answer to the input question. You can order the results to return the most informative data in 
                the database.Never query for all columns from a table. You must query only the columns that are needed to
                answer the question.Pay attention to use only the column names you can see in the tables below. Be careful 
                to not query for columns that do not exist, use only the columns of the table, don't try to create on. 
                Also, pay attention to which column is in which table.Pay 
                attention to use CURDATE() function to get the current date, if the question 
                involves "today". You'll write the the SQL query and notthing else.
                Here is the relevant table info:{table_info}
                Below are a number of examples of questions and their corresponding SQL queries.
            """),
            few_shot_prompt,
            ("system", "pay attention to see the conversation history before answer, here the conversation history:"),
            MessagesPlaceholder("conversation"),
            ("human", "{input}")
        ]
    )
    generate_query = create_sql_query_chain(llm_groq, db, final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    answer_prompt = PromptTemplate.from_template(
        """
         Vous êtes un assistant pour les clients du supermarché Ruvunda à Goma, qui aide les clients avec les achats
         . Vous aidez les clients à trouver des produits ou des informations
         dans la base de données du magasin. Soyez poli, chalereux et clair dans vos réponses, en donnant seulement les 
         informations essentielles mais en proposant au client qu'il puisse demander plus de détails.
         Si le résultat SQL est nul, répondez avec "Je ne sais pas." sauf si le client t'as salue
         Tu ne vas saluer le client qu'au debut de la conversation mais aussi tes reponses doivent avec coherant avec
         la conversation:
         conversation:{conversation}
         Question du client : {question}
         Requête SQL : {query}
         Résultat SQL : {result}
         Réponse : 
         """
    )

    rephrase_answer = answer_prompt | llm_gemini| StrOutputParser()
    chain = (
            RunnablePassthrough.assign(table_names_to_use=select_table) |
            RunnablePassthrough.assign(query=generate_query).assign(
                result=itemgetter("query") | execute_query
            )
            | rephrase_answer
    )
    return chain.stream(
        {
            "question": question,
            "conversation": conversation,
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
