from flask import Flask, render_template, request, Response
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prompt", methods=["POST"])
def prompt():
    messages = request.json["messages"]
    conversation = build_conversation_dict(messages)
    return Response(event_stream(conversation), mimetype="text/event-stream")


def build_conversation_dict(messages: list) -> list:
    return [
        HumanMessage(content=message) if i % 2 == 0 else AIMessage(content=message)
        for i, message in enumerate(messages)
    ]


def event_stream(conversation: list) -> str:
    llm = ChatGroq(model="llama-3.1-70b-versatile")
    question = ""
    for questions in conversation:
        if isinstance(questions, HumanMessage):
            question = questions.content

    template = """Repond a la question en fonction de l'historique de la conversation
                question: {question}
                historique de message: {chat_history}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (prompt | llm | StrOutputParser())
    response_stream = chain.stream(
        {"question": question,
         "chat_history": conversation
         }
    )
    for text in response_stream:
        yield text


if __name__ == '__main__':
    app.run(debug=True)



