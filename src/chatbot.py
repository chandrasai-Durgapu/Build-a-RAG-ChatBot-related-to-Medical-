# chatbot.py
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

def create_chat_model(api_key: str):
    """Initialize ChatGroq model."""
    chatModel = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=api_key
    )
    return chatModel


def build_rag_chain(chatModel, retriever):
    """Build retrieval augmented generation chain."""
    system_prompt = (
        "You are an Medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def answer_questions(rag_chain, questions):
    """Invoke the chain on a list of questions and print answers."""
    for question in questions:
        response = rag_chain.invoke({"input": question})
        print(f"Q: {question}\nA: {response['answer']}\n")
