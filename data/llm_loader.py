from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from data.model.retriever_model import ResponseModel
import os
import ast

load_dotenv()
def createModel(model_name: str, api_key: str):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024   
    )

    return llm

def generateLLMResponse(context: str, query: str):
    model_name="llama-3.1-8b-instant"
    api_key=os.getenv("groq_api_key")
    print("Groq api key", api_key)
    llm = createModel(model_name, api_key)

    promptTemplate = PromptTemplate(
        input_variables=["context","query"],
        template="""You are a helpful AI assistant. Use the following context to answer the question correctly and perfectly.
                    Context: {context}
                    Question: {query}
                    Provide clean and clear, informative answer based on the context above. If context doesn't contain enough information to the question, say so"""
    )

    formatted_query = promptTemplate.format(context=context, query=query)

    try:
        messages = [HumanMessage(content=formatted_query)]
        response = llm.invoke(messages)
        print(response.content)
        questions = responseBasedQuestion(response.content)
        return ResponseModel(content=response.content, questions=questions)
    except Exception as e:
        raise

def responseBasedQuestion(data: str):
    model_name="llama-3.1-8b-instant"
    api_key=os.getenv("groq_api_key")

    llm = createModel(model_name, api_key)

    promptTemplate = PromptTemplate(
    input_variables=["query"],
    template="""You are an assistant. Based on the user's query and the provided response, generate exactly 5 relevant follow-up questions that the user might ask next. 
            - Format the output as a **Python-style list** like this:
            ["Question1", "Question2", "Question3", "Question4", "Question5"]

            - Do not include anything else.  
            - Ensure the questions are clear, concise, and relevant to the original query.

            Response to analyze: {query}""")


    formatted_query = promptTemplate.format(query=data)
    try:
        messages = [HumanMessage(content=formatted_query)]
        response = llm.invoke(messages)
        print("response of question", response.content)
        return ast.literal_eval(response.content)
    except Exception as e:
        raise