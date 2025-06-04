import json
import re
import os 
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from transformers import TextIteratorStreamer
from langchain.prompts import ChatPromptTemplate
import json


api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

class RatedAnswer(BaseModel):
    clarity_rating: int = Field(ge=0, le=100, description="Clarity rating out of 100.")
    clarity_description: str = Field(description="Justification for the clarity rating.")
    
    depth_rating: int = Field(ge=0, le=100, description="Depth rating out of 100.")
    depth_description: str = Field(description="Justification for the depth rating.")
    
    relevance_rating: int = Field(ge=0, le=100, description="Relevance rating out of 100.")
    relevance_description: str = Field(description="Justification for the relevance rating.")

class Evaluation(BaseModel):
    evaluations: list[RatedAnswer] = Field(
        description="List of ratings and justifications for each of the three LLM-generated answers."
    )

structured_llm = llm.with_structured_output(Evaluation)

# Create prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """You are a judge evaluating 3 answers to the same question.
            Rate each answer from 0 to 100 for:
            - Clarity
            - Depth
            - Relevance

            Then explain each score in 1-2 sentences.

            Here are the answers:
            Answer 1: "..."  
            Answer 2: "..."  
            Answer 3: "..."

            Respond strictly in the Evaluation format."""
        ),
        ("human", """
            The query is {query}, 
            Answer 1: is {answer1}
            Answer 2: is {answer2}
            Answer 3: is {answer3}"""
        )
    ]
) 

clarity_scores = [0.0, 0.0, 0.0]
depth_scores = [0.0, 0.0, 0.0]
relevance_scores = [0.0, 0.0, 0.0]

with open("results_dylan.json") as f1, open("results_langgraph.json") as f2, open("results_cot.json") as f3:
    responses1 = json.load(f1)["responses"]
    responses2 = json.load(f2)["responses"]
    responses3 = json.load(f3)["responses"]

for i, question in enumerate(responses1):
    ans1 = responses1[question]
    ans2 = responses2[question]
    ans3 = responses3[question]

    messages = prompt_template.format_messages(
        query=question,
        answer1=ans1,
        answer2=ans2,
        answer3=ans3
    )

    result = structured_llm.invoke(messages)

    print(result)

    for j, evaluation in enumerate(result.evaluations):
        clarity_scores[j] += evaluation.clarity_rating / 100
        depth_scores[j] += evaluation.depth_rating / 100
        relevance_scores[j] += evaluation.relevance_rating / 100

    print(clarity_scores)
    print(depth_scores)
    print(relevance_scores)






    