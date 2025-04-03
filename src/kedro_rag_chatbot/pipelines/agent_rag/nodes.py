import logging
from typing import Any, Callable

import questionary
from deeplake.core.vectorstore import VectorStore
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def create_tools(
    vector_store: VectorStore, embedding_function: Callable
) -> list[Callable]:
    @tool
    def get_context_from_vector_store(user_question: str) -> str:
        """Returns the context found in vector store based on user question."""
        return vector_store.search(
            embedding_data=user_question, embedding_function=embedding_function, k=1
        )["text"][0]

    return [get_context_from_vector_store]


def init_llm(
    gpt_3_5_turbo: ChatOpenAI, tools: list[Callable]
) -> tuple[ChatOpenAI, Runnable]:
    llm_with_tools = gpt_3_5_turbo.bind_tools(tools)
    return gpt_3_5_turbo, llm_with_tools


def create_chat_prompt(system_prompt: str) -> ChatPromptTemplate:
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return chat_prompt


def create_agent(
    llm_with_tools: Runnable, chat_prompt: ChatPromptTemplate
) -> RunnableSerializable:
    agent: RunnableSerializable = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | chat_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent


def create_agent_executor(
    agent: RunnableSerializable, tools: list[Callable]
) -> AgentExecutor:
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, return_intermediate_steps=True
    )


def invoke_agent(agent_executor: AgentExecutor, input_query: str) -> dict[str, Any]:
    return agent_executor.invoke({"input": input_query})


def invoke_llm(llm: ChatOpenAI, input_query: str) -> str:
    return llm.invoke(input_query).text()


def user_interaction_loop(agent_executor: AgentExecutor, llm: ChatOpenAI) -> str:
    """Interactive node to get input from the user loop."""
    res = []
    while True:
        user_query = questionary.text(
            "Enter question about Kedro or type `exit` to stop:"
        ).ask()
        if user_query.lower() == "exit":
            break
        llm_response = invoke_llm(llm, user_query)
        agent_response = invoke_agent(agent_executor, user_query)

        input_res = f"### User Input: {user_query}\n"
        llm_res = f"### LLM Output:\n{llm_response}\n"
        agent_res = f"### Agent Output:\n{agent_response['output']}\n"
        agent_intermediate_steps = f"### Agent Intermediate Steps:\n```json\n{agent_response['intermediate_steps']}\n```\n"
        retrieved_context = (
            f"### Retrieved Context:\n{agent_response['intermediate_steps'][0][1]}\n"
        )

        res.append(
            "\n".join(
                [
                    input_res,
                    llm_res,
                    agent_res,
                    retrieved_context,
                    agent_intermediate_steps,
                ]
            )
        )

        logger.info(input_res)
        logger.info(llm_res)
        logger.info(agent_res)

    return "\n\n".join(res)
