import os
import asyncio
from typing import List, Sequence
from langgraph.graph import END, MessageGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Global variables
generate = None
reflect = None

# Definitions
async def generation_node(state: Sequence[BaseMessage]):
    return await generate.ainvoke({"messages": state})

async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = await reflect.ainvoke({"messages": translated})
    # We treat the output of this as human feedback for the generator
    return HumanMessage(content=res.content)

# Function to determine if the conversation should continue
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        # End after 3 iterations
        return END
    return "reflect"

async def main():
    # Import Definitions
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名小说评论家。请写一篇出色的五段的小说评论."
                " 根据用户的要求，竭尽所能，写最好的文章."
                " 当用户提出修改意见，请提供更加用户的意见修改后的版本。",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Reading API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

    global generate 
    generate = prompt | llm

    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名修改评论文章的老师，请对给出的评论文章提出批评和建议."
                "请提供各方面详细的修改意见， 比如文章的长度，深度，文章的风格等等.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    global reflect
    reflect = reflection_prompt | llm

    # Graph Construction
    builder = MessageGraph()
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")

    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")
    graph = builder.compile()

    async for event in graph.astream(
        [
            HumanMessage(
                content="请对小说‘射雕英雄传’写篇评论"
            )
        ],
    ):
        if 'generate' in event:
            print("---------- 以下是生成的评论文章------------ ")
            print(event['generate'].content)
        elif 'reflect' in event:
            print("---------- 以下是修改意见------------ ")
            print(event['reflect'].content)
        else:
            raise KeyError

if __name__ == "__main__":
    asyncio.run(main())

