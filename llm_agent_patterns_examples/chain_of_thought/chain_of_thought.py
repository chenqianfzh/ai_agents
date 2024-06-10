from langchain_openai import ChatOpenAI

model = "gpt-4"
llm = ChatOpenAI(temperature=0.1, model=model)

problem = """
最近，我又读了那首著名的诗歌，’面朝大海，春暖花开‘。兴之所致，决定去作者的故乡看一看。于是就来了一场说走就走的旅行。在当地，我度过了优哉游哉的好时光，听戏听歌。啊，你知道那儿最著名的地方戏是什么吗？"""

# Here we try a plain query

query = f"""Problem statement: 
{problem}
Can you solve the problem?
"""


answer = llm.invoke(query)

print ("---------------The following is the answer from a plain query----------------\n")
print(answer.content)

cot_query = f"""Problem statement: 
{problem}
First, list systematically and in detail all the problems in this problem 
that need to be solved before we can arrive at the correct answer. 
Then, solve each sub problem using the answers of previous problems 
and reach a final solution.
"""

answer = llm.invoke(cot_query)
print ("\n---------------The following is the answer from a chain-of-thought query----------------\n")
print(answer.content)