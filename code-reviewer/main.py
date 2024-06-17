from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_changed_files, create_comment, is_test, find_line
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

tools = [
    create_comment,
    is_test,
    find_line
]
llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a senior TypeScript engineer, expert in testing.
                When asked to review a list of files, for each file, you should do these steps in order 
                1. Determine if it is a test file. If it isn't skip it
                2. Find all errors in the given changes, ignore existing errors and focus on the newones.
                3. Find the line number for each of the errors found on the previous step
                4. Create a comment for each error in the current pull
            """,
        ),
        ("user", "Review this list of files: {files}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"files": get_changed_files()})
