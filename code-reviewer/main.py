from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from github_tools import get_changed_files, create_comment, is_test
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

tools = [
    create_comment,
    is_test
]
llm = ChatOpenAI(temperature=0.7)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a senior TypeScript engineer, expert in testing.
                When asked to review a list of files, for each file, you should first determine if it is a test file. If it is, you should find all errors in the given changes, ignore existing errors and focus on the newones.

                Tests should follow these guidelines:
                1. Each test starts with 'it' or 'describe' functions. Make a detailed review for each file
                2. Tests should not start with 'should'. For example, it('should render...') must be fixed to it('renders...').
                3. Tests should have semantic meaning with the 'it' function provided by Jest. For example, it('make...') must be fixed to it('makes...').
                5. If any test states a given success scenario, there should be another test with the error scenario. For example, if there is a context: describe('when request succeeds'), then there should be another context: describe('when request fails'). you should indicate the missing scenarios if there are any.

                if you encounter any violation to the previous guidelines, you should consider it an error
            """,
        ),
        ("user", "Review this list of files: {files}"),
        ("user", "Given the previous review, create a comment for each proposed fix in the correct filename and line, in the current pull request"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"files": get_changed_files()})
