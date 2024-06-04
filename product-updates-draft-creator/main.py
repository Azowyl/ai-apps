from dotenv import load_dotenv
from github_data import *
from document import *
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

import os

load_dotenv()

def get_context(llm):
    github_data = GithubData(os.getenv('REPO_URL'), os.getenv('GITHUB_ACCESS_TOKEN'))
    document_handler = DocumentHandler()

    base_query = """
        Given the following commit messages and changes corresponding to a Pull request, generate a short description of what the pull request is about.
        Consider the changes are in the following format: filename: "changed file", patch: "changes made".

        commit messages: {commits}, changes: {changes}
    """
    pr_number = int(os.getenv('PR_NUMBER'))
    template = PromptTemplate(template=base_query, input_variables=["commits", "changes"])
    query_for_documents = llm.invoke(template.invoke({"commits": github_data.get_pr_commit_messages(pr_number), "changes": github_data.get_pr_diff(pr_number)})).content

    return document_handler.retrieve_relevant_documents(query_for_documents)

def get_product_update_draft(llm):
    query = """
        Create a title and a detailed description of the implemented feature in a way that a non-technical person can understand. The description should cover the following points in a single paragraph without subtitles:

        Why the feature was needed: Explain the reason or motivation behind developing the feature.
        What problem it solved: Describe the issue or challenge that the feature addresses.
        How we define it as successful: Outline the criteria or metrics that indicate the feature's success.
        How we define it as a failure: Specify the conditions or outcomes that would signify the feature's failure.
        Additionally, provide two versions of this description: one in Spanish and one in Portuguese.
    """
    template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that writes excellent feature summaries using pitches and tech designs as context to respond to queries"
            )
        ),
        HumanMessagePromptTemplate.from_template("Query: {query} Context: {context}"),
    ])
    prompt_with_context = template.invoke({"query": query, "context": get_context(llm)})

    return llm.invoke(prompt_with_context).content

llm = ChatOpenAI(temperature=0.7)
print(get_product_update_draft(llm))