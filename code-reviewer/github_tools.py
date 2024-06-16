from langchain.tools import tool
from dotenv import load_dotenv
from github import Auth
from github import Github
import base64

import os

load_dotenv()

@tool
def create_comment(comment: str, filename: str, line: int):
    """Creates a comment in the current pull request in the specified file (filename) and on the specified line."""

    print('comment: ')
    print(comment)

@tool
def is_test(filename):
    """Returns true if the current file is a test file, false otherwhise"""
    return "spec" in filename or "test" in filename

def get_changed_files():
    """Returns a list of dicts, where each element in the list has the following format:
    {
        filename: name of the file
        file: file contents,
        changes: changes made
    }
    """

    pr = get_pull_request()
    return [
        {
            "filename": file.filename,
            "file": get_file(file.filename),
            "changes": file.patch
        }
        for file in pr.get_files()
    ]

def get_file(filename):
    repo = get_repo()
    pr = get_pull_request()
    return base64.b64decode(repo.get_contents(filename, pr.base.ref).raw_data.get('content'))

def get_pull_request():
    return get_repo().get_pull(int(os.getenv('PR_NUMBER')))

def get_repo():
    auth = Auth.Token(os.getenv('GITHUB_ACCESS_TOKEN'))
    github = Github(auth=auth)
    return github.get_repo(os.getenv('REPO_URL'))
