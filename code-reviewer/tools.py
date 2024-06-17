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

    pr = get_pull_request()
    last_commit = pr.get_commits()[pr.commits - 1]
    pr.create_comment(comment, last_commit, filename, line)

@tool
def is_test(filename):
    """Returns true if the current file is a test file, false otherwhise"""
    return "spec" in filename or "test" in filename

@tool
def find_line(code_extract: bytes, filename: str):
    """Finds the line number of the given code_extract as bytes in the filename
    Example parameters: code_extreact: b'console.log("debug")', filename: 'src/Desktop.tsxt'
    """
    file_content = get_file(filename)
    lines = file_content.split(b'\n')
    for i, line in enumerate(lines):
        if code_extract in line:
            return i + 1

def get_changed_files():
    pr = get_pull_request()
    return [
        {
            "filename": file.filename,
            "changes": file.patch
        }
        for file in pr.get_files()
    ]

def get_file(filename):
    repo = get_repo()
    pr = get_pull_request()
    return base64.b64decode(repo.get_contents(filename, pr.head.ref).raw_data.get('content'))

def get_pull_request():
    return get_repo().get_pull(int(os.getenv('PR_NUMBER')))

def get_repo():
    auth = Auth.Token(os.getenv('GITHUB_ACCESS_TOKEN'))
    github = Github(auth=auth)
    return github.get_repo(os.getenv('REPO_URL'))
