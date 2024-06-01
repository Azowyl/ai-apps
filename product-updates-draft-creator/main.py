from dotenv import load_dotenv
from github_data import *
import os

load_dotenv()

github = GithubData(os.getenv('REPO_URL'), os.getenv('GITHUB_ACCESS_TOKEN'))