import strictyaml
import difflib
import requests
import base64

#Config loader to load UI configs
class UIConfig:
    def __init__(self, config_path: str="./config.yaml"):
        self.config_path = config_path
        self.config = None
        self.load_yaml()

    def load_yaml(self):
        # Load YAML file
        with open(self.config_path, "r", encoding="utf-8") as file:
            yaml_content = file.read()

        # Parse YAML
        parsed_yaml = strictyaml.load(yaml_content)

        # Convert to standard dictionary
        self.config = parsed_yaml.data

    def get_page_title(self):
        return self.config["PAGE_TITLE"]

    def get_llm_options(self):
        return self.config["LLM_OPTIONS"]

    def get_usecase_options(self):
        return self.config["USECASE_OPTIONS"]

    def get_groq_model_options(self):
        return self.config["GROQ_MODEL_OPTIONS"]

    def get_ollama_model_options(self):
        return self.config["OLLAMA_MODEL_OPTIONS"]

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as file:
            file.write(self.config)



def highlight_diff(original: str, revised: str) -> str:
    """Generate an HTML-based diff between original and revised text."""
    diff = difflib.ndiff(original.splitlines(), revised.splitlines())
    diff_html = []
    
    for line in diff:
        if line.startswith("+ "):
            diff_html.append(f'<span style="background-color: #d4f4dd;">{line[2:]}</span>')  # Green for added
        elif line.startswith("- "):
            diff_html.append(f'<span style="background-color: #f4d4d4;">{line[2:]}</span>')  # Red for removed
        else:
            diff_html.append(line[2:])  # No changes
    
    return "<br>".join(diff_html)

def create_jira_ticket(story):
    """Create a Jira ticket for an approved user story."""
    config = UIConfig()
    url = f"{config.get('JIRA_URL')}/rest/api/2/issue"
    auth = (config.get('JIRA_USER'), config.get('JIRA_API_TOKEN'))
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "fields": {
            "project": {"key": config.get('JIRA_PROJECT_KEY')},
            "summary": story.feature_request,
            "description": story.generated_story,
            "issuetype": {"name": "Story"},
        }
    }

    response = requests.post(url, json=payload, auth=auth, headers=headers)
    
    if response.status_code == 201:
        story.jira_ticket = response.json()["key"]  # Store Jira ticket ID
        return story.jira_ticket
    else:
        print("Failed to create Jira ticket:", response.text)
        return None
    
def upload_design_doc(jira_ticket, file_path: str):
    """Upload or update a design document to a Jira ticket."""
    config = UIConfig()
    url = f"{config.get('JIRA_URL')}/rest/api/2/issue/{jira_ticket}/attachments"
    auth = (config.get('JIRA_USER'), config.get('JIRA_API_TOKEN'))
    headers = {
        "X-Atlassian-Token": "no-check"  # Required for file uploads
    }

    with open(file_path, "rb") as file:
        files = {"file": (file_path.split("/")[-1], file, "application/octet-stream")}
        response = requests.post(url, files=files, auth=auth, headers=headers)

    if response.status_code == 200:
        print(f"✅ Design document uploaded successfully to Jira {jira_ticket}")
        return True
    else:
        print(f"❌ Failed to upload design document: {response.text}")
        return False
    
def deploy_to_github(file_path: str, github_file_path: str, commit_message: str, jira_ticket: str):
    """Deploys code to GitHub by creating a pull request."""
    config = UIConfig()
    headers = {
        "Authorization": f"token {config.get('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json",
    }
    GITHUB_MAIN_BRANCH = "main"
    GITHUB_PR_BRANCH_PREFIX = "develop"

    # Step 1: Create a new branch
    branch_name = f"{GITHUB_PR_BRANCH_PREFIX}-{jira_ticket}"
    create_branch_url = f"https://api.github.com/repos/{config.get('GITHUB_REPO_NAME')}/git/refs"

    # Get the latest commit SHA from main
    response = requests.get(f"https://api.github.com/repos/{config.get('GITHUB_REPO_NAME')}/git/refs/heads/{GITHUB_MAIN_BRANCH}", headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed to fetch main branch info: {response.text}")
        return False

    latest_commit_sha = response.json()["object"]["sha"]

    # Create new branch
    data = {"ref": f"refs/heads/{branch_name}", "sha": latest_commit_sha}
    response = requests.post(create_branch_url, json=data, headers=headers)

    if response.status_code not in [200, 201]:
        print(f"❌ Failed to create branch {branch_name}: {response.text}")
        return False

    print(f"✅ Created new branch: {branch_name}")

    # Step 2: Commit the file to the new branch
    url = f"https://api.github.com/repos/{config.get('GITHUB_REPO_NAME')}/contents/{github_file_path}"

    with open(file_path, "rb") as file:
        content = base64.b64encode(file.read()).decode("utf-8")

    data = {
        "message": commit_message,
        "content": content,
        "branch": branch_name,
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code not in [200, 201]:
        print(f"❌ Failed to commit code: {response.text}")
        return False

    print(f"✅ Code successfully pushed to {branch_name}")

    # Step 3: Create a Pull Request
    pr_url = f"https://api.github.com/repos/{config.get('GITHUB_REPO_NAME')}/pulls"
    pr_data = {
        "title": f"Feature {jira_ticket}: {commit_message}",
        "head": branch_name,
        "base": GITHUB_MAIN_BRANCH,
        "body": f"### Changes for Jira {jira_ticket}\n\nThis PR implements the feature requested in {jira_ticket}.",
    }

    response = requests.post(pr_url, json=pr_data, headers=headers)

    if response.status_code in [200, 201]:
        pr_link = response.json()["html_url"]
        print(f"✅ Pull Request Created: {pr_link}")
        return pr_link
    else:
        print(f"❌ Failed to create Pull Request: {response.text}")
        return False
    
def upload_pr_to_jira(jira_ticket: str, pr_link: str):
    """Attach the GitHub PR link to the Jira issue."""
    config = UIConfig()
    url = f"{config.get('JIRA_URL')}/rest/api/2/issue/{jira_ticket}/comment"
    auth = (config.get('JIRA_USER'), config.get('JIRA_API_TOKEN'))
    headers = {"Content-Type": "application/json"}

    data = {"body": f"🔗 GitHub PR created: {pr_link}"}

    response = requests.post(url, json=data, auth=auth, headers=headers)

    if response.status_code == 201:
        print(f"✅ PR link updated in Jira {jira_ticket}")
    else:
        print(f"❌ Failed to update Jira: {response.text}")