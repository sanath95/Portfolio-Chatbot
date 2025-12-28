from os import getenv
from aiohttp import ClientSession
from json import dumps

async def fetch_project_repos(url: str) -> str:
    """
    Fetch Sanath's GitHub repositories, sorted by creation date (desc), for project linking.
    
    Args:
        url: API endpoint url for fetching user repos.
        
    Returns:
    str: A JSON-formatted string containing a list of repositories.
            Each repository entry includes:
            - name (str): Repository name
            - description (str | None): Repository description
            - html_url (str): Public GitHub URL of the repository

    Raises:
        RuntimeError: If the GITHUB_TOKEN environment variable is not set.
        aiohttp.ClientResponseError: If the GitHub API request fails.
    """
    token = getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable not set")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params = {
        "sort": "created",
        "direction": "desc",
    }

    async with ClientSession(headers=headers) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            repos = await response.json()
    
    filtered_response = [
        {
            "name": repo["name"],
            "description": repo["description"],
            "html_url": repo["html_url"],
        }
        for repo in repos
    ]
    return dumps(filtered_response)