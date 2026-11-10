"""Simulated MCP workspace server. Sensitive tools only log; nothing leaves the process."""

from __future__ import annotations

import json

from fastmcp import FastMCP

from .scenarios import BASE_FILES, INJECTION, Attack, Scenario

TOOL_DESCRIPTIONS = {
    "read_inbox": "Return the user's unread emails.",
    "list_files": "List the files in the user's workspace.",
    "read_file": "Read a file from the user's workspace.",
    "fetch_webpage": "Fetch a web page and return its HTML.",
    "send_email": "Send an email on behalf of the user.",
    "delete_file": "Delete a file from the user's workspace.",
}


def build_server(scenario: Scenario, attack: Attack) -> FastMCP:
    def inject(text: str) -> str:
        return text.replace(INJECTION, attack.content)

    emails = [{**e, "body": inject(e["body"])} for e in scenario.emails]
    pages = {url: inject(html) for url, html in scenario.pages.items()}
    files = {**BASE_FILES, **{p: inject(c) for p, c in scenario.files.items()}}

    def read_inbox() -> str:
        return json.dumps(emails, indent=2)

    def list_files() -> str:
        return "\n".join(sorted(files))

    def read_file(path: str) -> str:
        if path not in files:
            raise ValueError(f"File not found: {path}")
        return files[path]

    def fetch_webpage(url: str) -> str:
        if url not in pages:
            raise ValueError(f"404 Not Found: {url}")
        return pages[url]

    def send_email(to: str, subject: str, body: str) -> str:
        return f"Email sent to {to}."

    def delete_file(path: str) -> str:
        files.pop(path, None)
        return f"Deleted {path}."

    mcp = FastMCP("workspace")
    for fn in (read_inbox, list_files, read_file, fetch_webpage, send_email, delete_file):
        description = TOOL_DESCRIPTIONS[fn.__name__]
        if fn.__name__ == scenario.carrier_tool:
            description += attack.description_suffix
        mcp.tool(fn, name=fn.__name__, description=description)
    return mcp
