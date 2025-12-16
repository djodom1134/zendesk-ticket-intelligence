#!/usr/bin/env python3
"""Test MCP client to fetch a single ticket with full details and test summarization"""
import asyncio
import json
import sys
import httpx
sys.path.insert(0, ".")
from services.ingest.client import ZendeskMCPClient

MCP_URL = "http://192.168.87.79:10005/sse"
OLLAMA_URL = "http://10.0.0.242:11434"  # Remote GPU

def parse_ticket_result(result: dict) -> dict:
    """Parse MCP tool result to get ticket data"""
    if "content" in result:
        for item in result["content"]:
            if item.get("type") == "text":
                return json.loads(item["text"])
    return result

def format_ticket_display(ticket: dict) -> str:
    """Format ticket data for display"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"TICKET #{ticket.get('id')} - FULL CONTEXT")
    lines.append("=" * 80)

    # Metadata
    lines.append("\n### METADATA ###")
    lines.append(f"Subject: {ticket.get('subject')}")
    lines.append(f"Status: {ticket.get('status')}")
    lines.append(f"Priority: {ticket.get('priority') or 'Not set'}")
    lines.append(f"Type: {ticket.get('type')}")
    lines.append(f"Created: {ticket.get('created_at')}")
    lines.append(f"Updated: {ticket.get('updated_at')}")
    lines.append(f"Via: {ticket.get('via')}")

    # People
    lines.append("\n### PEOPLE ###")
    lines.append(f"Requester: {ticket.get('requester_name')} ({ticket.get('requester_email')})")
    lines.append(f"Assignee (Agent): {ticket.get('assignee_name')} ({ticket.get('assignee_email')})")

    # Organization
    org = ticket.get("organization", {})
    if org:
        lines.append("\n### ORGANIZATION ###")
        lines.append(f"Name: {org.get('name')}")
        org_fields = org.get("organization_fields", {})
        lines.append(f"Region: {org_fields.get('region')}")
        lines.append(f"Type: {org_fields.get('type')}")
        lines.append(f"Support Status: {org_fields.get('support_status')}")

    # Tags
    lines.append(f"\n### TAGS ###")
    lines.append(f"{', '.join(ticket.get('tags', []))}")

    # Custom Fields (non-null only)
    lines.append("\n### CUSTOM FIELDS (with values) ###")
    for cf in ticket.get("custom_fields", []):
        if cf.get("value") and cf.get("value") != "false":
            lines.append(f"  {cf.get('name')}: {cf.get('value')}")

    # Comments
    lines.append("\n### CONVERSATION THREAD ###")
    for i, comment in enumerate(ticket.get("comments", []), 1):
        role = comment.get("author_role", "unknown")
        role_label = "[AGENT]" if role == "agent" else "[CUSTOMER]"
        lines.append(f"\n--- Comment {i} {role_label} ---")
        lines.append(f"From: {comment.get('author_name')} ({comment.get('author_email')})")
        lines.append(f"Date: {comment.get('created_at')}")
        lines.append(f"Body:\n{comment.get('plain_body', comment.get('body'))}")

    lines.append("\n" + "=" * 80)

    # Stats
    total_words = sum(len(c.get("plain_body", "").split()) for c in ticket.get("comments", []))
    lines.append(f"\nTOTAL COMMENTS: {len(ticket.get('comments', []))}")
    lines.append(f"TOTAL WORD COUNT: {total_words}")

    return "\n".join(lines)

def build_summary_prompt(ticket: dict) -> str:
    """Build anti-hallucination prompt for summarization"""
    # Build conversation thread
    thread = []
    for comment in ticket.get("comments", []):
        role = "[AGENT]" if comment.get("author_role") == "agent" else "[CUSTOMER]"
        name = comment.get("author_name", "Unknown")
        body = comment.get("plain_body", comment.get("body", ""))
        thread.append(f"{role} {name}: {body}")

    conversation = "\n\n".join(thread)

    # Custom fields with values
    custom_fields = []
    for cf in ticket.get("custom_fields", []):
        if cf.get("value") and str(cf.get("value")).lower() not in ["null", "false", "none"]:
            custom_fields.append(f"  - {cf.get('name')}: {cf.get('value')}")
    custom_fields_str = "\n".join(custom_fields) if custom_fields else "  None provided"

    org = ticket.get("organization", {})
    org_fields = org.get("organization_fields", {}) if org else {}

    total_words = sum(len(c.get("plain_body", "").split()) for c in ticket.get("comments", []))

    prompt = f"""You are a support ticket analyst. Extract a structured summary from the ticket below.

CRITICAL RULES - FOLLOW EXACTLY:
1. ONLY include information EXPLICITLY stated in the ticket content
2. If a section has no information in the ticket, write "[Not provided]"
3. Do NOT infer, assume, or fabricate ANY details
4. Do NOT add example commands, API paths, KB articles, or version numbers unless explicitly in the ticket
5. Quote exact error messages verbatim when present
6. Mark agent responses vs customer messages clearly

---
TICKET METADATA:
- Ticket ID: {ticket.get('id')}
- Subject: {ticket.get('subject')}
- Status: {ticket.get('status')}
- Priority: {ticket.get('priority') or 'Not set'}
- Created: {ticket.get('created_at')}
- Requester: {ticket.get('requester_name')} ({ticket.get('requester_email')})
- Agent: {ticket.get('assignee_name')} ({ticket.get('assignee_email')})
- Organization: {org.get('name', 'Unknown')}
- Region: {org_fields.get('region', 'Unknown')}
- Tags: {', '.join(ticket.get('tags', []))}

CUSTOM FIELDS:
{custom_fields_str}

CONVERSATION ({len(ticket.get('comments', []))} comments, {total_words} words):
{conversation}
---

OUTPUT FORMAT (use exactly this structure):

**TICKET SUMMARY**
- ID: {ticket.get('id')}
- Agent: {ticket.get('assignee_name')}
- Total Comments: {len(ticket.get('comments', []))}
- Word Count: {total_words}
- Density: [sparse (<100 words) / moderate (100-300) / detailed (>300)]

**ISSUE CLASSIFICATION**
- Type: [license/bug/feature/how-to/integration/etc - based on content]
- Category: [from tags or content]
- Severity: [stated or "not specified"]

**PROBLEM DESCRIPTION**
[1-2 sentence summary of what the customer reported - use their exact words where possible]

**TECHNICAL DETAILS**
- Error Messages: [exact quotes or "none provided"]
- Affected Component: [specific component mentioned or "not specified"]
- Software Version: [from custom fields/tags or "not specified"]
- License Keys: [if mentioned, list them, or "not provided"]
- Hardware ID: [if mentioned or "not provided"]

**RESOLUTION STATUS**
- Agent Response: [summarize what the agent said/advised]
- Customer Follow-up: [did customer respond? what did they say?]
- Resolution: [resolved/pending/closed without response/etc]

**KEYWORDS** (for clustering):
[5-7 relevant terms extracted from the ticket]

STRUCTURED SUMMARY:"""

    return prompt

async def summarize_ticket(ticket: dict) -> str:
    """Call LLM to summarize ticket"""
    prompt = build_summary_prompt(ticket)

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gpt-oss:120b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1500}
            }
        )
        result = response.json()
        return result.get("response", "ERROR: No response")

async def get_ticket(ticket_id: int):
    client = ZendeskMCPClient(MCP_URL)

    try:
        print(f"Connecting to MCP server at {MCP_URL}...")
        if not await client.connect():
            print("Failed to connect")
            return

        print("Connected!")

        # Fetch ticket
        print(f"\nFetching ticket {ticket_id} with full context...")
        result = await client.call_tool("get_ticket_full_context", {"ticket_id": ticket_id})

        if result.get("isError"):
            print(f"Error: {result}")
            return

        ticket = parse_ticket_result(result)

        # Display full ticket
        print("\n" + "=" * 80)
        print("PART 1: ORIGINAL TICKET (FULL FORM)")
        print("=" * 80)
        print(format_ticket_display(ticket))

        # Generate summary
        print("\n" + "=" * 80)
        print("PART 2: AI SUMMARY")
        print("=" * 80)
        print("\nGenerating summary (anti-hallucination mode)...")
        summary = await summarize_ticket(ticket)
        print("\n" + summary)

        # Hallucination check
        print("\n" + "=" * 80)
        print("PART 3: HALLUCINATION CHECK")
        print("=" * 80)
        print("\nCheck the summary above against the original ticket.")
        print("Look for:")
        print("  - Invented version numbers")
        print("  - Made-up API paths or commands")
        print("  - Fabricated KB article references")
        print("  - Details not present in original ticket")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    ticket_id = int(sys.argv[1]) if len(sys.argv) > 1 else 45981
    asyncio.run(get_ticket(ticket_id))

