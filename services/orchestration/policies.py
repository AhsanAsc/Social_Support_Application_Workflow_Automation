"""
Prompt snippets + guardrails used by 'recommend' node.
Keep it tiny; the LLM call itself is already in services.llm.ollama_client.
"""

REVIEWER_SYS = (
    "You are a senior case reviewer. Summarize eligibility clearly for another reviewer. "
    "State missing documents, key numeric facts, and a short recommendation. "
    "Be concise (<=120 words). No fluff."
)


def render_user_prompt(app_snapshot: dict, rules_result: dict, evidence: str) -> str:
    passed = [r for r in rules_result.get("rules", []) if r.get("passed")]
    failed = [r for r in rules_result.get("rules", []) if not r.get("passed")]
    return f"""
Applicant:
- name: {app_snapshot.get('applicant_full_name')}
- household_size: {app_snapshot.get('household_size')}
- monthly_income: {app_snapshot.get('monthly_income')}

Rules (passed):
{chr(10).join(f"- {r['id']}: {r['reason']}" for r in passed)}

Rules (failed):
{chr(10).join(f"- {r['id']}: {r['reason']}" for r in failed)}

Evidence:
{evidence}

Output 2–3 bullets and a one-line recommendation.
""".strip()
