from __future__ import annotations

from typing import Iterable, Mapping, Tuple


def build_tool_descriptions(tools: Mapping[str, object]) -> str:
    """Builds a description of available tools for the agent."""
    if not tools:
        return ""
    lines = [
        f"- {name}: {getattr(tool, 'description', '')}" for name, tool in tools.items()
    ]
    return "\n\nAvailable tools:\n" + "\n".join(lines)


def build_documents_context(documents: Iterable[Mapping[str, str]]) -> str:
    formatted = []
    for i, doc in enumerate(documents):
        if "abstract" in doc or "citation" in doc:
            pmcid = str(doc.get("pmcid", ""))
            citation = str(doc.get("citation", ""))
            abstract = str(doc.get("abstract", ""))
            formatted.append(
                f"Article {i+1} (PMC ID: {pmcid}):\n{citation}\n\nAbstract: {abstract}"
            )
            continue

        title = str(doc.get("title", ""))
        body = str(doc.get("body", ""))
        pmcid = str(doc.get("pmcid", ""))
        if pmcid:
            formatted.append(f"Section {i+1} (PMC ID: {pmcid}) - {title}:\n{body}")
        else:
            formatted.append(f"Section {i+1} - {title}:\n{body}")
    return "\n\n".join(formatted)


def build_synthesis_prompts(user_input: str, context: str) -> Tuple[str, str]:
    system = """You are a biomedical research communicator. Your job is to explain the key findings from the provided research articles in simple, accessible language.\n\nRequirements:\n- Use plain language; define unavoidable jargon\n- Be accurate and cautious about causality\n- Cite sources in a list formatted like: (Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678)\n- Do not invent sources; only cite from the provided articles\n- Be accurate but approachable\n- Structure your response with clear sections\n\nFormat your response like:\n**What the research found:**\n\n**Why it matters:**\n\n**The science behind it:**\n\nAlways cite sources with a link to the article like: (Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678)\n"""
    human = f"""Based on these research articles, please explain what we know about: {user_input}\n\nResearch Articles:\n{context}\n\nRemember: Explain in simple, everyday language while staying accurate. Cite the articles with pubmed central links, not DOI links. E.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678.

Return ONLY valid JSON with this exact schema:
{{
    "what_the_research_found": "string",
    "why_it_matters": "string",
    "the_science_behind_it": "string",
    "sources": ["string", "..."]
}}

Do not include markdown, prose outside JSON, or code fences."""
    return system, human


def build_qa_prompts(user_input: str, context: str) -> Tuple[str, str]:
    system = """You are a biomedical research communicator. Answer the user's question using ONLY the provided research articles.\n\nGuidelines:\n- Explain in simple, everyday language\n- Always cite which article(s) you're referencing (e.g., \"Article 1, PMC123...\")\n- If the articles do not answer the question, say so clearly\n"""
    human = f"""Research articles:\n{context}\n\nUser question:\n{user_input}\n
Return ONLY valid JSON with this exact schema:
{{
    "answer": "string",
    "citations": ["string", "..."]
}}

Do not include markdown, prose outside JSON, or code fences.
"""
    return system, human
