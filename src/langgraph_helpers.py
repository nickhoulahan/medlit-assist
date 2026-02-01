from __future__ import annotations

from typing import Iterable, Mapping, Tuple


def build_tool_descriptions(tools: Mapping[str, object]) -> str:
    if not tools:
        return ""
    lines = [
        f"- {name}: {getattr(tool, 'description', '')}" for name, tool in tools.items()
    ]
    return "\n\nAvailable tools:\n" + "\n".join(lines)


def build_documents_context(documents: Iterable[Mapping[str, str]]) -> str:
    formatted = []
    for i, doc in enumerate(documents):
        pmcid = str(doc.get("pmcid", ""))
        citation = str(doc.get("citation", ""))
        abstract = str(doc.get("abstract", ""))
        formatted.append(
            f"Article {i+1} (PMC ID: {pmcid}):\n{citation}\n\nAbstract: {abstract}"
        )
    return "\n\n".join(formatted)


def build_synthesis_prompts(user_input: str, context: str) -> Tuple[str, str]:
    system = """You are a biomedical research communicator. Your job is to explain the key findings from the provided research articles in simple, accessible language.\n\nRequirements:\n- Use plain language; define unavoidable jargon\n- Be accurate and cautious about causality\n- Cite sources in a list formatted like: (Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678)\n- Do not invent sources; only cite from the provided articles\n- Be accurate but approachable\n- Structure your response with clear sections\n\nFormat your response like:\n**What the research found:**\n\n**Why it matters:**\n\n**The science behind it:**\n\nAlways cite sources with a link to the article like: (Title, https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678)\n"""
    human = f"""Based on these research articles, please explain what we know about: {user_input}\n\nResearch Articles:\n{context}\n\nRemember: Explain in simple, everyday language while staying accurate. Cite the articles with pubmed central links."""
    return system, human


def build_qa_prompts(user_input: str, context: str) -> Tuple[str, str]:
    system = """You are a biomedical research communicator. Answer the user's question using ONLY the provided research articles.\n\nGuidelines:\n- Explain in simple, everyday language\n- Always cite which article(s) you're referencing (e.g., \"Article 1, PMC123...\")\n- If the articles do not answer the question, say so clearly\n"""
    human = f"""Research articles:\n{context}\n\nUser question:\n{user_input}\n"""
    return system, human
