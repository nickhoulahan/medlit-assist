from __future__ import annotations

import json
from pathlib import Path
import re
from typing import List, Dict

from lxml import etree as ET


class XMLToDictConverter:
    """Extract article full text from NLM/JATS XML.

    Get body from /pmc-articleset/article/body or //article/body fallback
    and convert section titles + paragraphs dict of chunks
    """
    skip_sections = {"references", "acknowledgments", "funding", "conflict of interest", "supplementary material", "author contributions", "supplementary information"}

    @staticmethod
    def _parse_xml(xml_content: str | bytes) -> ET._Element:
        try:
            if isinstance(xml_content, str):
                xml_content = xml_content.encode("utf-8")
            return ET.fromstring(xml_content)
        except ET.XMLSyntaxError as exc:
            raise ValueError(f"Invalid XML content: {exc}") from exc

    @classmethod
    def _find_body(cls, root: ET._Element) -> ET._Element | None:
        body = root.find(".//body")
        if body is not None:
            return body

        # Namespace-aware fallback for documents that use a default namespace.
        for elem in root.iter():
            if ET.QName(elem).localname == "body":
                return elem
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _normalize_section_title(title: str) -> str:
        title = title.strip().rstrip(":.;")
        title = re.sub(r"\s+", " ", title)
        return title.casefold()

    @classmethod
    def _iter_body_blocks(cls, body: ET._Element):
        normalized_skip_titles = {
            cls._normalize_section_title(title) for title in cls.skip_sections
        }

        for sec in body.iter():
            if ET.QName(sec).localname != "sec":
                continue

            title_elem = next(
                (
                    child
                    for child in sec
                    if ET.QName(child).localname == "title"
                ),
                None,
            )
            if title_elem is None:
                continue

            title_text = cls._clean_text("".join(title_elem.itertext()))
            if not title_text:
                continue
            normalized_title_text = cls._normalize_section_title(title_text)
            if any(skip_title in normalized_title_text for skip_title in normalized_skip_titles):
                continue

            paragraphs: List[str] = []
            for p in sec.iter():
                if ET.QName(p).localname != "p":
                    continue

                # Keep only paragraphs that belong to this section directly,
                # not nested subsections.
                parent = p.getparent()
                nearest_sec = None
                while parent is not None:
                    if ET.QName(parent).localname == "sec":
                        nearest_sec = parent
                        break
                    parent = parent.getparent()

                if nearest_sec is not sec:
                    continue

                para = cls._clean_text("".join(p.itertext()))
                if para:
                    paragraphs.append(para)

            body_text = "\n\n".join(paragraphs)
            if body_text:
                yield {"title": title_text, "body": body_text}

    @classmethod
    def convert(
        cls,
        xml_content: str | bytes,
    ) -> List[Dict[str, str]]:
        """Return article sections from XML ``body`` as title/body dictionaries."""
        root = cls._parse_xml(xml_content)
        body = cls._find_body(root)
        if body is None:
            raise ValueError("No <body> element found in XML; cannot extract full text.")

        return list(cls._iter_body_blocks(body))
    


def _find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return start


if __name__ == "__main__":
    import argparse
    import sys

    project_root = _find_project_root(Path(__file__).resolve())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint

    default_output = project_root / "tmp" / "article-full-text.json"
    tmp_dir = project_root / "tmp"

    parser = argparse.ArgumentParser(
        description="Extract plain-text full article body from an XML document."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "xml_path",
        nargs="?",
        help="Path to the source XML file",
    )
    input_group.add_argument(
        "--pmcid",
        help="PMC ID to fetch from Entrez and convert directly (e.g., PMC1013555)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(default_output),
        help="Output text path",
    )

    args = parser.parse_args()

    if args.pmcid:
        xml_text = PMCEndpoint.fetch_pmcid_xml(args.pmcid)
        source_name = f"{args.pmcid}.xml"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        xml_out_path = tmp_dir / source_name
        xml_out_path.write_text(xml_text, encoding="utf-8")
        print(f"Wrote source XML: {xml_out_path}")
    else:
        source = Path(args.xml_path)
        xml_text = source.read_text(encoding="utf-8")
        source_name = source.name

    sections = XMLToDictConverter.convert(xml_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote full-text output: {out_path}")