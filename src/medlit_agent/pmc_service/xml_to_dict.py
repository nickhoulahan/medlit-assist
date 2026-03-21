from __future__ import annotations

import re
from typing import Dict, List

from lxml import etree as ET


class XMLToDictConverter:
    """Extract article full text from NLM/JATS XML.

    get body from /pmc-articleset/article/body or //article/body fallback
    and convert section titles + paragraphs dict of chunks
    """

    skip_sections = {
        "references",
        "acknowledgments",
        "funding",
        "conflict of interest",
        "supplementary material",
        "author contributions",
        "supplementary information",
    }

    @staticmethod
    def _parse_xml(xml_content: str | bytes) -> ET._Element:
        try:
            if isinstance(xml_content, str):
                xml_content = xml_content.encode("utf-8")
            return ET.fromstring(xml_content)
        except ET.XMLSyntaxError as exc:
            raise ValueError(f"Invalid XML content: {exc}") from exc

    @staticmethod
    def _localname(node: ET._Element) -> str | None:
        """return an element local name, skipping non-element iter nodes."""
        tag = getattr(node, "tag", None)
        if not isinstance(tag, str):
            return None
        return ET.QName(node).localname

    @classmethod
    def _find_body(cls, root: ET._Element) -> ET._Element | None:
        body = root.find(".//body")
        if body is not None:
            return body

        # namespace-aware fallback for documents that use a default namespace.
        for elem in root.iter():
            if cls._localname(elem) == "body":
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
            if cls._localname(sec) != "sec":
                continue

            title_elem = next(
                (child for child in sec if cls._localname(child) == "title"),
                None,
            )
            if title_elem is None:
                continue

            title_text = cls._clean_text("".join(title_elem.itertext()))
            if not title_text:
                continue
            normalized_title_text = cls._normalize_section_title(title_text)
            if any(
                skip_title in normalized_title_text
                for skip_title in normalized_skip_titles
            ):
                continue

            paragraphs: List[str] = []
            for p in sec.iter():
                if cls._localname(p) != "p":
                    continue

                # keep only paragraphs that belong to this section directly,
                # not nested subsections.
                parent = p.getparent()
                nearest_sec = None
                while parent is not None:
                    if cls._localname(parent) == "sec":
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
    def _extract_body_paragraphs(cls, body: ET._Element) -> List[str]:
        paragraphs: List[str] = []

        for p in body.iter():
            if cls._localname(p) != "p":
                continue

            para = cls._clean_text("".join(p.itertext()))
            if para:
                paragraphs.append(para)

        return paragraphs

    @classmethod
    def convert(
        cls,
        xml_content: str | bytes,
    ) -> List[Dict[str, str]]:
        """return article sections from XML ``body`` as title/body dictionaries."""
        root = cls._parse_xml(xml_content)
        body = cls._find_body(root)
        if body is None:
            raise ValueError(
                "No <body> element found in XML; cannot extract full text."
            )
        sections = list(cls._iter_body_blocks(body))
        if sections:
            return sections

        # some PMC documents have paragraphs under <body> but no <sec> wrappers.
        paragraphs = cls._extract_body_paragraphs(body)
        if not paragraphs:
            return []

        return [{"title": "Body", "body": "\n\n".join(paragraphs)}]
