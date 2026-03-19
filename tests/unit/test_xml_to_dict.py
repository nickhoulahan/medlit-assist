from pathlib import Path

import pytest
from lxml import etree as ET

from src.medlit_agent.pmc_service.xml_to_dict import (
    XMLToDictConverter,
    _find_project_root,
)


def test_parse_xml_invalid_raises_value_error():
    with pytest.raises(ValueError, match="Invalid XML content"):
        XMLToDictConverter._parse_xml("<broken")


def test_localname_ignores_non_element_nodes():
    root = ET.fromstring("<root><!-- comment --><body/></root>")
    comment_node = next(root.iter(ET.Comment))

    assert XMLToDictConverter._localname(comment_node) is None
    assert XMLToDictConverter._localname(root.find(".//body")) == "body"


def test_convert_uses_namespace_body_fallback_and_filters_sections():
    xml = """
    <article xmlns=\"urn:test\">
      <front/>
      <body>
        <sec>
          <title>Introduction</title>
          <p>Top-level intro paragraph.</p>
          <sec>
            <title>Nested Section</title>
            <p>Nested details.</p>
          </sec>
        </sec>
        <sec>
          <title>References</title>
          <p>Should be skipped entirely.</p>
        </sec>
      </body>
    </article>
    """

    sections = XMLToDictConverter.convert(xml)

    assert len(sections) == 2
    assert sections[0]["title"] == "Introduction"
    assert sections[0]["body"] == "Top-level intro paragraph."
    assert sections[1]["title"] == "Nested Section"
    assert sections[1]["body"] == "Nested details."


def test_convert_raises_when_body_missing():
    xml = "<article><front/></article>"

    with pytest.raises(ValueError, match="No <body> element found"):
        XMLToDictConverter.convert(xml)


def test_find_body_direct_path_is_used():
    root = ET.fromstring("<article><body><sec><title>A</title></sec></body></article>")
    body = XMLToDictConverter._find_body(root)

    assert body is not None
    assert body.tag == "body"


def test_iter_body_blocks_skips_missing_and_blank_titles():
    xml = """
    <article>
      <body>
        <sec>
          <p>Section with no title should be skipped.</p>
        </sec>
        <sec>
          <title>   </title>
          <p>Blank title should be skipped.</p>
        </sec>
      </body>
    </article>
    """

    assert XMLToDictConverter.convert(xml) == []


def test_find_project_root_prefers_marker_and_falls_back_to_start(tmp_path: Path):
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[tool.black]\n", encoding="utf-8")

    assert _find_project_root(nested) == repo_root

    orphan = tmp_path / "orphan"
    orphan.mkdir()
    assert _find_project_root(orphan) == orphan
