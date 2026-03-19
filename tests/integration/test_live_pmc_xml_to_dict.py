from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint
from src.medlit_agent.pmc_service.xml_to_dict import XMLToDictConverter


def test_xml_to_dict_live():
    service = PMCEndpoint()
    converter = XMLToDictConverter()
    pmcid = "PMC9759163"
    xml_content = service.fetch_pmcid_xml(pmcid)
    list_output = converter.convert(xml_content)

    assert isinstance(list_output, list)
    for item in list_output:
        assert isinstance(item, dict)
        assert "title" in item
        assert isinstance(item.get("title"), str)

        assert "body" in item
        assert isinstance(item.get("body"), str)


def test_xml_to_dict_live_no_body():
    service = PMCEndpoint()
    converter = XMLToDictConverter()
    pmcid = "PMC11111"  #  This article doesn't exist, so the XML will be missing the <body> element
    xml_content = service.fetch_pmcid_xml(pmcid)

    try:
        converter.convert(xml_content)
        assert False, "Expected ValueError for missing <body> element"
    except ValueError as e:
        assert str(e) == "No <body> element found in XML; cannot extract full text."
