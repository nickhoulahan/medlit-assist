from lxml import etree

from src.medlit_agent.pmc_service.pmc_endpoint import PMCEndpoint


def test_live_pmc_endpoint():
    service = PMCEndpoint()
    results = service.fetch_pmc_records("diabetes")
    assert isinstance(results, list)
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)


def test_live_pmc_endpoint_fetch_pmcids():
    service = PMCEndpoint()
    results = service._fetch_pmc_ids("diabetes")
    assert isinstance(results, list)
    assert len(results) == 5
    assert all(isinstance(pmcid, str) for pmcid in results)
    # endpoint returns number part of PMCIDs, so we can check that they are digits, e.g. "PMC123456" vs "123456"
    assert all(pmcid.isdigit() for pmcid in results)


def test_live_pmc_endpoint_no_results():
    service = PMCEndpoint()
    results = service.fetch_pmc_records("aasdfasdfasdfasdf")
    assert isinstance(results, list)
    assert len(results) == 0


def test_live_pmcid_endpoint_fetch_xml():
    pmcid = "PMC9759163"
    service = PMCEndpoint()

    full_text = service.fetch_pmcid_xml(pmcid)
    root = etree.fromstring(full_text)

    assert root.tag == "pmc-articleset"
    article = root.find(".//article")
    assert article is not None
    article_id = article.find(".//front//article-meta//article-id")
    assert article_id is not None
    assert article_id.text == pmcid

    article_title = article.find(".//front//article-meta//title-group//article-title")
    assert article_title is not None
    assert article_title.text == (
        "Changing the Name of Diabetes Insipidus: A Position "
        "Statement of the Working Group for Renaming Diabetes Insipidus"
    )


def test_live_parse_article_xml():
    pmcid = "PMC9759163"
    service = PMCEndpoint()

    full_text = service.fetch_pmcid_xml(pmcid)
    root = etree.fromstring(full_text)

    article_data = service._parse_article(root, pmcid)

    assert article_data["pmcid"] == pmcid
    assert article_data["apa_citation"] == (
        "Arima, H., Cheetham, T., Christ-Crain, M., Cooper, D., Drummond, J., "
        "Gurnell, M., Levy, M., McCormack, A., Newell-Price, J., Verbalis, J., "
        "Wass, J., The Working Group for Renaming Diabetes Insipidus, Arima, H., "
        "Cheetham, T., Christ-Crain, M., Gurnell, M., Cooper, D., Drummond, J., "
        "Levy, M., Gurnell, M., McCormack, A., Verbalis, J., Newell-Price, J., & Wass, J. "
        "(2022). Changing the Name of Diabetes Insipidus: A Position Statement of the Working "
        "Group for Renaming Diabetes Insipidus. The Journal of Clinical Endocrinology and Metabolism, "
        "108(1), 1–3. https://doi.org/10.1210/clinem/dgac547"
    )
    assert article_data["abstract"] == (
        "Recent data show that patients with a diagnosis of diabetes insipidus (DI) are coming to harm. "
        "Here we give the rationale for a name change to arginine vasopressin deficiency and resistance for "
        "central and nephrogenic DI, respectively."
    )
