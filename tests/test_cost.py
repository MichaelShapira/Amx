import json

import boto3
import pytest
from botocore.stub import Stubber

from bench import cost


def _product(price="0.8467200000", unit="Hrs"):
    return json.dumps({
        "product": {"sku": "SKU123", "attributes": {"instanceType": "m8i.4xlarge"}},
        "publicationDate": "2026-11-07T00:28:52Z",
        "terms": {"OnDemand": {"SKU123.JRTCKXETXF": {
            "offerTermCode": "JRTCKXETXF", "effectiveDate": "2026-11-01T00:00:00Z",
            "priceDimensions": {"SKU123.JRTCKXETXF.6YS6EN2CT7": {
                "unit": unit, "pricePerUnit": {"USD": price}}}}}},
    })


def _client():
    return boto3.client("pricing", region_name="us-east-1", aws_access_key_id="x",
                        aws_secret_access_key="x")


def test_fetch_price_parses_on_demand():
    pc = _client()
    with Stubber(pc) as st:
        st.add_response("get_products", {"PriceList": [_product()], "FormatVersion": "aws_v1"})
        p = cost.fetch_price(pc, "m8i.4xlarge", "us-east-1", "arn:x")
    assert p.usd_per_hour == pytest.approx(0.84672)
    assert p.sku == "SKU123" and p.effective_date.startswith("2026-11-01")
    assert p.price_list_arn == "arn:x" and p.publication_date.startswith("2026-11-07")


def test_missing_price_raises():
    pc = _client()
    with Stubber(pc) as st:
        st.add_response("get_products", {"PriceList": [], "FormatVersion": "aws_v1"})
        with pytest.raises(cost.PriceUnavailable):
            cost.fetch_price(pc, "m8i.4xlarge", "us-east-1")


def test_ambiguous_price_raises():
    pc = _client()
    with Stubber(pc) as st:
        st.add_response("get_products", {"PriceList": [_product(), _product("0.9")],
                                         "FormatVersion": "aws_v1"})
        with pytest.raises(cost.PriceUnavailable):
            cost.fetch_price(pc, "m8i.4xlarge", "us-east-1")


def test_unit_costs():
    # $0.8048/h at 1,000 tok/s = 3.6M tok/h -> $0.2236 per 1M tokens
    assert cost.usd_per_million_tokens(0.8048, 1000) == pytest.approx(0.8048 / 3.6)
    assert cost.usd_per_thousand_requests(0.8048, 2) == pytest.approx(0.8048 / 7.2)
    assert cost.usd_per_gb(1.0, 1e6) == pytest.approx(1.0 / 3.6)
    assert cost.gb_per_hour(1e6) == pytest.approx(3.6)
    assert cost.usd_per_million_tokens(1.0, 0) is None


def test_capacity_equivalence():
    ce = cost.capacity_equivalence(gpu_throughput=1000, cpu_throughput=300, cpu_vcpus=16,
                                   gpu_vcpus=4, cpu_price=0.84672, gpu_price=0.8048)
    assert ce.cpu_instances_per_gpu == 4
    assert ce.standard_vcpus_needed == 64
    assert ce.hourly_cost_ratio == pytest.approx(4 * 0.84672 / 0.8048)
    exact = cost.capacity_equivalence(gpu_throughput=600, cpu_throughput=300, cpu_vcpus=16,
                                      gpu_vcpus=4, cpu_price=1, gpu_price=1)
    assert exact.cpu_instances_per_gpu == 2
    assert cost.capacity_equivalence(gpu_throughput=1, cpu_throughput=0, cpu_vcpus=1,
                                     gpu_vcpus=1, cpu_price=1, gpu_price=1) is None
