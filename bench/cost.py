"""Prices from the AWS Price List API, cost per unit of work, and capacity equivalence.

Prices are never hard-coded (Requirement 10.2). Cost is reported as information; it does
not decide the migration verdict (Requirement 10.4).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

PRICING_REGION = "us-east-1"  # the Price List API endpoint region


@dataclass(frozen=True)
class Price:
    instance_type: str
    region: str
    usd_per_hour: float
    sku: str
    offer_term_code: str
    rate_code: str
    effective_date: str
    publication_date: str | None
    price_list_arn: str | None
    source: str = "AWS Price List API (GetProducts, AmazonEC2, OnDemand)"

    def to_dict(self) -> dict:
        return asdict(self)


class PriceUnavailable(RuntimeError):
    pass


def _filters(instance_type: str, region: str) -> list[dict]:
    f = {"instanceType": instance_type, "regionCode": region, "operatingSystem": "Linux",
         "tenancy": "Shared", "preInstalledSw": "NA", "capacitystatus": "Used"}
    return [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in f.items()]


def fetch_price(pricing_client, instance_type: str, region: str,
                price_list_arn: str | None = None) -> Price:
    resp = pricing_client.get_products(ServiceCode="AmazonEC2",
                                       Filters=_filters(instance_type, region), MaxResults=10)
    found: list[Price] = []
    for raw in resp.get("PriceList", []):
        product = json.loads(raw)
        for term in (product.get("terms", {}).get("OnDemand") or {}).values():
            for rate_code, dim in term.get("priceDimensions", {}).items():
                usd = float(dim["pricePerUnit"]["USD"])
                if dim.get("unit") != "Hrs" or usd <= 0:
                    continue
                found.append(Price(
                    instance_type=instance_type, region=region, usd_per_hour=usd,
                    sku=product["product"]["sku"], offer_term_code=term["offerTermCode"],
                    rate_code=rate_code, effective_date=term["effectiveDate"],
                    publication_date=product.get("publicationDate"),
                    price_list_arn=price_list_arn))
    if len(found) != 1:
        raise PriceUnavailable(
            f"expected exactly one on-demand Linux price for {instance_type} in {region}, "
            f"got {len(found)}")
    return found[0]


def current_price_list_arn(pricing_client, region: str, effective_iso: str) -> str | None:
    resp = pricing_client.list_price_lists(ServiceCode="AmazonEC2", CurrencyCode="USD",
                                           EffectiveDate=effective_iso, RegionCode=region,
                                           MaxResults=1)
    lists = resp.get("PriceLists") or []
    return lists[0]["PriceListArn"] if lists else None


def snapshot(pricing_client, instance_types: list[str], region: str, effective_iso: str) -> dict:
    arn = current_price_list_arn(pricing_client, region, effective_iso)
    return {t: fetch_price(pricing_client, t, region, arn).to_dict() for t in sorted(set(instance_types))}


def per_unit(usd_per_hour: float, units_per_s: float | None, per: float) -> float | None:
    """USD per `per` units of work at a sustained rate; None when there is no throughput."""
    if not units_per_s or units_per_s <= 0:
        return None
    return usd_per_hour / (units_per_s * 3600.0) * per


def usd_per_million_tokens(usd_per_hour: float, tok_s: float | None) -> float | None:
    return per_unit(usd_per_hour, tok_s, 1e6)


def usd_per_thousand_requests(usd_per_hour: float, req_s: float | None) -> float | None:
    return per_unit(usd_per_hour, req_s, 1e3)


def usd_per_million_items(usd_per_hour: float, items_s: float | None) -> float | None:
    return per_unit(usd_per_hour, items_s, 1e6)


def usd_per_gb(usd_per_hour: float, bytes_s: float | None) -> float | None:
    """GB = 10^9 bytes, matching Amazon Macie's unit."""
    return per_unit(usd_per_hour, bytes_s, 1e9)


def gb_per_hour(bytes_s: float | None) -> float | None:
    return None if not bytes_s else bytes_s * 3600.0 / 1e9


@dataclass(frozen=True)
class CapacityEquivalence:
    cpu_instances_per_gpu: int
    standard_vcpus_needed: int
    gpu_vcpus_replaced: int
    hourly_cost_ratio: float
    gpu_throughput: float
    cpu_throughput: float


def capacity_equivalence(*, gpu_throughput: float, cpu_throughput: float, cpu_vcpus: int,
                         gpu_vcpus: int, cpu_price: float, gpu_price: float) -> CapacityEquivalence | None:
    """CPU instances (and Standard vCPUs) that replace one GPU instance (design 12.1)."""
    if not gpu_throughput or not cpu_throughput or cpu_throughput <= 0:
        return None
    n = max(1, math.ceil(gpu_throughput / cpu_throughput - 1e-9))
    return CapacityEquivalence(
        cpu_instances_per_gpu=n, standard_vcpus_needed=n * cpu_vcpus, gpu_vcpus_replaced=gpu_vcpus,
        hourly_cost_ratio=n * cpu_price / gpu_price, gpu_throughput=gpu_throughput,
        cpu_throughput=cpu_throughput)
