"""The SEC fact contract — ONE list, three consumers.

`universe.ingest_next_filing_facts` fetches tags, `neural._fundamental_series`
builds features from tags, and the research loop gates training on whether
enough issuers are "covered". Those three each carried their own idea of which
tags mattered, and they drifted:

* The ingester's list was widened on 2026-07-15. Issuers already marked
  complete only re-fetch weekly, so five days later no issuer had refreshed and
  the store still held the OLD narrow set.
* `_fundamental_series` asked for 14 tags; the store had 6 of them.
* The coverage gate counted issuers with ANY fact row, so 213 issuers holding
  5 of the 14 needed tags reported as fully covered.

Net effect: training ran with 12 of 14 fundamental features pinned at constant
zero, reported a challenger, and nothing failed. The features were not wrong —
they were absent, and absence looked identical to "present and uninformative".

Anything that fetches, reads, or counts SEC facts imports from here.
"""
from __future__ import annotations

# Tags `neural._fundamental_series` actually reads. If a tag leaves this set,
# the feature that consumes it must go too.
REQUIRED_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "OperatingIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "Assets",
    "NetIncomeLoss",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "CommonStocksIncludingAdditionalPaidInCapital",
    "CommonStockSharesOutstanding",
)

# Additionally fetched: used by valuation/event features or kept because the
# SEC returns them in the same request and re-fetching later is far more
# expensive than storing them now.
SUPPLEMENTARY_TAGS = (
    "EarningsPerShareDiluted",
    "GrossProfit",
    "OperatingExpenses",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "ShortTermBorrowings",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
)

FETCH_TAGS = frozenset(REQUIRED_TAGS) | frozenset(SUPPLEMENTARY_TAGS)

# An issuer counts as covered only if it carries at least this share of the
# tags the features need. Tags genuinely absent from an issuer's filings (a
# firm with no long-term debt never reports LongTermDebt) make a 100%
# requirement unreachable, so this is a threshold rather than an equality.
MIN_TAGS_PER_ISSUER = 0.5


def covered_issuer_sql(alias: str = "f") -> str:
    """SQL fragment counting DISTINCT tags from REQUIRED_TAGS for one issuer."""
    marks = ",".join(f"'{tag}'" for tag in REQUIRED_TAGS)
    return f"COUNT(DISTINCT CASE WHEN {alias}.tag IN ({marks}) THEN {alias}.tag END)"


def required_tag_floor() -> int:
    """How many distinct required tags an issuer needs to count as covered."""
    return max(1, int(len(REQUIRED_TAGS) * MIN_TAGS_PER_ISSUER))
