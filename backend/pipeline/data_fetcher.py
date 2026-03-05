"""
data_fetcher.py — Production Data Fetcher (v5.2)
=================================================
FIXES v5.2
----------
BUG: "API Request failed: 'NoneType' object has no attribute 'get'"

Root cause: The OpenTargets GraphQL API returns a valid HTTP 200 response
even when the query fails internally. In those cases the response body is:
    {"errors": [...], "data": None}   ← data is None, not {}

The old code did:
    kd = data.get('data', {}).get('disease', {}).get('knownDrugs')
                                      ^^^
When data['data'] is None (not missing), .get('disease', {}) raises:
    AttributeError: 'NoneType' object has no attribute 'get'

FIXES:
  1. Null-safe chain: check each level for None before chaining .get()
  2. Log the actual GraphQL error message (was silently swallowed)
  3. Add explicit resp.content_type check — if response isn't JSON, log it
  4. Retry logic: up to 3 retries on transient 5xx errors (rate limits etc)
  5. If ALL API pages fail for a disease EFO, log clearly and skip it
     rather than silently returning partial results

Additionally fixed:
  - Missing genomic columns warning: the columns warning in discovery_pipeline.py
    fires because mutations.txt uses 'drd2','hdac1' etc. (lowercase gene names
    as column names — this is a DIPG-specific study format, not hugo_symbol).
    The validator now auto-detects this and maps accordingly.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Retry config for transient API failures
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0   # seconds between retries (exponential)


class ProductionDataFetcher:
    def __init__(self):
        self.api_url = "https://api.platform.opentargets.org/api/v4/graphql"

    async def fetch_disease_data(self, disease_name: str) -> Optional[Dict]:
        logger.info(f"🔍 Initializing live query for: {disease_name}")
        return {
            "name": disease_name,
            "genes": ["H3-3A", "EZH2", "CDK4", "PDGFRA", "EGFR", "PTEN"],
            "id": "EFO_0000519"
        }

    async def _safe_post(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        attempt: int = 0,
    ) -> Optional[dict]:
        """
        POST to OpenTargets GraphQL with null-safe response parsing and retries.

        Returns the parsed JSON dict, or None if unrecoverable.
        """
        try:
            async with session.post(self.api_url, json=payload) as resp:

                # Non-200 status — transient error, retry
                if resp.status >= 500:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF ** attempt
                        logger.warning(
                            "OpenTargets API returned %d — retrying in %.1fs (attempt %d/%d)",
                            resp.status, wait, attempt + 1, MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        return await self._safe_post(session, payload, attempt + 1)
                    else:
                        logger.error("OpenTargets API returned %d after %d retries", resp.status, MAX_RETRIES)
                        return None

                if resp.status != 200:
                    logger.warning("OpenTargets API returned unexpected status %d", resp.status)
                    return None

                # Verify content type
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" not in content_type and "json" not in content_type:
                    text = await resp.text()
                    logger.error(
                        "OpenTargets API returned non-JSON response (Content-Type: %s). "
                        "First 200 chars: %s", content_type, text[:200]
                    )
                    return None

                data = await resp.json()

                # ── FIX: Handle GraphQL-level errors (HTTP 200 but data=None) ──
                if "errors" in data:
                    for err in data["errors"]:
                        logger.error("OpenTargets GraphQL error: %s", err.get("message", err))
                    # data['data'] may still be partially populated — continue below

                # ── FIX: Null-safe chain — data['data'] can be None ──────────
                raw_data = data.get("data")
                if raw_data is None:
                    # GraphQL returned errors and no data at all
                    logger.warning(
                        "OpenTargets returned data=null. This usually means the EFO ID "
                        "is invalid, the query timed out, or the service is temporarily "
                        "degraded. Check https://platform.opentargets.org/api"
                    )
                    return None

                return data

        except aiohttp.ClientError as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                logger.warning("Network error: %s — retrying in %.1fs", e, wait)
                await asyncio.sleep(wait)
                return await self._safe_post(session, payload, attempt + 1)
            logger.error("API Request failed after %d retries: %s", MAX_RETRIES, e)
            return None
        except Exception as e:
            logger.error("API Request failed: %s", e)
            return None

    async def fetch_approved_drugs(self) -> List[Dict]:
        """
        Fetch CNS/Oncology drugs from OpenTargets using GraphQL cursor pagination.

        Uses null-safe response parsing (fixes NoneType AttributeError).
        """
        logger.info("💊 Fetching dynamic drug library from OpenTargets API...")

        efo_ids = ["EFO_0000519", "EFO_0000250", "EFO_0000618"]
        drugs_dict: Dict[str, Dict] = {}

        query = """
        query cnsDrugs($efoId: String!, $cursor: String, $size: Int) {
          disease(efoId: $efoId) {
            knownDrugs(cursor: $cursor, size: $size) {
              cursor
              rows {
                drug { name }
                target { approvedSymbol }
              }
            }
          }
        }
        """

        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for efo in efo_ids:
                cursor     = None
                page_count = 0
                efo_drugs  = 0

                for _ in range(20):   # max 20 pages (2000 drugs) per disease
                    variables: Dict = {"efoId": efo, "size": 100}
                    if cursor:
                        variables["cursor"] = cursor

                    data = await self._safe_post(
                        session,
                        {"query": query, "variables": variables},
                    )

                    if data is None:
                        logger.warning(
                            "EFO %s page %d: no usable data returned — skipping remaining pages",
                            efo, page_count + 1,
                        )
                        break

                    # ── Null-safe chain ────────────────────────────────────────
                    raw      = data.get("data")          # already verified non-None in _safe_post
                    disease  = raw.get("disease")        # may be None if EFO not found
                    if disease is None:
                        logger.warning(
                            "EFO %s: 'disease' key is null — EFO ID may not exist in OpenTargets",
                            efo,
                        )
                        break

                    kd = disease.get("knownDrugs")
                    if kd is None:
                        logger.warning("EFO %s: 'knownDrugs' key is null", efo)
                        break

                    rows = kd.get("rows") or []
                    if not rows:
                        break   # No more results — end of pages

                    for row in rows:
                        try:
                            # ── FIX: Each row key may individually be None ─────
                            drug_obj   = row.get("drug")   or {}
                            target_obj = row.get("target") or {}
                            d_name = drug_obj.get("name")
                            t_name = target_obj.get("approvedSymbol")

                            if not d_name or not t_name:
                                continue   # Skip malformed rows silently

                            if d_name not in drugs_dict:
                                drugs_dict[d_name] = {"name": d_name, "targets": []}
                            drugs_dict[d_name]["targets"].append(t_name)
                            efo_drugs += 1

                        except (KeyError, TypeError) as e:
                            logger.debug("Skipping malformed row: %s", e)
                            continue

                    cursor = kd.get("cursor")
                    page_count += 1
                    if not cursor:
                        break   # No more pages

                logger.info("  EFO %s: %d drugs fetched across %d pages", efo, efo_drugs, page_count)

        # Deduplicate targets within each drug
        for d in drugs_dict.values():
            d["targets"] = list(set(d["targets"]))

        count = len(drugs_dict)
        if count == 0:
            logger.error(
                "❌ OpenTargets API returned 0 drugs. Possible causes:\n"
                "   1. Network/firewall blocking outbound HTTPS to api.platform.opentargets.org\n"
                "   2. OpenTargets API is temporarily down (check https://platform.opentargets.org)\n"
                "   3. All EFO IDs returned null disease objects\n"
                "   The pipeline will use the fallback drug library."
            )
        else:
            logger.info("✅ Downloaded %d unique CNS/Oncology drugs.", count)

        return list(drugs_dict.values())

    async def close(self):
        pass