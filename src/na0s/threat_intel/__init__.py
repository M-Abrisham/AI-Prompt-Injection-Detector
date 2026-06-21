"""Layer 15 — External Threat Intelligence Sync & Monitoring.

Monitors upstream threat intelligence sources (MITRE ATLAS, Garak, AIID,
JailbreakBench, OWASP LLM Top 10, SafetyPrompts) and syncs new attack
techniques, probes, and datasets into Na0S's local taxonomy and detection
pipeline.

This layer runs as a scheduled GitHub Actions workflow (weekly) and can
also be invoked manually via the CLI or Python API.

Components
----------
- **ThreatIntelSource** (base.py): Abstract interface for all sync modules
- **TaxonomyDiffEngine** (diff_engine.py): Compares taxonomy snapshots
- **AtlasSync** (atlas_sync.py): MITRE ATLAS YAML sync
- **GarakSync** (garak_sync.py): Garak probe monitoring
- **AiidSync** (aiid_sync.py): AI Incident Database polling
- **OWASPSync** (owasp_sync.py): OWASP LLM Top 10 monitoring
- **Orchestrator** (orchestrator.py): Runs all sources, produces reports
"""

from na0s.threat_intel.base import (
    ApplyResult,
    SourceSnapshot,
    SyncReport,
    TaxonomyDiff,
    ThreatIntelSource,
)
from na0s.threat_intel.diff_engine import TaxonomyDiffEngine

__all__ = [
    "ApplyResult",
    "SourceSnapshot",
    "SyncReport",
    "TaxonomyDiff",
    "TaxonomyDiffEngine",
    "ThreatIntelSource",
]
