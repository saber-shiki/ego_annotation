"""Extracted formal interval mechanisms; no historical script imports."""
from __future__ import annotations

def pose_graph_readiness(report: dict[str, Any]) -> dict[str, Any]:
    from typing import Any
    graph = report.get('graph_support') if isinstance(report.get('graph_support'), dict) else {}
    annotation_ready = report.get('annotation_ready')
    support_sufficient = graph.get('sufficient')
    explicitly_unready = annotation_ready is False or support_sufficient is False
    explicitly_ready = annotation_ready is True and support_sufficient is not False
    return {'pose_report_status': report.get('status'), 'annotation_ready': annotation_ready if isinstance(annotation_ready, bool) else None, 'graph_support_sufficient': support_sufficient if isinstance(support_sufficient, bool) else None, 'explicitly_unready': explicitly_unready, 'explicitly_ready': explicitly_ready, 'legacy_readiness_fields_missing': annotation_ready is None and support_sufficient is None}
