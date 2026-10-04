"""Self-contained report assets; no training, network or shared-output writes."""
from html import escape
import json


def render_html(report):
    rows = []
    for e in report.get("estimators", []):
        rows.append('<tr>' + ''.join(f'<td>{escape(str(e[k]))}</td>' for k in ("method", "value", "standard_error")) + '</tr>')
    gates = ''.join(f'<li>{escape(g["gate"])}: {escape(g["status"])} — {escape(g["reason"])}</li>' for g in report["gates"])
    limits = ''.join(f'<li>{escape(s)}</li>' for s in report["limitations"])
    details = escape(json.dumps(report, indent=2, allow_nan=False))
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>OxyFormer evidence report</title></head><body>'
            f'<h1>{escape(report["state"].upper())} — {escape(report["evidence_label"])}</h1>'
            '<p>Completed computation alone does not authorize scientific release.</p>'
            '<h2>All estimators</h2><table><tr><th>Method</th><th>Estimate</th><th>Stored SE</th></tr>' + ''.join(rows) + '</table>'
            '<p>Aligned cluster covariance and every 50/100/200-km spatial sensitivity are in the full record below. '
            'No estimator, seed or bandwidth is selected by sign or significance.</p>'
            f'<h2>Gates</h2><ul>{gates}</ul><h2>Limitations</h2><ul>{limits}</ul>'
            f'<h2>Full diagnostic and provenance record</h2><pre>{details}</pre></body></html>')


def render_forest(report):
    """Point estimates only; never invent an interval when inference is missing."""
    rows = report.get("estimators", [])
    values = [e["value"] for e in rows]
    bound = max([abs(v) for v in values] + [1.0])
    height = 100 + 42 * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="12" y="24">{escape(report["state"].upper())}: {escape(report["evidence_label"])}</text>',
             '<text x="12" y="46">All point estimates; covariance and spatial sensitivity intervals require the full report.</text>']
    for i, e in enumerate(rows):
        y = 80 + 42 * i
        x = 650 + 250 * e["value"] / bound
        parts.extend([f'<text x="12" y="{y}">{escape(e["method"])}: {e["value"]:.6g} {escape(e["spec"]["outcome_scale"])}</text>',
                      f'<circle cx="{x}" cy="{y-4}" r="4"/>'])
    parts.append('</svg>')
    return '\n'.join(parts)
