from pathlib import Path


REFERENCE_ROWS = [
    {
        "repo": "Mwxinnn/AA-CLIP",
        "commit": "53db195f230442aa118c246876c94ba1c76139cc",
        "status": "PARTIAL_CODE",
        "note": "Remote HEAD was resolved. Source was not imported into this experiment; no implementation code is borrowed.",
    },
    {
        "repo": "Faustinaqq/AF-CLIP",
        "commit": "bb7edec4128a76f29cb573cd3002538bf250b2fe",
        "status": "PARTIAL_CODE",
        "note": "Remote HEAD was resolved. Source was not imported into this experiment; no implementation code is borrowed.",
    },
    {
        "repo": "gaobb/AdaptCLIP",
        "commit": "354d9e3332ec5348b3d8e4439111d34f8e94c0a9",
        "status": "PARTIAL_CODE",
        "note": "Remote HEAD was resolved. Source was not imported into this experiment; no implementation code is borrowed.",
    },
]


def write_reference_audit(path):
    lines = [
        "# Phase VIII Reference Code Audit",
        "",
        "Only repositories with verified core source and verified train/test call paths were used as implementation references.",
        "",
        "No external repository reached `CODE_VERIFIED` status in this local audit, so Phase VIII-A uses only the internally specified symmetric metric transform.",
        "",
        "| Repository | Fixed commit | Status | Note |",
        "|---|---|---|---|",
    ]
    for row in REFERENCE_ROWS:
        lines.append(f"| {row['repo']} | `{row['commit']}` | {row['status']} | {row['note']} |")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

