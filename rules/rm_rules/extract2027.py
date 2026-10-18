"""提取 2027 四份新手册为逐页 JSON + 带页标 txt（沿用 extract.py 的模式）。

产物（rules_text/）：
  rmul2027_v100.json/txt    高校联盟赛 比赛规则手册 V1.0.0
  rmuc2027_v100.json/txt    超级对抗赛 比赛规则手册 V1.0.0
  rmul2027_cansai.json/txt  高校联盟赛 参赛手册 V1.0.0
  spec2027_v100.json/txt    高校系列赛 机器人制作规范手册 V1.0.0
"""
import json
from pathlib import Path

import pymupdf

ROOT = Path(r"C:\RM")
PDF_DIR = ROOT / "rules"          # 2026-10-16 整理：PDF 与抽取文本都在 rules/ 下
OUT = PDF_DIR / "rules_text"
OUT.mkdir(exist_ok=True)

FILES = {
    "rmul2027_v100": "RoboMaster 2027 机甲大师高校联盟赛比赛规则手册 V1.0.0（20261015）.pdf",
    "rmuc2027_v100": "RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）.pdf",
    "rmul2027_cansai": "RoboMaster 2027 机甲大师高校联盟赛参赛手册V1.0.0（20261015）.pdf",
    "spec2027_v100": "RoboMaster 2027 机甲大师高校系列赛机器人制作规范手册V1.0.0（20261015）.pdf",
}

for tag, name in FILES.items():
    pdf = PDF_DIR / name
    if not pdf.exists():
        print(f"[skip] {name} 不存在")
        continue
    doc = pymupdf.open(pdf)
    pages = [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]
    doc.close()
    (OUT / f"{tag}.json").write_text(
        json.dumps({"file": name, "pages": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(OUT / f"{tag}.txt", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n===== p{p['page']} =====\n")
            f.write(p["text"])
    print(f"{tag}: {len(pages)} pages")
