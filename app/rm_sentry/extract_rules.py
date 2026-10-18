"""把项目根目录下的规则 PDF 抽成文本，便于逐条核对环境常量。

用法（Windows 与 Ubuntu 通用）：
    cd <仓库根>/app
    python -m rm_sentry.extract_rules
输出：
    <仓库根>/rules/rules_text/<pdf 名>.txt  以及 .json（分页）

路径解析规则（跨平台）：
    1. 若设置了环境变量 RM_ROOT，用它；
    2. 否则用本文件向上三级 —— 即 <仓库根>/app/rm_sentry/extract_rules.py -> <仓库根>。
不写死任何绝对路径，Windows / Linux 都能跑。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pymupdf


def project_root() -> Path:
    env = os.environ.get("RM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # 本文件位于 <root>/app/rm_sentry/extract_rules.py
    return Path(__file__).resolve().parent.parent.parent


ROOT = project_root()
PDF_DIR = ROOT / "rules"          # 2026-10-16 整理：规则 PDF 与抽取文本都在 rules/ 下
OUT = PDF_DIR / "rules_text"


def slug(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name)
    return s.strip("_")[:80]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"在 {ROOT} 下没找到 PDF")
        print("（可设环境变量 RM_ROOT 指向规则 PDF 所在目录）")
        return
    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        pages = [doc[i].get_text() for i in range(doc.page_count)]
        base = slug(pdf.stem)
        (OUT / f"{base}.txt").write_text("\n".join(pages), encoding="utf-8")
        (OUT / f"{base}.json").write_text(
            json.dumps({"file": pdf.name, "pages": pages}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[ok] {pdf.name}: {doc.page_count} 页 -> {base}.txt")
        doc.close()
    print(f"\n输出目录: {OUT}")


if __name__ == "__main__":
    main()
