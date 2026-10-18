"""下载 GitHub 仓库目录到本地（走 web_fetch 的替代路径：api.github.com）。

为什么不用 git clone
-------------------
本机沙箱**禁止出站网络**（`git clone` 报 "Failed to connect to github.com:443"），
但 DSH 的 `web_fetch` 工具可以访问 `api.github.com`。
本脚本不自己发请求（发不出去），而是：
    1. 给出该目录下所有文件的 api URL 清单（供 web_fetch 批量抓）
    2. 把抓回来的 base64 响应解码、按原目录结构落盘

用法：
    # 第一步：生成 URL 清单（把清单里的 URL 交给 web_fetch）
    python fetch_github.py plan <owner/repo> <dir> --branch main

    # 第二步：把 web_fetch 的响应存成 <name>.b64 放进 inbox/，然后解码落盘
    python fetch_github.py unpack third_party/<repo> --inbox inbox/

设计取舍：不做并发、不做重试 —— 抓取由 web_fetch 负责，本脚本只管"解码+落盘"，
保持简单可审计（符合 工程目标.md 的"稳 ＞ 高速"）。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def api_url(repo: str, path: str, branch: str) -> str:
    if path in ("", "."):
        return f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    return f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def cmd_plan(args: argparse.Namespace) -> int:
    """列出目录下所有文件，给出每个文件的 api URL。"""
    # 从本地缓存 tree（若有）里解析；否则只给 tree URL 让用户先抓一次
    tree_file = Path(args.tree) if args.tree else None
    if not tree_file or not tree_file.exists():
        print("先用 web_fetch 抓这个 URL（会返回整个仓库的文件树）：")
        print(f"  {api_url(args.repo, '', args.branch)}")
        print(f"\n存成文件后用: python fetch_github.py plan {args.repo} {args.dir} "
              f"--tree <那个文件>")
        return 0

    text = tree_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0))
    prefix = args.dir.strip("/")
    files = [t["path"] for t in obj.get("tree", [])
             if t["type"] == "blob" and t["path"].startswith(prefix)]
    print(f"# {args.repo} 下 {prefix}/ 共 {len(files)} 个文件")
    print(f"# 用 web_fetch 逐个抓下面的 URL，响应分别存为 inbox/<序号>.b64")
    for i, p in enumerate(files):
        print(f"{i:03d}\t{api_url(args.repo, p, args.branch)}\t{p}")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    """把 inbox/*.b64 解码，按映射落盘到目标目录。"""
    inbox = Path(args.inbox)
    out = Path(args.dest)
    if not inbox.exists():
        print(f"[xx] inbox 不存在: {inbox}")
        return 1
    # 映射文件：<序号>.b64 对应的仓库内路径，由 plan 的输出生成
    map_file = inbox / "MAPPING.tsv"
    mapping: dict[str, str] = {}
    if map_file.exists():
        for line in map_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                mapping[parts[0]] = parts[2]
    n_ok = n_skip = 0
    for f in sorted(inbox.glob("*.b64")):
        stem = f.stem
        rel = mapping.get(stem)
        if rel is None:
            n_skip += 1
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
        if not m:
            print(f"  [skip] {f.name}: 未找到 content 字段")
            n_skip += 1
            continue
        raw = m.group(1).encode("utf-8").decode("unicode_escape")
        data = base64.b64decode(raw)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        n_ok += 1
    print(f"落盘 {n_ok} 个文件 -> {out}   （跳过 {n_skip}）")
    return 0 if n_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="GitHub 目录抓取（沙箱禁网时的替代路径）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="列出目录下文件与 api URL")
    p1.add_argument("repo")
    p1.add_argument("dir")
    p1.add_argument("--branch", default="main")
    p1.add_argument("--tree", default=None, help="已抓好的 tree 响应文件")
    p1.set_defaults(func=cmd_plan)

    p2 = sub.add_parser("unpack", help="解码 inbox/*.b64 并落盘")
    p2.add_argument("dest")
    p2.add_argument("--inbox", default="inbox")
    p2.set_defaults(func=cmd_unpack)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
