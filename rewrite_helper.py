#!/usr/bin/env python3
"""
AIGC改写辅助工具
================

提供改写过程中的辅助功能,与 analyze.py 配套使用。

子命令:
    locate    定位需要改写的位置(长句/AI爱用词/并列结构/缺失人工补充)
    suggest   生成【人工补充】候选标记位置
    split     拆分长句(基于标点)
    replace   将AI爱用词替换为模糊限定建议(只输出建议,不实际替换)
    chapters  拆分文档章节并按重要度评估降率优先级
    cohesion  跨章节一致性检查(拼接预警检测) - 句长/人称/复杂度
    critique  为文献综述提供批判性注入候选点位

用法:
    python rewrite_helper.py locate <文件>
    python rewrite_helper.py suggest <文件>
    python rewrite_helper.py split <文件>
    python rewrite_helper.py replace <文件>
    python rewrite_helper.py chapters <文件>
    python rewrite_helper.py cohesion <文件>     # 拼接预警检查
    python rewrite_helper.py critique <文件>     # 批判性注入候选

无外部依赖,仅依赖标准库。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from analyze import (
        AI_LOVE_WORDS,
        COLLOQUIAL_WORDS,
        FIRST_PERSON,
        HEDGE_WORDS,
        HUMAN_FILL_PATTERN,
        PARALLEL_PATTERNS,
        SENT_END_PATTERN,
        split_paragraphs,
        split_sentences,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze import (  # type: ignore[no-redef]
        AI_LOVE_WORDS,
        COLLOQUIAL_WORDS,
        FIRST_PERSON,
        HEDGE_WORDS,
        HUMAN_FILL_PATTERN,
        PARALLEL_PATTERNS,
        SENT_END_PATTERN,
        split_paragraphs,
        split_sentences,
    )


# AI爱用词 → 替换建议
REPLACE_DICT: dict[str, list[str]] = {
    "显著": ["大致", "颇为", "在不小程度上", "看得见地"],
    "重要": ["值得一提的", "关键中的关键", "牵一发的", "不能绕开的"],
    "关键": ["要害的", "卡脖子的", "核心的", "牵一发的"],
    "进一步": ["接着", "再往前推一步", "顺着这条线"],
    "系统": ["整套", "成体系地", "顺着一条线"],
    "机制": ["路子", "做法", "套路", "架构"],
    "全面": ["整体上", "面儿上", "粗略说"],
    "深入": ["往里头", "贴近地", "细究一下"],
    "持续": ["一直", "持久", "陆续"],
    "切实": ["实打实", "真刀真枪", "见效"],
    "充分": ["够用", "足量", "差不多够"],
    "总体": ["整体上", "粗略看", "大方向上"],
    "整体": ["整盘", "大方向", "粗略看"],
    "有效": ["管用", "见效", "立得住"],
    "广泛": ["铺得开", "覆盖较多", "多处"],
    "深刻": ["深一层", "往里挖", "见骨"],
    "良好": ["不错", "尚可", "拿得出手"],
    "明显": ["看得见", "肉眼可见", "一望而知"],
    "积极": ["主动地", "推得动", "肯出力"],
    "高效": ["快当", "省事儿", "效率不低"],
    "优化": ["调一调", "理顺", "再打磨"],
    "提升": ["拔一拔", "上一个台阶", "往上走"],
    "推进": ["往前推", "继续做", "动起来"],
    "加强": ["再加把劲", "往实里做", "夯实"],
    "完善": ["补齐", "再细化", "把缺口补上"],
    "构建": ["搭起来", "拼出来", "造出"],
    "形成": ["攒成", "落出", "凝出"],
    "实现": ["把...做成", "落地", "跑通"],
    "促进": ["带一带", "推一把", "帮衬"],
    "推动": ["拉一把", "撬动", "把...带起来"],
    "助力": ["帮一手", "添把柴", "搭把手"],
    "赋能": ["撑一撑", "给底气", "托底"],
}


# 适合放置【人工补充】的句末提示词
FILL_TRIGGERS = [
    "数据", "比例", "占比", "约", "大约", "估计", "粗略", "左右", "%",
    "瓶颈", "短板", "障碍", "堵点", "卡点", "难点",
    "调研", "访谈", "了解到", "发现", "注意到",
    "时间", "时期", "阶段", "节点",
    "案例", "例如", "举例", "比如",
    "风险", "教训", "失败",
]


@dataclass
class Issue:
    line_no: int
    col: int
    kind: str
    text: str
    hint: str


def find_issues(text: str) -> list[Issue]:
    """扫描文本,定位所有需要改写的点"""
    issues: list[Issue] = []
    lines = text.splitlines()

    for ln, line in enumerate(lines, 1):
        # 长句
        for sent in split_sentences(line):
            if len(sent) >= 50:
                col = line.find(sent[:20])
                issues.append(Issue(ln, max(col, 0), "长句", sent[:30] + "...",
                                    f"长度{len(sent)}字,建议拆为2-3句"))

        # AI爱用词
        for word in AI_LOVE_WORDS:
            for m in re.finditer(re.escape(word), line):
                hint = "替换为: " + " / ".join(REPLACE_DICT.get(word, ["更口语化的表达"])[:3])
                issues.append(Issue(ln, m.start(), "AI爱用词", word, hint))

        # 并列结构
        for word in PARALLEL_PATTERNS:
            for m in re.finditer(re.escape(word), line):
                issues.append(Issue(ln, m.start(), "并列结构", word,
                                    "改为段落式叙述或思维跳跃"))

    return issues


def suggest_fill_marks(text: str) -> list[tuple[int, str, str]]:
    """识别可植入【人工补充】的位置,返回 (段号,触发词,建议类型)"""
    suggestions: list[tuple[int, str, str]] = []
    paragraphs = split_paragraphs(text)

    for i, para in enumerate(paragraphs, 1):
        # 检查段落已有的人工补充
        existing = len(HUMAN_FILL_PATTERN.findall(para))
        if existing >= 2:
            continue

        # 数据/比例触发
        if re.search(r"\d+(\.\d+)?\s*[%家个项条次]", para) or "数据" in para or "占比" in para:
            suggestions.append((i, "数据", "粗略本地数据,如「约X家但配套的不足X%」"))
        # 障碍/堵点触发
        if any(w in para for w in ["瓶颈", "短板", "堵点", "卡点", "难点", "障碍"]):
            suggestions.append((i, "障碍", "具体制度障碍,如「审批涉及X方,周期约X个月」"))
        # 调研/访谈触发
        if any(w in para for w in ["调研", "访谈", "走访", "了解"]):
            suggestions.append((i, "访谈", "1-2句匿名访谈原话"))
        # 风险触发
        if any(w in para for w in ["风险", "若...", "若不", "脱节", "失败"]):
            suggestions.append((i, "风险", "具体的项目风险或历史教训"))
        # 时间锚点
        if any(w in para for w in ["近年", "目前", "当前", "最近", "未来"]):
            suggestions.append((i, "时间", "具体月份或政策节点,如「2024年三季度」"))

    return suggestions


def split_long_sentence(sent: str, target_len: int = 30) -> list[str]:
    """按逗号/分号尝试将长句拆分,使各片段接近 target_len 字"""
    if len(sent) <= target_len:
        return [sent]

    # 在逗号、分号、顿号处尝试切分
    parts = re.split(r"([,，;；])", sent)
    if len(parts) == 1:
        return [sent]

    result: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if p in ",，;；":
            buf += p
            continue
        if len(buf) + len(p) > target_len and buf:
            result.append(buf.rstrip(",，;；"))
            buf = p
        else:
            buf += p
    if buf:
        result.append(buf.rstrip(",，;；"))
    return result


def detect_chapters(text: str) -> list[tuple[int, str, str, int]]:
    """识别文档章节(按 # / 一、二、 / 数字编号),返回 (序号,章节名,类型,权重)"""
    weight_map = {
        "摘要": 2.0, "abstract": 2.0,
        "引言": 1.8, "前言": 1.8, "导论": 1.8, "introduction": 1.8,
        "结论": 1.8, "conclusion": 1.8, "总结": 1.5,
        "讨论": 1.3, "discussion": 1.3,
        "意义": 1.3, "贡献": 1.3,
        "方法": 1.0, "method": 1.0,
        "结果": 1.0, "result": 1.0,
        "正文": 1.0,
    }

    chapters: list[tuple[int, str, str, int]] = []
    lines = text.splitlines()
    chapter_idx = 0

    md_h_re = re.compile(r"^(#{1,4})\s+(.+)$")
    cn_idx_re = re.compile(r"^([一二三四五六七八九十]+)、\s*(.+)$")
    num_idx_re = re.compile(r"^(\d+(?:\.\d+)*)\s*[、.]\s*(.+)$")

    for ln, line in enumerate(lines, 1):
        line_s = line.strip()
        title = None
        if (m := md_h_re.match(line_s)):
            title = m.group(2).strip()
        elif (m := cn_idx_re.match(line_s)):
            title = m.group(2).strip()
        elif (m := num_idx_re.match(line_s)):
            title = m.group(2).strip()

        if title:
            chapter_idx += 1
            weight = 1.0
            for kw, w in weight_map.items():
                if kw in title.lower():
                    weight = w
                    break
            kind = "高权重" if weight >= 1.5 else ("中权重" if weight >= 1.2 else "标准")
            chapters.append((chapter_idx, title, kind, int(weight * 10)))

    return chapters


# ==================== 拼接预警/章节细分 ====================

@dataclass
class ChapterSection:
    idx: int
    title: str
    weight: float          # 权重(2.0/1.8/1.5/1.0/0.8...)
    kind: str              # 高权重/中权重/标准/低权重
    start_line: int
    end_line: int
    content: str           # 该章节正文(不含标题行)


def segment_chapters(text: str) -> list[ChapterSection]:
    """将文本分段为章节,返回 ChapterSection 列表(含正文)"""
    weight_map = {
        "摘要": 1.8, "abstract": 1.8,
        "引言": 1.5, "前言": 1.5, "导论": 1.5, "introduction": 1.5,
        "结论": 1.5, "conclusion": 1.5, "总结": 1.5,
        "创新点": 1.5, "创新": 1.5,
        "讨论": 1.0, "discussion": 1.0,
        "综述": 1.0, "文献": 1.0, "review": 1.0,
        "方法": 1.0, "method": 1.0, "设计": 1.0,
        "结果": 1.0, "result": 1.0, "分析": 1.0,
        "正文": 1.0,
        "背景": 0.7, "理论": 0.7, "局限": 0.7, "致谢": 0.7,
    }

    md_h_re = re.compile(r"^(#{1,4})\s+(.+)$")
    cn_idx_re = re.compile(r"^([一二三四五六七八九十]+)、\s*(.+)$")
    num_idx_re = re.compile(r"^(\d+(?:\.\d+)*)\s*[、.]\s*(.+)$")

    lines = text.splitlines()

    # 找所有章节标题行
    headers: list[tuple[int, str]] = []  # (line_no_zero_indexed, title)
    for i, line in enumerate(lines):
        line_s = line.strip()
        title = None
        if (m := md_h_re.match(line_s)):
            title = m.group(2).strip()
        elif (m := cn_idx_re.match(line_s)):
            title = m.group(2).strip()
        elif (m := num_idx_re.match(line_s)):
            title = m.group(2).strip()
        if title:
            headers.append((i, title))

    sections: list[ChapterSection] = []
    for idx, (line_no, title) in enumerate(headers, 1):
        end_line = headers[idx][0] - 1 if idx < len(headers) else len(lines) - 1
        body = "\n".join(lines[line_no + 1:end_line + 1]).strip()

        weight = 1.0
        for kw, w in weight_map.items():
            if kw in title.lower():
                weight = w
                break
        if weight >= 1.5:
            kind = "高权重"
        elif weight >= 1.0:
            kind = "中权重"
        else:
            kind = "低权重"

        sections.append(ChapterSection(
            idx=idx, title=title, weight=weight, kind=kind,
            start_line=line_no + 1, end_line=end_line + 1,
            content=body,
        ))

    return sections


def chapter_metrics(content: str) -> dict[str, float]:
    """计算章节内容的关键指纹: 平均句长/句长std/人称/复杂度"""
    sentences = split_sentences(content)
    sent_lens = [len(s) for s in sentences if s.strip()]

    avg_sent_len = (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
    if len(sent_lens) > 1:
        mean = avg_sent_len
        var = sum((x - mean) ** 2 for x in sent_lens) / len(sent_lens)
        std = var ** 0.5
    else:
        std = 0.0

    # 人称使用
    we_cnt = content.count("我们")
    team_cnt = content.count("课题组")
    author_cnt = content.count("笔者")

    # 复杂度近似: 长句(≥40字)占比
    long_pct = (sum(1 for x in sent_lens if x >= 40) / len(sent_lens) * 100) if sent_lens else 0.0

    # 标点比例(逗号:句号)
    text_no_marks = HUMAN_FILL_PATTERN.sub("", content)
    comma = sum(text_no_marks.count(c) for c in "，,")
    period = sum(text_no_marks.count(c) for c in "。!?！?")
    comma_period = (comma / period) if period > 0 else 0.0

    return {
        "char_count": float(len(content)),
        "sent_count": float(len(sent_lens)),
        "avg_sent_len": round(avg_sent_len, 1),
        "sent_len_std": round(std, 1),
        "long_sent_pct": round(long_pct, 1),
        "we_count": float(we_cnt),
        "team_count": float(team_cnt),
        "author_count": float(author_cnt),
        "comma_period": round(comma_period, 2),
    }


# ==================== 批判性注入候选检测 ====================

# 引述既有研究的提示词(出现这类词后如果没有批判性评价,就是批判注入候选位)
QUOTE_INDICATORS = [
    "研究表明", "研究发现", "研究指出", "研究认为",
    "学者认为", "学者指出", "学者发现",
    "已有研究", "现有研究", "既有研究", "前人研究",
    "et al", "等(", "等(",
    "提出", "认为", "指出", "发现",
]

# 已存在批判性评价的标志词(若同段已有,无需再加)
CRITIQUE_INDICATORS = [
    "样本仅限", "样本局限", "推广性", "外部效度",
    "内生性", "因果不稳健", "选择偏差", "测量误差",
    "数据截至", "时效性", "适用性存疑",
    "假定", "忽略了", "未考虑", "理论假设过强",
    "未在", "未能复现", "尚未复现",
    "但", "然而", "不过", "可惜", "局限",
]


def find_critique_candidates(text: str) -> list[tuple[int, str, str]]:
    """识别需要批判性注入的段落.

    返回 [(段号, 段落预览, 建议批判类型)]
    """
    paragraphs = split_paragraphs(text)
    candidates: list[tuple[int, str, str]] = []

    for i, para in enumerate(paragraphs, 1):
        # 段落含"研究表明/研究发现"等引述提示词
        has_quote = any(w in para for w in QUOTE_INDICATORS)
        if not has_quote:
            continue

        # 已经有批判性评价的跳过
        critique_density = sum(1 for w in CRITIQUE_INDICATORS if w in para)
        if critique_density >= 1:
            continue

        # 选择最相关的批判类型
        if "样本" in para or "数据" in para:
            critique_type = "样本局限批判: (但样本仅限X,能否推广到Y不明)"
        elif "假设" in para or "假定" in para or "理性" in para:
            critique_type = "假设偏差批判: (假定个体决策完全理性,忽略了行为偏差)"
        elif "因果" in para or "影响" in para or "效应" in para:
            critique_type = "方法缺陷批判: (该研究存在内生性问题,因果推断不稳健)"
        elif any(y in para for y in ["20", "19", "201", "202"]):
            critique_type = "时效性批判: (研究数据较早,在当前背景下适用性存疑)"
        else:
            critique_type = "结论强度批判: (结论虽具启发性,但未在跨文化样本中复现)"

        # 摘要预览
        preview = para[:60].replace("\n", " ") + ("..." if len(para) > 60 else "")
        candidates.append((i, preview, critique_type))

    return candidates


# ==================== 命令实现 ====================

def cmd_locate(text: str) -> str:
    issues = find_issues(text)
    if not issues:
        return "✓ 未发现明显问题点"

    out: list[str] = []
    out.append(f"共发现 {len(issues)} 处需改写位置\n")
    out.append(f"{'行':>4} {'列':>4} {'类型':<8} {'内容':<32} 建议")
    out.append("-" * 80)
    for it in issues[:80]:
        text_disp = it.text if len(it.text) <= 30 else it.text[:28] + ".."
        out.append(f"{it.line_no:>4} {it.col:>4} {it.kind:<8} {text_disp:<32} {it.hint}")
    if len(issues) > 80:
        out.append(f"... 还有 {len(issues) - 80} 处未显示")
    return "\n".join(out)


def cmd_suggest(text: str) -> str:
    sugs = suggest_fill_marks(text)
    if not sugs:
        return "✓ 当前文本无明显需要植入【人工补充】的段落"

    out: list[str] = []
    out.append(f"建议植入 {len(sugs)} 处【人工补充】标记\n")
    seen_paras: set[int] = set()
    for para_no, kind, hint in sugs:
        marker = "  " if para_no in seen_paras else f"段{para_no}"
        seen_paras.add(para_no)
        out.append(f"  {marker:6s} [{kind}] 【人工补充: {hint}】")
    out.append("\n注意: 实际内容必须由真人手写,AI不要代笔")
    return "\n".join(out)


def cmd_split(text: str) -> str:
    out: list[str] = []
    sentences = split_sentences(text)
    long_sents = [s for s in sentences if len(s) >= 50]
    if not long_sents:
        return "✓ 未发现需拆分的长句"

    out.append(f"发现 {len(long_sents)} 个长句(≥50字),拆分建议:\n")
    for i, sent in enumerate(long_sents[:20], 1):
        parts = split_long_sentence(sent, target_len=30)
        out.append(f"[{i}] 原句({len(sent)}字):")
        out.append(f"    {sent}")
        out.append(f"    → 建议拆为 {len(parts)} 句:")
        for j, p in enumerate(parts, 1):
            out.append(f"      {j}. {p} ({len(p)}字)")
        out.append("")
    return "\n".join(out)


def cmd_replace(text: str) -> str:
    out: list[str] = []
    found: dict[str, int] = {}
    for word in AI_LOVE_WORDS:
        c = text.count(word)
        if c > 0:
            found[word] = c

    if not found:
        return "✓ 未发现典型AI爱用词"

    out.append(f"发现 {len(found)} 个AI爱用词,替换建议:\n")
    for word, count in sorted(found.items(), key=lambda x: -x[1]):
        replacements = REPLACE_DICT.get(word, ["(无建议)"])
        out.append(f"  「{word}」 出现 {count} 次")
        out.append(f"    → 候选: {' / '.join(replacements)}")
    out.append("\n注意: 不要全部替换,保留 1-2 处自然表达即可,避免过度刻意")
    return "\n".join(out)


def cmd_chapters(text: str) -> str:
    chapters = detect_chapters(text)
    if not chapters:
        return "未识别到明显章节结构 (建议手动按摘要/引言/结论分块处理)"

    out: list[str] = []
    out.append(f"识别到 {len(chapters)} 个章节,降率优先级如下:\n")
    out.append(f"{'序号':<4} {'权重':<8} {'章节名'}")
    out.append("-" * 60)
    for idx, title, kind, weight in chapters:
        out.append(f"{idx:<4} {kind:<8} {title}")

    out.append("\n建议处理顺序:")
    high = [c for c in chapters if c[2] == "高权重"]
    mid = [c for c in chapters if c[2] == "中权重"]
    if high:
        out.append("  优先级1 (人工注血加倍):")
        for _, t, _, _ in high:
            out.append(f"    - {t}")
    if mid:
        out.append("  优先级2 (常规处理):")
        for _, t, _, _ in mid:
            out.append(f"    - {t}")
    return "\n".join(out)


def cmd_cohesion(text: str) -> str:
    """跨章节一致性检查 - 检测拼接预警风险.

    维普2026会比对章节间风格差异:
    - 句长差异 >5字 → 易触发拼接预警
    - 人称切换(我们/课题组混用)
    - 复杂度突变(长句占比波动)
    """
    sections = segment_chapters(text)
    if len(sections) < 2:
        return ("未识别到多个章节,无法做拼接一致性检查\n"
                "提示: 请确保文档使用 # / 一、 / 1. 等章节标记")

    out: list[str] = []
    out.append("=" * 70)
    out.append("           跨章节一致性检查 (拼接预警检测)")
    out.append("=" * 70)
    out.append("")

    # 各章节指标
    section_data: list[tuple[ChapterSection, dict[str, float]]] = []
    for sec in sections:
        if len(sec.content) < 50:
            continue
        m = chapter_metrics(sec.content)
        section_data.append((sec, m))

    if not section_data:
        return "所有章节内容过短,无法分析"

    out.append(f"{'章节':<24} {'权重':<6} {'字数':>5} {'句长':>6} {'长句%':>6} {'我们':>4} {'课题组':>6} {'笔者':>4}")
    out.append("-" * 70)
    for sec, m in section_data:
        title_disp = sec.title if len(sec.title) <= 22 else sec.title[:20] + ".."
        out.append(
            f"{title_disp:<24} ×{sec.weight:<5} {int(m['char_count']):>5} "
            f"{m['avg_sent_len']:>6.1f} {m['long_sent_pct']:>5.1f}% "
            f"{int(m['we_count']):>4} {int(m['team_count']):>6} {int(m['author_count']):>4}"
        )
    out.append("")

    # 风险检测
    warnings: list[str] = []

    # 1) 句长差异
    avg_lens = [m["avg_sent_len"] for _, m in section_data]
    if avg_lens:
        max_len = max(avg_lens)
        min_len = min(avg_lens)
        diff = max_len - min_len
        if diff > 5:
            max_sec = next(s for s, m in section_data if m["avg_sent_len"] == max_len)
            min_sec = next(s for s, m in section_data if m["avg_sent_len"] == min_len)
            warnings.append(
                f"⚠️  句长差异 {diff:.1f}字 (>5字易触发拼接预警)\n"
                f"     最长: 「{max_sec.title}」 {max_len:.1f}字 vs "
                f"最短: 「{min_sec.title}」 {min_len:.1f}字"
            )

    # 2) 人称混用
    total_we = sum(m["we_count"] for _, m in section_data)
    total_team = sum(m["team_count"] for _, m in section_data)
    total_author = sum(m["author_count"] for _, m in section_data)
    persons_used = sum(1 for x in [total_we, total_team, total_author] if x > 0)
    if persons_used >= 2:
        warnings.append(
            f"⚠️  人称混用: 我们={int(total_we)} / 课题组={int(total_team)} / "
            f"笔者={int(total_author)}\n"
            f"     建议全文统一用一种人称,避免拼接预警"
        )

    # 3) 长句占比突变
    long_pcts = [m["long_sent_pct"] for _, m in section_data]
    if long_pcts:
        max_pct = max(long_pcts)
        min_pct = min(long_pcts)
        if max_pct - min_pct > 30:
            max_sec = next(s for s, m in section_data if m["long_sent_pct"] == max_pct)
            min_sec = next(s for s, m in section_data if m["long_sent_pct"] == min_pct)
            warnings.append(
                f"⚠️  复杂度差异: 长句占比从 {min_pct:.0f}% 到 {max_pct:.0f}% "
                f"(差 {max_pct-min_pct:.0f}%)\n"
                f"     简洁: 「{min_sec.title}」 vs 复杂: 「{max_sec.title}」"
            )

    # 4) 标点比例突变
    cp_ratios = [m["comma_period"] for _, m in section_data]
    if cp_ratios:
        max_cp = max(cp_ratios)
        min_cp = min(cp_ratios)
        if max_cp - min_cp > 1.5 and min_cp > 0:
            warnings.append(
                f"⚠️  标点节奏差异: 逗号:句号 比例从 {min_cp:.2f} 到 {max_cp:.2f}"
            )

    if not warnings:
        out.append("✓ 章节间风格基本一致,拼接预警风险较低")
    else:
        out.append(f"发现 {len(warnings)} 处拼接预警风险:")
        out.append("")
        for w in warnings:
            out.append(w)
            out.append("")
        out.append("规避方法:")
        out.append("  1. 全文人称统一(\"我们\" 或 \"课题组\",择一)")
        out.append("  2. 各章节平均句长差控制在 ±5 字内")
        out.append("  3. 低权重章节(背景/局限/致谢)也要轻度改写以拉齐风格")
        out.append("  4. 通读时主动调整句长,使其与相邻章节接近")

    return "\n".join(out)


def cmd_critique(text: str) -> str:
    """文献综述批判性注入候选位检测.

    在引述既有研究后未加批判评价的段落,提示植入注入点.
    """
    candidates = find_critique_candidates(text)
    if not candidates:
        return ("✓ 未发现需要批判性注入的段落\n"
                "(若你正在改写文献综述,确保每段引述后有1处方法/样本/时效批判)")

    out: list[str] = []
    out.append(f"发现 {len(candidates)} 处可植入批判性评价的段落\n")
    out.append("(在引述「研究表明/已有研究/学者认为」后添加批判,可降率15-30%)\n")
    out.append("-" * 70)
    for para_no, preview, hint in candidates[:30]:
        out.append(f"\n段{para_no}: {preview}")
        out.append(f"   建议批判: {hint}")

    out.append("\n" + "-" * 70)
    out.append("批判注入5种模板:")
    out.append("  1. 样本局限: (但样本仅限X,能否推广到Y不明)")
    out.append("  2. 方法缺陷: (该研究存在内生性问题,因果推断不稳健)")
    out.append("  3. 时效性: (数据截至X年,在数字化加速背景下适用性存疑)")
    out.append("  4. 假设偏差: (假定个体决策完全理性,忽略了行为偏差)")
    out.append("  5. 复现强度: (结论虽具启发性,但未在跨文化样本中复现)")
    out.append("\n注意: 不改既有陈述,只在每段引述后加 1 句即可")
    return "\n".join(out)


# ==================== CLI ====================

CMDS = {
    "locate": cmd_locate,
    "suggest": cmd_suggest,
    "split": cmd_split,
    "replace": cmd_replace,
    "chapters": cmd_chapters,
    "cohesion": cmd_cohesion,
    "critique": cmd_critique,
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AIGC改写辅助工具")
    parser.add_argument("command", choices=list(CMDS.keys()))
    parser.add_argument("path", help="文本文件路径,或 - 从stdin读取")
    args = parser.parse_args()

    if args.path == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.path).read_text(encoding="utf-8")

    print(CMDS[args.command](text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
