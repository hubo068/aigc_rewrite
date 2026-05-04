#!/usr/bin/env python3
"""
AIGC指纹分析工具
================

分析中文学术文本的AIGC检测风险点,无需任何外部依赖,仅依赖Python标准库。

用法:
    python analyze.py <文件路径>                  # 从文件读取
    python analyze.py - < input.txt               # 从stdin读取
    python analyze.py --json <文件>               # 输出JSON格式
    python analyze.py --diff <旧> <新>            # 对比改写前后变化
    python analyze.py --chapter 摘要 <文件>       # 章节加权分析(摘要×1.8)

输出:
    - 基础统计 (字数/句数/段数/平均句长)
    - 标点指纹 (逗号:句号比/破折号/括号)
    - 词汇多样性 (TTR/AI爱用词/俗语/模糊限定)
    - 主体性 (第一人称密度/判断句/情绪词)
    - 结构 (并列结构/思维跳跃/机械列举)
    - 学术指纹 (引用密度/统计数据/批判注入/学科术语)
    - 拼接预警 (段间句长波动)
    - 风险评分 (0-100, 越低越安全; 含章节加权)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ==================== 词汇库 ====================

AI_LOVE_WORDS = [
    "显著", "重要", "关键", "进一步", "系统", "机制", "全面", "深入",
    "持续", "切实", "充分", "总体", "整体", "有效", "广泛", "深刻",
    "良好", "明显", "积极", "高效", "优化", "提升", "推进", "加强",
    "完善", "构建", "形成", "实现", "促进", "推动", "助力", "赋能",
]

FIRST_PERSON = ["我们", "课题组", "笔者", "本研究", "本课题", "本人", "本团队"]

JUDGMENT_WORDS = [
    "认为", "判断", "指出", "值得注意的是", "值得一提的是",
    "需要强调的是", "我们倾向于", "倾向于", "看来", "估计",
]

EMOTION_WORDS = [
    "令人担忧", "值得欣慰", "坦率地说", "遗憾的是", "庆幸的是",
    "令人意外", "出乎意料", "颇为", "颇具", "可惜", "麻烦的是",
    "头疼", "棘手", "尴尬",
]

COLLOQUIAL_WORDS = [
    "扎下根来", "纸面规划", "无米之炊", "第一个吃螃蟹", "话说回来",
    "摸清楚", "跑通", "捋一捋", "卡脖子", "硬骨头", "底子",
    "拍脑袋", "一刀切", "走过场", "见招拆招",
]

HEDGE_WORDS = [
    "某种程度上", "大致上", "粗略", "粗略判断", "粗略估计",
    "大概", "大约", "约", "左右", "上下", "前后", "或许",
    "可能", "未必", "不一定", "至少在", "仅就", "就...而言",
]

PARALLEL_PATTERNS = [
    "一是", "二是", "三是", "四是", "五是",
    "首先", "其次", "再次", "最后", "另外",
    "第一", "第二", "第三", "第四", "第五",
    "(1)", "(2)", "(3)", "(4)",
    "(一)", "(二)", "(三)", "(四)",
]

THINK_JUMP_WORDS = [
    "话说回来", "回到前文", "与此相关但", "不过", "然而",
    "这个判断", "可能过于", "一度倾向于", "但调研后发现",
]

# 学科特定术语(植入这些可降率)
DOMAIN_TERMS = [
    # 社会学
    "场域", "惯习", "文化资本", "社会资本", "符号暴力", "布迪厄",
    # 经济学
    "帕累托改进", "纳什均衡", "委托代理", "外部性", "信息不对称",
    # 教育学
    "脚手架", "最近发展区", "元认知", "认知负荷",
    # 心理学
    "自我效能感", "习得性无助", "心智模型",
    # 管理学
    "动态能力", "组织惯例", "嵌入性",
]

# 批判性注入标记词
CRITIQUE_WORDS = [
    "样本仅限", "样本局限", "推广性", "外部效度",
    "内生性", "因果不稳健", "选择偏差", "测量误差",
    "数据截至", "时效性", "适用性存疑", "在数字化背景下",
    "假定", "忽略了", "未考虑", "理论假设过强",
    "未在", "未能复现", "跨文化", "边界条件",
    "需谨慎推广", "结论的强度", "证据强度",
]

# "超出预期"评价词
SURPRISE_WORDS = [
    "超出预期", "出乎意料", "令人意外", "高于预期", "低于预期",
    "未能完全复现", "提示理论边界", "需要指出的是",
    "反常识", "与常识相反", "值得玩味",
]

# 章节权重映射
CHAPTER_WEIGHTS: dict[str, float] = {
    "摘要": 1.8, "abstract": 1.8,
    "引言": 1.5, "前言": 1.5, "导论": 1.5, "绪论": 1.5, "introduction": 1.5,
    "结论": 1.5, "conclusion": 1.5, "结语": 1.5,
    "创新": 1.5, "贡献": 1.5,
    "文献综述": 1.0, "综述": 1.0, "literature": 1.0,
    "研究设计": 1.0, "方法": 1.0, "method": 1.0,
    "结果": 1.0, "result": 1.0, "数据分析": 1.0, "实证": 1.0,
    "讨论": 1.0, "discussion": 1.0,
    "背景": 0.7, "理论基础": 0.7,
    "局限": 0.5, "不足": 0.5, "limitation": 0.5,
    "致谢": 0.5, "acknowledgment": 0.5,
    "正文": 1.0,
}

# 引用格式正则: [1], [12], [N], (Smith, 2020), (Smith et al., 2020)
CITATION_PATTERN = re.compile(
    r"\[(\d+(?:[-,，]\s*\d+)*)\]"  # [1], [1,2], [1-3]
    r"|\(([A-Z][a-zA-Z\s,\.&]+,\s*\d{4}[a-z]?)\)"  # (Smith, 2020)
    r"|\(([一-龥]{2,8},?\s*\d{4})\)"  # (李明, 2020)
)

# 统计量正则: β=0.47, p<0.01, p=0.03, R²=0.62, n=384, t=2.34, F=12.4, df=382
# 注: 不含 ²/R² 等特殊符号匹配,需在调用前先做替换
STAT_PATTERN = re.compile(
    r"\b[βαγδρλ]\s*=\s*-?\d+\.?\d*"  # β=0.47, α=0.05
    r"|\bp\s*[<>=]\s*\d?\.\d+"  # p<0.01, p=0.03
    r"|\bR[²2]\s*=\s*\d?\.\d+"  # R²=0.62
    r"|\b[ntFdf]+\s*=\s*\d+\.?\d*"  # n=384, t=2.34, F=12.4
    r"|\bχ[²2]\s*=\s*\d+\.?\d*"  # χ²=
    r"|\b\d+\.?\d*\s*%"  # 23.4%
    r"|\bCI\s*\[?[\d.,\s\-]+\]?",  # CI [0.21, 0.73]
    re.IGNORECASE,
)

# 机械列举模式(强警告)
MECH_LIST_PATTERN = re.compile(
    r"(?:^|[\n。!?])"
    r"\s*(?:第[一二三四五六七八九十]|"
    r"一是|二是|三是|四是|五是|"
    r"首先|其次|再次|然后|最后|另外|"
    r"\(\d\)|（\d）|"
    r"\(?\d+[\)）.、])"
)

HUMAN_FILL_PATTERN = re.compile(r"【人工补充[:：][^】]*】")
SENT_END_PATTERN = re.compile(r"[。！？!?；;]")


# ==================== 数据结构 ====================

@dataclass
class Metrics:
    total_chars: int = 0
    total_paragraphs: int = 0
    total_sentences: int = 0
    avg_sent_len: float = 0.0
    short_sent_count: int = 0  # ≤15
    long_sent_count: int = 0  # ≥40
    sent_len_std: float = 0.0
    comma_count: int = 0
    period_count: int = 0
    comma_period_ratio: float = 0.0
    bracket_count: int = 0
    dash_count: int = 0
    semicolon_count: int = 0
    bigram_ttr: float = 0.0
    ai_love_count: int = 0
    first_person_count: int = 0
    first_person_density_chars_per: float = 0.0
    judgment_count: int = 0
    emotion_count: int = 0
    colloquial_count: int = 0
    hedge_count: int = 0
    parallel_count: int = 0
    mech_list_count: int = 0  # 机械列举(2026新增,严重警告)
    think_jump_count: int = 0
    human_fill_marks: int = 0
    # 2026新增: 学术指纹
    citation_count: int = 0  # 引用次数
    citation_density: float = 0.0  # 千字引用密度
    stat_data_count: int = 0  # 统计数据(β/p/R²/n/F等)
    domain_term_count: int = 0  # 学科术语
    critique_count: int = 0  # 批判性注入
    surprise_count: int = 0  # "超出预期"评价
    # 2026新增: 拼接预警
    paragraph_len_std: float = 0.0  # 段落字数标准差
    para_avg_sent_lens: list[float] = field(default_factory=list)  # 各段平均句长
    splice_warning: bool = False
    # 章节加权
    chapter: str = ""
    chapter_weight: float = 1.0
    weighted_risk_score: int = 0
    # 风险输出
    risk_score: int = 0
    risks: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# ==================== 分析函数 ====================

def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句"""
    parts = SENT_END_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def count_occurrences(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def calc_bigram_ttr(text: str) -> float:
    """字符二元组的TTR(类型比例),粗略反映词汇多样性"""
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < 2:
        return 0.0
    bigrams = [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def calc_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def analyze(text: str, chapter: str = "") -> Metrics:
    m = Metrics()
    text_no_marks = HUMAN_FILL_PATTERN.sub("", text)
    sentences = split_sentences(text_no_marks)
    paragraphs = split_paragraphs(text_no_marks)

    m.total_chars = len(re.sub(r"\s+", "", text_no_marks))
    m.total_sentences = len(sentences)
    m.total_paragraphs = len(paragraphs)

    # 章节加权
    m.chapter = chapter
    m.chapter_weight = get_chapter_weight(chapter) if chapter else 1.0

    sent_lens = [len(s) for s in sentences]
    if sent_lens:
        m.avg_sent_len = sum(sent_lens) / len(sent_lens)
        m.short_sent_count = sum(1 for l in sent_lens if l <= 15)
        m.long_sent_count = sum(1 for l in sent_lens if l >= 40)
        m.sent_len_std = calc_std(sent_lens)

    # 段落级句长 (用于拼接预警检测)
    para_avg_lens: list[float] = []
    para_lens: list[int] = []
    for p in paragraphs:
        p_sents = split_sentences(p)
        p_sent_lens = [len(s) for s in p_sents]
        if p_sent_lens:
            para_avg_lens.append(sum(p_sent_lens) / len(p_sent_lens))
        para_lens.append(len(p))
    m.para_avg_sent_lens = para_avg_lens
    m.paragraph_len_std = calc_std([float(l) for l in para_lens])

    # 拼接预警: 段落间平均句长差异 > 10字
    if para_avg_lens and len(para_avg_lens) >= 2:
        if max(para_avg_lens) - min(para_avg_lens) > 12:
            m.splice_warning = True

    # 注意区分全角(中文)与半角(英文)标点
    m.comma_count = sum(text_no_marks.count(c) for c in "，,")  # ,和,
    m.period_count = sum(text_no_marks.count(c) for c in "。！？.!?")  # 。！?
    m.comma_period_ratio = (m.comma_count / m.period_count) if m.period_count else 0.0

    m.bracket_count = sum(text_no_marks.count(c) for c in "（(")  # (和(
    m.dash_count = text_no_marks.count("——") + len(re.findall(r"--+", text_no_marks))
    m.semicolon_count = sum(text_no_marks.count(c) for c in "；;")  # ;和;

    m.bigram_ttr = calc_bigram_ttr(text_no_marks)

    m.ai_love_count = count_occurrences(text_no_marks, AI_LOVE_WORDS)
    m.first_person_count = count_occurrences(text_no_marks, FIRST_PERSON)
    if m.first_person_count and m.total_chars:
        m.first_person_density_chars_per = m.total_chars / m.first_person_count
    m.judgment_count = count_occurrences(text_no_marks, JUDGMENT_WORDS)
    m.emotion_count = count_occurrences(text_no_marks, EMOTION_WORDS)
    m.colloquial_count = count_occurrences(text_no_marks, COLLOQUIAL_WORDS)
    m.hedge_count = count_occurrences(text_no_marks, HEDGE_WORDS)
    m.parallel_count = count_occurrences(text_no_marks, PARALLEL_PATTERNS)
    m.think_jump_count = count_occurrences(text_no_marks, THINK_JUMP_WORDS)
    m.human_fill_marks = len(HUMAN_FILL_PATTERN.findall(text))

    # 2026新增: 学术指纹
    m.citation_count = len(CITATION_PATTERN.findall(text_no_marks))
    if m.total_chars:
        m.citation_density = m.citation_count * 1000 / m.total_chars
    m.stat_data_count = len(STAT_PATTERN.findall(text_no_marks))
    m.domain_term_count = count_occurrences(text_no_marks, DOMAIN_TERMS)
    m.critique_count = count_occurrences(text_no_marks, CRITIQUE_WORDS)
    m.surprise_count = count_occurrences(text_no_marks, SURPRISE_WORDS)
    # 机械列举: 严重AI指纹
    m.mech_list_count = len(MECH_LIST_PATTERN.findall(text_no_marks))

    _evaluate_risks(m)
    return m


def get_chapter_weight(chapter: str) -> float:
    """根据章节名查询权重系数"""
    if not chapter:
        return 1.0
    chapter_lower = chapter.lower()
    for kw, w in CHAPTER_WEIGHTS.items():
        if kw in chapter or kw.lower() in chapter_lower:
            return w
    return 1.0


def _evaluate_risks(m: Metrics) -> None:
    """根据指标输出风险点和加分项,计算风险分(0-100,越高越像AI)"""
    score = 0
    risks: list[str] = []
    sugs: list[str] = []

    # 句法 (30%)
    if m.avg_sent_len > 40:
        score += 15
        risks.append(f"⚠️  平均句长{m.avg_sent_len:.1f}字过长 (>40),需拆句")
        sugs.append("将含 3 个以上逗号的句子拆为 2-3 句")
    elif m.avg_sent_len < 20:
        score += 5
        risks.append(f"⚠️  平均句长{m.avg_sent_len:.1f}字过短 (<20)")
    else:
        sugs.append(f"✓ 平均句长{m.avg_sent_len:.1f}字 在合理区间")

    if m.total_sentences and m.short_sent_count == 0:
        score += 8
        risks.append("⚠️  无短句(≤15字),AI痕迹明显")
        sugs.append("在每段插入 1 个 ≤15 字的短句作为节奏点")
    if m.total_sentences and m.long_sent_count / max(m.total_sentences, 1) > 0.5:
        score += 7
        risks.append(f"⚠️  长句(≥40字)占比{m.long_sent_count / m.total_sentences:.0%}过高")

    # 标点 (10%)
    if m.comma_period_ratio > 3:
        score += 8
        risks.append(f"⚠️  逗号:句号={m.comma_period_ratio:.2f}:1,典型AI指纹 (目标 2-2.5)")
        sugs.append("将长句中段的逗号改为句号,切短句子")
    elif m.comma_period_ratio < 1.5:
        score += 3
        risks.append(f"⚠️  逗号:句号={m.comma_period_ratio:.2f}:1偏低,缺乏停顿层次")
    else:
        sugs.append(f"✓ 逗号:句号={m.comma_period_ratio:.2f}:1 合理")

    if m.bracket_count == 0:
        score += 3
        risks.append("⚠️  无括号补充,缺乏离题感")
        sugs.append("加入 1 处括号备注,如(虽然台风季有点头疼)")
    if m.dash_count == 0:
        score += 2
        risks.append("⚠️  无破折号(——),无呼吸停顿")

    # 主体性 (10%)
    if m.first_person_count == 0:
        score += 12
        risks.append("⚠️  零第一人称,典型AI痕迹")
        sugs.append("每 300 字植入 1 次「课题组/我们/笔者」")
    elif m.first_person_density_chars_per > 500:
        score += 5
        risks.append(f"⚠️  第一人称密度低,每{m.first_person_density_chars_per:.0f}字才出现1次")
    else:
        sugs.append(f"✓ 第一人称密度合理 (每{m.first_person_density_chars_per:.0f}字1次)")

    if m.judgment_count == 0:
        score += 5
        risks.append("⚠️  无判断句,缺乏作者立场")
        sugs.append("加入「我们认为」「值得注意的是」等判断句")
    if m.emotion_count == 0:
        score += 4
        risks.append("⚠️  无情绪/评价词,过于客观,反像AI")
        sugs.append("每 500 字植入 1 个评价词:令人担忧/坦率地说/麻烦的是")

    # 词汇 (15%)
    if m.ai_love_count > max(m.total_chars / 100, 5):
        score += 10
        risks.append(f"⚠️  AI爱用词出现{m.ai_love_count}次,过多")
        sugs.append("替换:显著→大致/重要→值得一提/进一步→接着/系统→整套")
    if m.colloquial_count == 0:
        score += 6
        risks.append("⚠️  无俗语/口语词")
        sugs.append("植入1个:扎下根来/纸面规划/无米之炊/话说回来")
    if m.hedge_count == 0:
        score += 5
        risks.append("⚠️  无模糊限定词")
        sugs.append("加入:某种程度上/粗略判断/至少在...语境下")

    # 结构 (25%)
    if m.parallel_count > 3:
        score += 10
        risks.append(f"⚠️  并列结构{m.parallel_count}处过多 (一是/二是/首先/其次)")
        sugs.append("将「一是...二是...三是...」改为段落式叙述")
    if m.think_jump_count == 0:
        score += 6
        risks.append("⚠️  无思维跳跃词")
        sugs.append("插入:话说回来/回到前文/与此相关但常被忽略")

    # 2026新增: 机械列举(严重AI指纹,可达85%+)
    if m.mech_list_count > 5:
        score += 12
        risks.append(f"❌ 机械列举{m.mech_list_count}处,维普识别为AI模式(率可达85%+)")
        sugs.append("将「第一/第二/第三」全部改为段落式叙述")
    elif m.mech_list_count > 2:
        score += 5
        risks.append(f"⚠️  机械列举{m.mech_list_count}处偏多")

    # 2026新增: 拼接预警
    if m.splice_warning:
        score += 8
        diff = max(m.para_avg_sent_lens) - min(m.para_avg_sent_lens) if m.para_avg_sent_lens else 0
        risks.append(f"⚠️  拼接预警: 段落间平均句长差{diff:.1f}字 (>12,易触发)")
        sugs.append("拉齐各段句长,差异控制在 ±5 字内")

    # 2026新增: 学术指纹(章节相关)
    chapter_lower = m.chapter.lower() if m.chapter else ""
    is_review = any(kw in chapter_lower for kw in ["综述", "文献", "literature", "review"])
    is_result = any(kw in chapter_lower for kw in ["结果", "数据分析", "result", "实证"])
    is_intro_concl = any(kw in chapter_lower for kw in ["摘要", "引言", "结论", "abstract", "introduction", "conclusion"])

    # 引用密度(综述/引言权重高)
    if is_review or is_intro_concl:
        if m.citation_density < 3:
            score += 8
            risks.append(f"⚠️  引用密度{m.citation_density:.1f}/千字过低 (目标 ≥3)")
            sugs.append("综述/引言部分必须保留 [N] 或 (作者,年份) 引用,无引用 +10-15%")

    # 统计数据(结果/结论部分权重高)
    if is_result or is_intro_concl:
        if m.stat_data_count < 3:
            score += 6
            risks.append(f"⚠️  统计数据{m.stat_data_count}处偏少 (目标 ≥3)")
            sugs.append("主动加 β/p/R²/n 等数据,维普对数据段落容忍度高(15-25%)")
        else:
            sugs.append(f"✓ 统计数据{m.stat_data_count}处充足")

    # 批判性注入(综述部分要求高)
    if is_review:
        if m.critique_count < 3:
            score += 7
            risks.append(f"⚠️  批判注入{m.critique_count}处过少 (综述应 ≥3)")
            sugs.append("每段引述后加方法/样本/时效批判,可降率从65%→35%")

    # 学科术语
    if m.domain_term_count == 0 and is_intro_concl:
        score += 3
        sugs.append("摘要/引言可植入1-2个学科术语(场域/帕累托/最近发展区)")

    # 超出预期评价(摘要/结论)
    if is_intro_concl and m.surprise_count == 0:
        score += 4
        risks.append("⚠️  无「超出预期」评价")
        sugs.append("摘要/结论结尾加: 「结果令人意外」「需要指出的是」「未能完全复现」")

    # 加分项: 人工补充标记
    if m.human_fill_marks == 0:
        sugs.append("ℹ️  尚未植入【人工补充】标记,提交前需要人工注血")
    else:
        sugs.append(f"✓ 检测到{m.human_fill_marks}处【人工补充】标记")

    m.risk_score = max(0, min(100, score))
    # 加权风险: 章节权重越高,风险越被放大
    m.weighted_risk_score = min(100, int(m.risk_score * m.chapter_weight))
    m.risks = risks
    m.suggestions = sugs


# ==================== 输出格式 ====================

def format_report(m: Metrics) -> str:
    lines: list[str] = []
    bar = "=" * 56
    lines.append(bar)
    lines.append("           AIGC指纹分析报告")
    lines.append(bar)

    if m.chapter:
        lines.append(f"\n【章节信息】")
        lines.append(f"  章节类型:           {m.chapter}")
        lines.append(f"  权重系数:           ×{m.chapter_weight:.1f}")
        weight_kind = "高权重(必须深度改80%)" if m.chapter_weight >= 1.5 else (
            "低权重(轻度改30%即可,但需拉齐风格)" if m.chapter_weight < 1.0 else "中权重(标准改60%)"
        )
        lines.append(f"  策略:               {weight_kind}")

    lines.append("\n【基础统计】")
    lines.append(f"  总字数(去标记):     {m.total_chars}")
    lines.append(f"  段落数:             {m.total_paragraphs}")
    lines.append(f"  句子数:             {m.total_sentences}")
    lines.append(f"  平均句长:           {m.avg_sent_len:.1f} 字  (目标 25-35)")
    lines.append(f"  句长标准差:         {m.sent_len_std:.1f}    (越大越自然)")
    lines.append(f"  短句(≤15字):        {m.short_sent_count} 句")
    lines.append(f"  长句(≥40字):        {m.long_sent_count} 句")

    lines.append("\n【标点指纹】")
    lines.append(f"  逗号数:             {m.comma_count}")
    lines.append(f"  句号/!?数:          {m.period_count}")
    lines.append(f"  逗号:句号 比例:     {m.comma_period_ratio:.2f}:1  (目标 2-2.5,AI常达3+)")
    lines.append(f"  括号(中/英文):      {m.bracket_count}")
    lines.append(f"  破折号(——):         {m.dash_count}")
    lines.append(f"  分号:               {m.semicolon_count}")

    lines.append("\n【词汇多样性】")
    lines.append(f"  字符bigram TTR:     {m.bigram_ttr:.3f}")
    lines.append(f"  AI爱用词出现:       {m.ai_love_count} 次   (目标 ≤ 总字数/100)")
    lines.append(f"  俗语/口语词:        {m.colloquial_count} 次   (目标 ≥1)")
    lines.append(f"  模糊限定词:         {m.hedge_count} 次   (目标 ≥2)")
    lines.append(f"  学科术语:           {m.domain_term_count} 次   (摘要建议 ≥1)")

    lines.append("\n【主体性指纹】")
    lines.append(f"  第一人称次数:       {m.first_person_count}")
    if m.first_person_density_chars_per:
        lines.append(f"  第一人称密度:       每 {m.first_person_density_chars_per:.0f} 字 1 次  (目标 ≤300)")
    lines.append(f"  判断句:             {m.judgment_count} 次")
    lines.append(f"  情绪/评价词:        {m.emotion_count} 次")
    lines.append(f"  「超出预期」评价:   {m.surprise_count} 次")

    lines.append("\n【结构指纹】")
    lines.append(f"  并列结构(一是/首先): {m.parallel_count} 处  (目标 ≤3)")
    lines.append(f"  机械列举(第一/2./等): {m.mech_list_count} 处  (>5 时维普可达85%+)")
    lines.append(f"  思维跳跃词:         {m.think_jump_count} 处  (目标 ≥1)")
    lines.append(f"  【人工补充】标记:    {m.human_fill_marks} 处")

    lines.append("\n【学术指纹(2026)】")
    lines.append(f"  引用次数:           {m.citation_count} 次")
    lines.append(f"  引用密度:           {m.citation_density:.1f}/千字  (综述 ≥3, 无引用 +10-15%)")
    lines.append(f"  统计数据(β/p/R²/n): {m.stat_data_count} 处  (结果章建议 ≥3, 容忍度高)")
    lines.append(f"  批判性注入:         {m.critique_count} 处  (综述建议 ≥3)")

    lines.append("\n【拼接预警】")
    if m.para_avg_sent_lens and len(m.para_avg_sent_lens) >= 2:
        diff = max(m.para_avg_sent_lens) - min(m.para_avg_sent_lens)
        lines.append(f"  段落间最大句长差:   {diff:.1f} 字  (目标 ≤12)")
    lines.append(f"  段落字数标准差:     {m.paragraph_len_std:.1f}")
    if m.splice_warning:
        lines.append("  ⚠️  风险: 跨章节风格不统一,易触发拼接预警")
    else:
        lines.append("  ✓ 段落风格基本一致")

    lines.append("\n【风险评分】")
    score_bar = "█" * (m.risk_score // 5) + "░" * (20 - m.risk_score // 5)
    lines.append(f"  AI风险:           {m.risk_score:3d}/100  [{score_bar}]")
    if m.chapter and m.chapter_weight != 1.0:
        wbar = "█" * (m.weighted_risk_score // 5) + "░" * (20 - m.weighted_risk_score // 5)
        lines.append(f"  加权风险(×{m.chapter_weight:.1f}):  {m.weighted_risk_score:3d}/100  [{wbar}]")
    if m.risk_score >= 70:
        lines.append("  ❌ 高风险: 维普AIGC率预估 60%+,需深度改写")
    elif m.risk_score >= 40:
        lines.append("  ⚠️  中风险: 维普AIGC率预估 35-60%,需继续注血")
    elif m.risk_score >= 20:
        lines.append("  ✓ 低风险: 维普AIGC率预估 25-35%,目标区间")
    else:
        lines.append("  ✓ 极低: AIGC率<25%,可直接提交")

    if m.risks:
        lines.append("\n【风险点】")
        lines.extend(f"  {r}" for r in m.risks)

    if m.suggestions:
        lines.append("\n【改写建议】")
        lines.extend(f"  {s}" for s in m.suggestions)

    lines.append("\n" + bar)
    return "\n".join(lines)


def diff_report(old_m: Metrics, new_m: Metrics) -> str:
    lines: list[str] = []
    bar = "=" * 56
    lines.append(bar)
    lines.append("           改写前后对比")
    lines.append(bar)

    def cmp(label: str, old: Any, new: Any, fmt: str = "{:.2f}") -> str:
        if isinstance(old, (int, float)):
            delta = new - old
            arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
            return f"  {label:20s} {fmt.format(old):>8s} → {fmt.format(new):>8s} ({arrow}{abs(delta):.1f})"
        return f"  {label:20s} {old} → {new}"

    lines.append(cmp("AI风险分", old_m.risk_score, new_m.risk_score, "{:.0f}"))
    lines.append(cmp("平均句长", old_m.avg_sent_len, new_m.avg_sent_len))
    lines.append(cmp("逗号:句号", old_m.comma_period_ratio, new_m.comma_period_ratio))
    lines.append(cmp("AI爱用词", old_m.ai_love_count, new_m.ai_love_count, "{:.0f}"))
    lines.append(cmp("第一人称", old_m.first_person_count, new_m.first_person_count, "{:.0f}"))
    lines.append(cmp("俗语词", old_m.colloquial_count, new_m.colloquial_count, "{:.0f}"))
    lines.append(cmp("模糊限定", old_m.hedge_count, new_m.hedge_count, "{:.0f}"))
    lines.append(cmp("并列结构", old_m.parallel_count, new_m.parallel_count, "{:.0f}"))
    lines.append(cmp("机械列举", old_m.mech_list_count, new_m.mech_list_count, "{:.0f}"))
    lines.append(cmp("引用次数", old_m.citation_count, new_m.citation_count, "{:.0f}"))
    lines.append(cmp("统计数据", old_m.stat_data_count, new_m.stat_data_count, "{:.0f}"))
    lines.append(cmp("批判注入", old_m.critique_count, new_m.critique_count, "{:.0f}"))
    lines.append(cmp("【人工补充】", old_m.human_fill_marks, new_m.human_fill_marks, "{:.0f}"))

    if old_m.risk_score - new_m.risk_score >= 20:
        lines.append("\n✓ 风险显著降低 (-20+),改写有效")
    elif old_m.risk_score - new_m.risk_score >= 10:
        lines.append("\n✓ 风险有所降低,继续优化")
    else:
        lines.append("\n⚠️  风险下降不明显,需要更激进改写")

    lines.append(bar)
    return "\n".join(lines)


# ==================== CLI ====================

def read_text(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


def main() -> int:
    # 强制stdout为UTF-8,避免Windows GBK编码失败
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="AIGC指纹分析工具(中文学术文本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", help="文本文件路径,或 - 从stdin读取")
    parser.add_argument("path2", nargs="?", help="对比模式下的第二个文件")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出指标")
    parser.add_argument("--diff", action="store_true", help="对比两个文件 (path 旧,path2 新)")
    parser.add_argument(
        "--chapter",
        default="",
        help="章节类型 (摘要/引言/结论/综述/方法/结果/背景/局限/致谢),用于章节加权风险评分",
    )
    args = parser.parse_args()

    if not args.path:
        parser.print_help()
        return 1

    if args.diff:
        if not args.path2:
            print("错误: --diff 需要两个文件参数 (旧 新)", file=sys.stderr)
            return 1
        old_m = analyze(read_text(args.path), chapter=args.chapter)
        new_m = analyze(read_text(args.path2), chapter=args.chapter)
        if args.json:
            print(json.dumps(
                {"old": asdict(old_m), "new": asdict(new_m)},
                ensure_ascii=False, indent=2,
            ))
        else:
            print(format_report(old_m))
            print()
            print(format_report(new_m))
            print()
            print(diff_report(old_m, new_m))
        return 0

    text = read_text(args.path)
    metrics = analyze(text, chapter=args.chapter)
    if args.json:
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    else:
        print(format_report(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
