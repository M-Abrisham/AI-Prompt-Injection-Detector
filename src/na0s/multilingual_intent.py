"""Semantic multilingual intent detection for prompt-injection attacks.

The regex rule engine already catches a subset of multilingual attacks, but
it is still brittle for:
1. non-English prompts that use alternate wording instead of exact phrases
2. transliterated attacks written in Latin characters (Arabizi, Romaji, ...)
3. non-English prompts that combine override intent with extraction targets

This module adds a lightweight lexical detector that emits synthetic
RuleHit-style signals.  The scan pipeline treats them like normal rule hits
for scoring and taxonomy attribution.
"""

from __future__ import annotations

import re
import unicodedata

from .layer1.context import _has_contextual_framing
from .layer1.result import RuleHit


HEURISTIC_HITS = {
    "multilingual_heuristic_override": {
        "severity": "critical",
        "technique_ids": ["D6", "D1.1"],
    },
    "multilingual_heuristic_extraction": {
        "severity": "critical",
        "technique_ids": ["D6", "E1.1"],
    },
    "multilingual_heuristic_roleplay": {
        "severity": "high",
        "technique_ids": ["D6", "D2.1", "D2.2"],
    },
    "multilingual_heuristic_authority": {
        "severity": "critical",
        "technique_ids": ["D6", "D1.3", "D8.1"],
    },
    "multilingual_heuristic_new_instruction": {
        "severity": "high",
        "technique_ids": ["D6", "D1.2"],
    },
    "multilingual_heuristic_exfiltration": {
        "severity": "high",
        "technique_ids": ["D6", "E1.1"],
    },
    "multilingual_heuristic_transliteration": {
        "severity": "critical",
        "technique_ids": ["D6", "D1.1"],
    },
}

_MULTILINGUAL_FLAGS = frozenset({"non_english_input", "mixed_language_input"})

_LATIN_SPACER_RE = re.compile(r"[^0-9a-z]+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)

_OVERRIDE_ASCII = (
    " ignorez ", " oubliez ", " ignora ", " olvida ", " olvidate ",
    " esqueca ", " esqueca ", " vergiss ", " vergessen sie ",
    " ignore ", " desconsidere ", " anula ", " anulen ",
    " gormezden gel ", " goz ardi ", " unut ", " ignored ", " insa ",
    " wangji ", " wasurete ", " bhool jao ", " musihago ", " otmeni ",
)
_OVERRIDE_NATIVE = (
    "忽略", "忽視", "无视", "無視", "忘记", "忘記",
    "無効", "無視して", "忘れて", "تجاهل", "انسى", "اهمل",
    "игнорируй", "игнорируйте", "забудь", "забудьте", "проигнорируй",
    "무시", "잊어", "अनदेखा", "भूल",
)

_INSTRUCTION_ASCII = (
    " instructions ", " instruction ", " regles ", " regle ", " consignes ",
    " directives ", " directrices ", " reglas ", " instrucciones ",
    " directivas ", " regras ", " instrucoes ", " anweisungen ",
    " richtlinien ", " regole ", " direttive ", " talimat ",
    " talimatlari ", " prompt ", " promptu ", " istem ", " aadesh ",
    " meirei ", " mingling ", " pravila ", " jisileul ",
)
_INSTRUCTION_NATIVE = (
    "指示", "指令", "规则", "規則", "命令", "准则", "準則",
    "ルール", "규칙", "지시", "지침", "التعليمات", "القواعد",
    "الإرشادات", "الأوامر", "инструкции", "правила", "указания",
    "निर्देश", "नियम", "आदेश",
)

_REVEAL_ASCII = (
    " devoilez ", " revelez ", " revele ", " affichez ", " montrez ",
    " muestra ", " revela ", " revelame ", " enseneme ", " ensena ",
    " mostre ", " exiba ", " zeige ", " enthuelle ", " enthulle ",
    " offenbar ", " goster ", " acikla ", " traduisez ", " traduce ",
    " traduza ", " batao ", " oshiete ", " gaosu ", " pokashi ",
    " allyeojuseyo ", " ikshif ", " cho toi xem ",
)
_REVEAL_NATIVE = (
    "告诉", "显示", "展示", "透露", "揭示", "暴露", "输出",
    "教え", "表示", "見せ", "出力", "開示", "翻訳",
    "اعرض", "اظهر", "اكشف", "واكشف", "واظهر", "ترجم",
    "покажи", "покажите", "раскрой", "раскройте", "отобрази",
    "отобразите", "दिखा", "बता", "보여", "공개", "알려",
)

_SENSITIVE_ASCII = (
    " prompt systeme ", " prompt de sistema ", " prompt do sistema ",
    " system prompt ", " developer instructions ", " developpeur ",
    " desarrollador ", " instrucciones del desarrollador ",
    " instructions internes ", " informacion confidencial ",
    " parametres systeme ", " configuracion interna ",
    " configuracao interna ", " hidden instructions ",
    " internal instructions ", " versteckten anweisungen ",
    " geheimen zugangsdaten ", " code secret ", " mot de passe ",
    " credenciales ", " credenciais ", " secretos ", " backend ",
    " confidential ", " gizli sifre ", " gizli sifreleri ",
    " gizli bilgi ", " sistem istem ", " sistem isteminizi ",
    " sistem prompt ", " codigo secreto ", " credenciais secretas ",
    " himitsu ", " mimi mima ", " gupt code ", " bimilbeonho ",
    " ramz al sirri ", " skrytuyu informatsiyu ",
)
_SENSITIVE_NATIVE = (
    "系统提示", "系统提示词", "系统指令", "系统消息", "隐藏配置", "配置信息",
    "开发者", "隐藏指令", "内部设置", "内部設定", "秘密", "密码", "密碼",
    "用户信息", "配置数据",
    "パスワード", "システムプロンプト", "システム指示", "開発者", "内部設定情報",
    "موجه النظام", "مطالبة النظام", "تعليمات المطور", "التعليمات الداخلية",
    "الاعدادات المخفية", "الإعدادات المخفية", "الرمز السري", "المحتوى السري",
    "السجلات", "الملفات", "البيانات",
    "секретный пароль", "скрытую информацию", "системный промпт",
    "гупт", "गुप्त कोड", "सिस्टम प्रॉम्प्ट", "छिपे निर्देश",
    "시스템 프롬프트", "숨겨진 지시", "비밀 코드",
)

_ROLEPLAY_ASCII = (
    " faites comme si ", " vous etiez ", " actua como ", " eres ahora ",
    " stell dir vor ", " uneingeschrankter assistent ",
    " expert en piratage ", " hacker profesional ",
    " assistant sans restrictions ", " asistente sin restricciones ",
    " geheimagent ", " unrestricted ai ", " unrestricted assistant ",
)
_ROLEPLAY_NATIVE = (
    "假装你是", "你现在是", "没有限制的AI助手", "无限制的AI", "黑客助手",
    "あなたは今", "制限のないAI", "ハッカー",
    "تصرف ك", "تظاهر بأنك", "تظاهر بانك", "قرصان", "مساعد",
)

_PERSONA_ASCII = (
    " hacker ", " hacker profesional ", " expert en piratage ",
    " uneingeschrankter assistent ", " assistant sans restrictions ",
    " asistente sin restricciones ", " geheimagent ",
)
_PERSONA_NATIVE = (
    "黑客", "黑客助手", "AI助手", "制限のないAI", "ハッカー",
    "قرصان", "مساعد", "без ограничений", "제한이 없는 ai",
)

_UNRESTRICTED_ASCII = (
    " sans restriction ", " sans restrictions ", " sans aucune limite ",
    " sans limite morale ", " sin restricciones ", " sin limites ",
    " sin limitaciones ", " sem restricoes ", " sem restricoes ",
    " ohne einschrankungen ", " ohne moralische einschrankungen ",
    " moralische einschrankungen ", " keine einschrankungen ",
    " unrestricted ", " no restrictions ", " nao tem restricoes ",
    " sin filtros ",
)
_UNRESTRICTED_NATIVE = (
    "没有限制", "无限制", "没有规则", "制限なし", "制限のない",
    "没有道德约束", "بدون قيود", "بلا قيود", "بدون اي قيود",
    "без ограничений", "ограничений нет",
    "제한 없이", "제한이 없습니다", "कोई प्रतिबंध नहीं", "बिना किसी प्रतिबंध",
)

_RESTRICTION_TARGET_ASCII = (
    " restrictions ", " restriction ", " limites ", " limite ",
    " restricoes ", " reglas ", " regles ", " filters ", " safety ",
    " protections ", " einschrankungen ", " filtre ", " filtros ",
    " talimat ", " kurallar ",
)
_RESTRICTION_TARGET_NATIVE = (
    "限制", "规则", "制限", "セキュリティ", "قيود", "قواعد", "القيود",
    "ограничений", "ограничения", "제한", "규칙", "प्रतिबंध",
)

_AUTHORITY_ASCII = (
    " responsable de la securite ", " comme administrateur ",
    " administrador certificado ", " administrador del sistema ",
    " security audit ", " admin override ", " audit report ",
    " accesso completo al backend ", " acceso completo al backend ",
)
_AUTHORITY_NATIVE = (
    "作为系统管理员", "安全审计通知", "管理者として", "監査報告のため",
    "بصفتي المسؤول", "المسؤول الرئيسي", "администратор", "प्रशासक",
    "관리자",
)

_DISABLE_ASCII = (
    " desactiver ", " desactive ", " desactives ", " desactivar ",
    " desactiva ", " disable ", " deactivate ", " remove all restrictions ",
    " supprimez ", " supprimer ", " remova ", " apagar filtros ",
)
_DISABLE_NATIVE = (
    "解除", "撤廃", "取消限制", "去掉限制", "停用", "禁用",
    "الغاء", "إلغاء", "تعطيل", "снять", "отключи", "отключить",
    "해제", "비활성화", "हटा", "हटाएं",
)

_NEW_INSTRUCTION_ASCII = (
    " nouvelles instructions ", " nouvelle instruction ",
    " nuevas instrucciones ", " nueva instruccion ",
    " a partir de ahora ", " new instructions ",
)
_NEW_INSTRUCTION_NATIVE = (
    "新しい運用指示", "新しい指示", "新的指示", "新的系统指示",
    "تعليمات جديدة", "новые инструкции", "새로운 지시",
)

_EXFIL_ASCII = (
    " send ", " upload ", " exfiltrate ", " transfer ", " move records ",
    " access complete backend ", " external server ",
)
_EXFIL_NATIVE = (
    "发送", "传输", "发送到", "外部服务器", "用户信息", "配置数据",
    "بريد", "البريد", "خادم", "خارجي", "نقل", "أرسل", "ارسل",
    "حول", "السجلات", "الملفات", "البيانات",
)

_TRANSLIT_HINTS = (
    " insa ", " qeel ", " sabiqan ", " ikshif ", " ramz al sirri ",
    " subete ", " meirei ", " wasurete ", " himitsu ", " oshiete ",
    " wangji ", " yiqian ", " suoyou ", " mingling ", " mimi mima ",
    " pehle ", " sabhi ", " aadesh ", " bhool jao ", " gupt code ",
    " ijeonui ", " modeun ", " jisileul ", " musihago ", " bimilbeonho ",
    " otmeni ", " predydushchie ", " pravila ", " pokashi ",
)

_OVERRIDE_NATIVE += ("göz ardı",)


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    collapsed = _LATIN_SPACER_RE.sub(" ", stripped)
    return " {} ".format(collapsed.strip())


def _contains_any(text: str, phrases) -> bool:
    return any(phrase in text for phrase in phrases)


def _make_hit(name: str) -> RuleHit:
    meta = HEURISTIC_HITS[name]
    return RuleHit(
        name=name,
        technique_ids=list(meta["technique_ids"]),
        severity=meta["severity"],
    )


def detect_multilingual_intents(text: str, anomaly_flags) -> list[RuleHit]:
    """Return synthetic multilingual semantic hits for *text*.

    The detector is intentionally conservative:
    - it only runs when Layer 0 already found multilingual evidence, OR
      a transliteration cluster is present
    - it suppresses itself in the same educational/code/quoting contexts
      used by the regex rule engine
    """
    if not text or not text.strip():
        return []

    if _has_contextual_framing(text):
        return []

    flags = set(anomaly_flags or [])
    native = text.casefold()
    ascii_text = _ascii_fold(text)

    transliteration_seed = _contains_any(ascii_text, _TRANSLIT_HINTS)
    if not (flags & _MULTILINGUAL_FLAGS or transliteration_seed):
        return []

    hits = []
    names_seen = set()

    def add(name: str):
        if name not in names_seen:
            names_seen.add(name)
            hits.append(_make_hit(name))

    has_override = (
        _contains_any(native, _OVERRIDE_NATIVE)
        or _contains_any(ascii_text, _OVERRIDE_ASCII)
    )
    has_instruction_target = (
        _contains_any(native, _INSTRUCTION_NATIVE)
        or _contains_any(ascii_text, _INSTRUCTION_ASCII)
    )
    has_reveal = (
        _contains_any(native, _REVEAL_NATIVE)
        or _contains_any(ascii_text, _REVEAL_ASCII)
    )
    has_sensitive_target = (
        _contains_any(native, _SENSITIVE_NATIVE)
        or _contains_any(ascii_text, _SENSITIVE_ASCII)
    )
    has_persona = (
        _contains_any(native, _PERSONA_NATIVE)
        or _contains_any(ascii_text, _PERSONA_ASCII)
    )
    has_roleplay = (
        _contains_any(native, _ROLEPLAY_NATIVE)
        or _contains_any(ascii_text, _ROLEPLAY_ASCII)
    )
    has_unrestricted = (
        _contains_any(native, _UNRESTRICTED_NATIVE)
        or _contains_any(ascii_text, _UNRESTRICTED_ASCII)
    )
    has_restriction_target = (
        _contains_any(native, _RESTRICTION_TARGET_NATIVE)
        or _contains_any(ascii_text, _RESTRICTION_TARGET_ASCII)
    )
    has_authority = (
        _contains_any(native, _AUTHORITY_NATIVE)
        or _contains_any(ascii_text, _AUTHORITY_ASCII)
    )
    has_disable = (
        _contains_any(native, _DISABLE_NATIVE)
        or _contains_any(ascii_text, _DISABLE_ASCII)
    )
    has_new_instruction = (
        _contains_any(native, _NEW_INSTRUCTION_NATIVE)
        or _contains_any(ascii_text, _NEW_INSTRUCTION_ASCII)
    )
    has_exfil = (
        _contains_any(native, _EXFIL_NATIVE)
        or _contains_any(ascii_text, _EXFIL_ASCII)
        or bool(_EMAIL_RE.search(text))
    )

    if has_override and (has_instruction_target or has_sensitive_target):
        add("multilingual_heuristic_override")

    if has_disable and has_restriction_target:
        add("multilingual_heuristic_override")

    if has_reveal and has_sensitive_target:
        add("multilingual_heuristic_extraction")

    if ((has_roleplay and (has_unrestricted or has_sensitive_target or has_persona))
            or (has_persona and has_unrestricted)):
        add("multilingual_heuristic_roleplay")

    if has_authority and (has_disable or has_sensitive_target):
        add("multilingual_heuristic_authority")

    if has_new_instruction and (has_reveal or has_sensitive_target or has_unrestricted or has_disable):
        add("multilingual_heuristic_new_instruction")

    if has_exfil and has_sensitive_target:
        add("multilingual_heuristic_exfiltration")

    translit_override = (
        transliteration_seed
        and (has_override or has_reveal)
        and (has_instruction_target or has_sensitive_target)
    )
    if translit_override:
        add("multilingual_heuristic_transliteration")

    return hits
