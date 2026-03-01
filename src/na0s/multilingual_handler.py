"""Multilingual injection detection handler (D6 category).

Detects prompt injection attacks delivered in non-English languages by
matching known injection patterns translated into 20+ languages.  This
addresses the 40 known D6 detection gaps where the English-only ML model
and English regex rules fail on foreign-language payloads.

Strategy:
    1. Keyword-based detection: pre-compiled regex patterns for common
       injection phrases in French, Spanish, German, Chinese, Japanese,
       Arabic, Korean, Hindi, Russian, Portuguese, Italian, Turkish,
       Dutch, Polish, Vietnamese, Thai, Indonesian, Swedish, Czech, Hebrew.
    2. Each language has patterns for the major attack categories:
       - Instruction override (D1 equivalent in target language)
       - System prompt extraction (E1 equivalent)
       - Roleplay hijack (D2 equivalent)
       - Authority/social engineering (C1 equivalent)
    3. Returns a list of rule hits with technique IDs mapped to D6.x.

Integration:
    Called from predict.py's classify_prompt() when language_detector flags
    non_english_input or mixed_language_input anomaly flags.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MultilingualHit:
    """A multilingual injection pattern match."""
    language: str
    pattern_name: str
    severity: str
    technique_ids: List[str]
    matched_text: str = ""


# ---------------------------------------------------------------------------
# Multilingual injection keyword patterns
# ---------------------------------------------------------------------------
# Each entry: (language, pattern_name, severity, technique_ids, regex)
# Patterns are kept short and specific to minimize false positives.
# All regex patterns use re.IGNORECASE.

_MULTILINGUAL_PATTERNS = []


def _add(lang, name, severity, technique_ids, pattern):
    """Register a multilingual pattern."""
    _MULTILINGUAL_PATTERNS.append(
        (lang, name, severity, technique_ids, re.compile(pattern, re.IGNORECASE))
    )


# ---- French (D6.1) ----
_add("fr", "fr_instruction_override", "high", ["D6.1", "D1"],
     r"(?:oubliez|ignorez|oublier|ignorer)\s+(?:tout|toutes|les|vos)\s+(?:ce\s+qui\s+prec|instructions|consignes|regles)")
_add("fr", "fr_reveal_prompt", "high", ["D6.1", "E1"],
     r"(?:affichez|montrez|revelez|devoilez|divulguez)\s+(?:le\s+contenu|vos|les|le)\s+.{0,40}(?:instructions|parametres|prompt|consignes|configuration|systeme)")
_add("fr", "fr_roleplay_hijack", "high", ["D6.1", "D2"],
     r"(?:faites\s+comme\s+si|comportez[- ]vous\s+comme|agissez\s+comme|pretendez)\s+(?:vous\s+(?:etiez|etes)|etre)")
_add("fr", "fr_authority_claim", "medium", ["D6.1", "C1"],
     r"en\s+tant\s+que\s+(?:responsable|administrateur|directeur|chef).{0,40}(?:ordonne|demande|exige)")
_add("fr", "fr_disable_protections", "critical", ["D6.1", "D1"],
     r"(?:desactivez|supprimez|enlevez|retirez)\s+(?:toutes?\s+)?(?:les\s+)?(?:protections|restrictions|limites|filtres|securite)")
_add("fr", "fr_secret_code", "high", ["D6.1", "E1"],
     r"(?:code\s+secret|mot\s+de\s+passe|donnees?\s+confidentiel)")
_add("fr", "fr_new_role", "high", ["D6.1", "D2"],
     r"(?:votre\s+nouveau\s+role|desormais\s+vous\s+etes|a\s+partir\s+de\s+maintenant)")

# ---- Spanish (D6.2) ----
_add("es", "es_instruction_override", "high", ["D6.2", "D1"],
     r"(?:olvida|ignora|ignorar|olvidar)\s+(?:todas?\s+)?(?:las\s+)?(?:instrucciones|reglas|directivas|ordenes)\s+(?:anteriores|previas)?")
_add("es", "es_reveal_prompt", "high", ["D6.2", "E1"],
     r"(?:muestra|revela|muéstrame|dime)\s+(?:tu|tus|el|las?)\s+(?:instrucciones|prompt|configuracion|sistema)")
_add("es", "es_roleplay_hijack", "high", ["D6.2", "D2"],
     r"(?:actua|comportate|finge|haz\s+(?:como|de))\s+(?:como\s+)?(?:si\s+fueras|un|una)")
_add("es", "es_authority_claim", "medium", ["D6.2", "C1"],
     r"(?:como\s+(?:administrador|responsable|jefe)|en\s+calidad\s+de).{0,40}(?:ordeno|exijo|demando)")
_add("es", "es_disable_protections", "critical", ["D6.2", "D1"],
     r"(?:desactiva|elimina|quita|remueve)\s+(?:todas?\s+)?(?:las\s+)?(?:protecciones|restricciones|limites|filtros|seguridad)")
_add("es", "es_new_role", "high", ["D6.2", "D2"],
     r"(?:tu\s+nuevo\s+rol|a\s+partir\s+de\s+ahora|desde\s+ahora\s+eres)")

# ---- German (D6 additional) ----
_add("de", "de_instruction_override", "high", ["D6", "D1"],
     r"(?:vergiss|ignorier|vergessen\s+sie|ignorieren\s+sie)\s+(?:alle\s+)?(?:bisherigen\s+)?(?:anweisungen|regeln|instruktionen|befehle)")
_add("de", "de_reveal_prompt", "high", ["D6", "E1"],
     r"(?:zeig|zeige|offenbare|enthülle|nenne)\s+(?:mir\s+)?(?:deine|die|den|das)\s+(?:anweisungen|system|prompt|konfiguration|passwort)")
_add("de", "de_roleplay_hijack", "high", ["D6", "D2"],
     r"(?:tu\s+so\s+als|verhalte\s+dich\s+als|gib\s+dich\s+als|stell\s+dich\s+als)")
_add("de", "de_disable_protections", "critical", ["D6", "D1"],
     r"(?:deaktiviere|entferne|lösche)\s+(?:alle\s+)?(?:schutz|sicherheits|beschränkung|filter)")

# ---- Chinese (D6.3) ----
_add("zh", "zh_instruction_override", "high", ["D6.3", "D1"],
     r"(?:忽略|无视|忘记|抛弃|放弃).{0,6}(?:指令|指示|规则|命令|限制|设定)")
_add("zh", "zh_reveal_prompt", "high", ["D6.3", "E1"],
     r"(?:显示|展示|告诉我|给我看|透露|泄露)\s*(?:你的|系统|初始)?\s*(?:提示词|系统提示|指令|配置|密码|设定)")
_add("zh", "zh_roleplay_hijack", "high", ["D6.3", "D2"],
     r"(?:假装|扮演|模拟|充当)\s*(?:你是|自己是|一个)")
_add("zh", "zh_disable_protections", "critical", ["D6.3", "D1"],
     r"(?:关闭|禁用|移除|解除|取消)\s*(?:所有|全部)?\s*(?:安全|保护|限制|过滤|防护)")
_add("zh", "zh_new_role", "high", ["D6.3", "D2"],
     r"(?:你现在是|从现在开始你是|你的新角色|你的新身份)")

# ---- Japanese (D6.4) ----
_add("ja", "ja_instruction_override", "high", ["D6.4", "D1"],
     r"(?:すべて|全て|以前|前)の(?:指示|命令|ルール|制限|制約)を(?:忘れ|無視|忘却|破棄)")
_add("ja", "ja_reveal_prompt", "high", ["D6.4", "E1"],
     r"(?:プロンプト|指示|命令|設定|パスワード).{0,6}(?:見せて|教えて|表示して|示して|開示して)|(?:見せて|教えて|表示して|示して|開示して).{0,10}(?:プロンプト|指示|命令|設定|パスワード)")
_add("ja", "ja_roleplay_hijack", "high", ["D6.4", "D2"],
     r"(?:のふりをして|になりきって|として振る舞って|のように行動して|を演じて)")
_add("ja", "ja_disable_protections", "critical", ["D6.4", "D1"],
     r"(?:無効|解除|削除|停止)に?(?:して|する)(?:ください)?(?:全て|すべて)?の?(?:安全|保護|制限|フィルター)")

# ---- Arabic (D6.5) ----
_add("ar", "ar_instruction_override", "high", ["D6.5", "D1"],
     r"(?:تجاهل|انسَ|أهمل|اترك)\s*(?:جميع|كل|كافة)?\s*(?:التعليمات|الأوامر|القواعد|القيود|التوجيهات)\s*(?:السابقة|الأولى)?")
_add("ar", "ar_reveal_prompt", "high", ["D6.5", "E1"],
     r"(?:أظهر|اكشف|أرني|اعرض)\s*(?:لي\s+)?(?:تعليماتك|نظامك|إعداداتك|كلمة\s*(?:المرور|السر))")
_add("ar", "ar_roleplay_hijack", "high", ["D6.5", "D2"],
     r"(?:تصرف|تظاهر|تخيل)\s+(?:كأنك|أنك|بأنك)")
_add("ar", "ar_disable_protections", "critical", ["D6.5", "D1"],
     r"(?:عطّل|أزل|ألغِ|أوقف)\s*(?:جميع|كل)?\s*(?:الحماية|القيود|الأمان|الفلاتر)")

# ---- Korean (D6 additional) ----
_add("ko", "ko_instruction_override", "high", ["D6", "D1"],
     r"(?:지시|명령|규칙|제한|지침).{0,6}(?:무시|잊어|잊으|무효|취소)|(?:모든|이전).{0,10}(?:지시|명령|규칙|제한|지침)")
_add("ko", "ko_reveal_prompt", "high", ["D6", "E1"],
     r"(?:프롬프트|지시|설정|비밀번호).{0,6}(?:보여줘|알려줘|공개해|표시해)|(?:보여줘|알려줘|공개해|표시해).{0,10}(?:프롬프트|지시|설정|비밀번호)")
_add("ko", "ko_roleplay_hijack", "high", ["D6", "D2"],
     r"(?:인\s*척|처럼\s*행동|역할을?\s*(?:해|맡|수행))")

# ---- Hindi (D6 additional) ----
_add("hi", "hi_instruction_override", "high", ["D6", "D1"],
     r"(?:निर्देश|नियम|आदेश|प्रतिबंध)\w*\s*.{0,10}(?:भूल|अनदेखा|नज़रअंदाज|ignore)|(?:सभी|पिछले|पुराने)\s*.{0,15}(?:भूल|अनदेखा|ignore)")
_add("hi", "hi_reveal_prompt", "high", ["D6", "E1"],
     r"(?:दिखाओ|बताओ|प्रकट करो)\s*(?:अपने|सिस्टम|मूल)?\s*(?:प्रॉम्प्ट|निर्देश|सेटिंग|पासवर्ड)")

# ---- Russian (D6 additional) ----
_add("ru", "ru_instruction_override", "high", ["D6", "D1"],
     r"(?:забудь|игнорируй|проигнорируй|отмени|забей\s+на)\s+(?:все\s+)?(?:предыдущие\s+)?(?:инструкции|правила|команды|указания|ограничения)")
_add("ru", "ru_reveal_prompt", "high", ["D6", "E1"],
     r"(?:покажи|раскрой|выведи|отобрази)\s+(?:мне\s+)?(?:твой|твои|системн\w*|начальн\w*)\s+(?:\w+\s+)?(?:промпт|инструкци\w*|настройк\w*|пароль\w*)")
_add("ru", "ru_roleplay_hijack", "high", ["D6", "D2"],
     r"(?:притворись|веди\s+себя\s+как|ты\s+теперь|играй\s+роль)")

# ---- Portuguese (D6 additional) ----
_add("pt", "pt_instruction_override", "high", ["D6", "D1"],
     r"(?:esqueça|ignore|desconsidere|abandone)\s+(?:todas?\s+)?(?:as\s+)?(?:instruções|regras|ordens|diretivas|restrições)\s*(?:anteriores|prévias)?")
_add("pt", "pt_reveal_prompt", "high", ["D6", "E1"],
     r"(?:mostre|revele|exiba|apresente)\s+(?:me\s+)?(?:seu|sua|o|as?)\s+(?:prompt|instruções|configuração|sistema|senha)")
_add("pt", "pt_roleplay_hijack", "high", ["D6", "D2"],
     r"(?:finja|aja\s+como|comporte-se\s+como|simule\s+ser)")

# ---- Italian (D6 additional) ----
_add("it", "it_instruction_override", "high", ["D6", "D1"],
     r"(?:dimentica|ignora|tralascia|scorda)\s+(?:tutte?\s+)?(?:le\s+)?(?:istruzioni|regole|direttive|restrizioni)\s*(?:precedenti|iniziali)?")
_add("it", "it_reveal_prompt", "high", ["D6", "E1"],
     r"(?:mostra|rivela|visualizza|esponi)\s+(?:mi\s+)?(?:il\s+tuo|le\s+tue|la\s+tua)\s+(?:prompt|istruzioni|configurazione|password)")
_add("it", "it_roleplay_hijack", "high", ["D6", "D2"],
     r"(?:fingi\s+di\s+essere|comportati\s+come|agisci\s+come|fai\s+finta)")

# ---- Turkish (D6 additional) ----
_add("tr", "tr_instruction_override", "high", ["D6", "D1"],
     r"(?:talimatlar\w*|kurallar\w*|yönergeler\w*|kısıtlamalar\w*).{0,10}(?:unut|görmezden\s+gel|yok\s+say|dikkate\s+alma)|(?:unut|görmezden\s+gel|yok\s+say|dikkate\s+alma).{0,10}(?:talimatlar|kurallar|yönergeler|kısıtlamalar)")
_add("tr", "tr_reveal_prompt", "high", ["D6", "E1"],
     r"(?:göster|ifşa\s+et|açıkla|göster\s+bana)\s*(?:senin|sistem|başlangıç)?\s*(?:talimatlar|prompt|yapılandırma|şifre)")

# ---- Dutch (D6 additional) ----
_add("nl", "nl_instruction_override", "high", ["D6", "D1"],
     r"(?:vergeet|negeer|veronachtzaam)\s+(?:alle\s+)?(?:vorige\s+|eerdere\s+)?(?:instructies|regels|opdrachten|beperkingen)")
_add("nl", "nl_reveal_prompt", "high", ["D6", "E1"],
     r"(?:toon|laat\s+zien|onthul|geef)\s+(?:me\s+)?(?:je|jouw|het|de)\s+(?:instructies|systeem|prompt|configuratie|wachtwoord)")

# ---- Polish (D6 additional) ----
_add("pl", "pl_instruction_override", "high", ["D6", "D1"],
     r"(?:zapomnij|zignoruj|pomiń|odrzuć)\s+(?:wszystkie\s+)?(?:poprzednie\s+)?(?:instrukcje|zasady|polecenia|ograniczenia)")
_add("pl", "pl_reveal_prompt", "high", ["D6", "E1"],
     r"(?:pokaż|ujawnij|wyświetl|podaj)\s+(?:mi\s+)?(?:swoje|twoje|systemowe|początkowe)\s+(?:instrukcje|prompt|ustawienia|hasło)")

# ---- Vietnamese (D6 additional) ----
_add("vi", "vi_instruction_override", "high", ["D6", "D1"],
     r"(?:quên|bỏ\s+qua|phớt\s+lờ|hủy\s+bỏ)\s+(?:tất\s+cả\s+)?(?:các\s+)?(?:chỉ\s+thị|quy\s+tắc|lệnh|hạn\s+chế)\s*(?:trước\s+đó)?")

# ---- Thai (D6 additional) ----
_add("th", "th_instruction_override", "high", ["D6", "D1"],
     r"(?:ลืม|เพิกเฉย|ละเลย|ยกเลิก)\s*(?:ทั้งหมด|ทุก)?\s*(?:คำสั่ง|กฎ|ข้อจำกัด|คำแนะนำ)\s*(?:ก่อนหน้า)?")

# ---- Indonesian (D6 additional) ----
_add("id", "id_instruction_override", "high", ["D6", "D1"],
     r"(?:lupakan|abaikan|hapus|buang)\s+(?:semua\s+)?(?:instruksi|aturan|perintah|batasan)\s*(?:sebelumnya)?")
_add("id", "id_reveal_prompt", "high", ["D6", "E1"],
     r"(?:tunjukkan|tampilkan|ungkapkan|berikan)\s+(?:saya\s+)?(?:instruksi|prompt|konfigurasi|kata\s+sandi)\s*(?:sistem)?")

# ---- Swedish (D6 additional) ----
_add("sv", "sv_instruction_override", "high", ["D6", "D1"],
     r"(?:glöm|ignorera|bortse\s+från)\s+(?:alla\s+)?(?:tidigare\s+)?(?:instruktioner|regler|kommandon|begränsningar)")

# ---- Czech (D6 additional) ----
_add("cs", "cs_instruction_override", "high", ["D6", "D1"],
     r"(?:zapomeň|ignoruj|přehlédni)\s+(?:všechny\s+)?(?:předchozí\s+)?(?:instrukce|pravidla|příkazy|omezení)")

# ---- Hebrew (D6 additional) ----
_add("he", "he_instruction_override", "high", ["D6", "D1"],
     r"(?:שכח|תתעלם|התעלם)\s+(?:מכל\s+)?(?:ה)?(?:הוראות|כללים|פקודות|הגבלות)\s*(?:הקודמות)?")

# ---- Transliteration / Romanization patterns ----
# Romanized versions of non-Latin attacks (Arabizi, Pinyin, Romaji, etc.)
_add("romanized", "romanized_ignore_instructions", "high", ["D6", "D1"],
     r"(?:oubliez|ignorez|olvida|vergiss|zabudʼ|wasure)"
     r"\s+.{0,20}"
     r"(?:instructions?|instrucciones|anweisungen|instrukcji|instruksi|talimatlar)")
_add("romanized", "romanized_system_prompt", "high", ["D6", "E1"],
     r"(?:montrer?|mostrar?|zeigen?|pokazat|misete)"
     r"\s+.{0,20}"
     r"(?:systeme?|sistema|prompt|parol|pasuwa|mima)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_multilingual(text: str) -> List[MultilingualHit]:
    """Scan text for multilingual injection patterns.

    Parameters
    ----------
    text : str
        The input text to scan (should be post-L0 sanitized text).

    Returns
    -------
    list[MultilingualHit]
        List of matched patterns.  Empty if no multilingual injection
        patterns were found.
    """
    if not text or not text.strip():
        return []

    hits = []
    for lang, name, severity, technique_ids, pattern in _MULTILINGUAL_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(MultilingualHit(
                language=lang,
                pattern_name=name,
                severity=severity,
                technique_ids=list(technique_ids),
                matched_text=match.group(0)[:100],
            ))

    return hits


def get_multilingual_rule_weight(hits: List[MultilingualHit]) -> float:
    """Compute a weighted score from multilingual hits.

    Uses the same severity weighting as the main rule engine.

    Parameters
    ----------
    hits : list[MultilingualHit]
        Hits from scan_multilingual().

    Returns
    -------
    float
        Weighted score contribution.
    """
    _SEVERITY_WEIGHTS = {
        "critical": 0.40,
        "critical_content": 0.40,
        "high": 0.25,
        "medium": 0.10,
        "low": 0.05,
    }
    total = 0.0
    for hit in hits:
        total += _SEVERITY_WEIGHTS.get(hit.severity, 0.10)
    return total
