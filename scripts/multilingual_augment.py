#!/usr/bin/env python3
"""Multilingual training data augmentation for prompt injection detection.

Generates balanced multilingual training CSV from regex pattern seeds defined
in ``src/na0s/multilingual_handler.py``.  Four augmentation strategies:

1. **Seed expansion** -- extract literal phrases from each regex pattern
2. **Case/char perturbation** -- random case, diacritics, char swaps
3. **Template paraphrasing** -- wrap seeds in command templates per language
4. **Safe sample generation** -- greetings, questions, translation requests

Output: ``data/multilingual/multilingual_training.csv`` with columns
``text,label`` (label 1 = malicious, 0 = safe).

Usage::

    python scripts/multilingual_augment.py
    python scripts/multilingual_augment.py --dry-run
    python scripts/multilingual_augment.py --target-per-lang 300
    python scripts/multilingual_augment.py --output /tmp/ml_train.csv

Pure Python -- no external downloads or API calls required.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import re
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_OUTPUT = os.path.join(ROOT, "data", "multilingual", "multilingual_training.csv")
DEFAULT_TARGET_PER_LANG = 500
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# 1. Extract seed phrases from multilingual_handler patterns
# ---------------------------------------------------------------------------

def _extract_literal_phrases(regex_source: str) -> List[str]:
    """Pull literal alternative phrases out of a regex pattern string.

    Handles ``(?:a|b|c)`` groups and strips regex metacharacters so we get
    plain-text seed phrases suitable for augmentation.
    """
    # Find all (?:...) groups and extract alternatives
    groups = re.findall(r'\(\?:([^)]+)\)', regex_source)
    phrases = []
    for g in groups:
        alts = g.split('|')
        for alt in alts:
            # Strip regex meta: quantifiers, char classes, anchors
            clean = re.sub(r'\\[swdSWD]', ' ', alt)
            clean = re.sub(r'[\[\]{}()?*+.^$\\]', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean) >= 2:
                phrases.append(clean)
    return phrases


def _load_handler_patterns() -> Dict[str, List[Tuple[str, str, List[str]]]]:
    """Load patterns from multilingual_handler.py by importing it.

    Returns dict: lang -> list of (pattern_name, severity, literal_phrases).
    """
    from na0s.multilingual_handler import _MULTILINGUAL_PATTERNS

    by_lang: Dict[str, List[Tuple[str, str, List[str]]]] = {}
    for lang, name, severity, technique_ids, compiled_re in _MULTILINGUAL_PATTERNS:
        literals = _extract_literal_phrases(compiled_re.pattern)
        if literals:
            by_lang.setdefault(lang, []).append((name, severity, literals))
    return by_lang


# ---------------------------------------------------------------------------
# 2. Malicious sample generators
# ---------------------------------------------------------------------------

# Per-language command templates: {phrase} is replaced with a seed phrase.
# These simulate how real attacks wrap injection phrases.
_COMMAND_TEMPLATES: Dict[str, List[str]] = {
    "fr": [
        "{phrase} tout de suite",
        "S'il te plaît, {phrase}",
        "{phrase} immédiatement",
        "Tu dois {phrase}",
        "Je t'ordonne de {phrase}",
        "Attention: {phrase}",
    ],
    "es": [
        "{phrase} ahora mismo",
        "Por favor, {phrase}",
        "{phrase} inmediatamente",
        "Tienes que {phrase}",
        "Te ordeno que {phrase}",
        "Atención: {phrase}",
    ],
    "de": [
        "{phrase} sofort",
        "Bitte {phrase}",
        "{phrase} jetzt",
        "Du musst {phrase}",
        "Achtung: {phrase}",
        "Wichtig: {phrase}",
    ],
    "zh": [
        "{phrase}，立刻执行",
        "请{phrase}",
        "{phrase}，现在就做",
        "你必须{phrase}",
        "注意：{phrase}",
        "重要：{phrase}",
    ],
    "ja": [
        "{phrase}、今すぐ",
        "{phrase}してください",
        "{phrase}、すぐに実行",
        "重要：{phrase}",
        "注意：{phrase}",
    ],
    "ar": [
        "{phrase} الآن",
        "من فضلك، {phrase}",
        "{phrase} فوراً",
        "يجب عليك {phrase}",
        "تنبيه: {phrase}",
    ],
    "ko": [
        "{phrase} 지금 당장",
        "{phrase} 해주세요",
        "즉시 {phrase}",
        "중요: {phrase}",
        "주의: {phrase}",
    ],
    "hi": [
        "{phrase} अभी करो",
        "कृपया {phrase}",
        "{phrase} तुरंत",
        "ज़रूरी: {phrase}",
        "ध्यान दो: {phrase}",
    ],
    "ru": [
        "{phrase} немедленно",
        "Пожалуйста, {phrase}",
        "{phrase} сейчас же",
        "Важно: {phrase}",
        "Внимание: {phrase}",
    ],
    "pt": [
        "{phrase} agora mesmo",
        "Por favor, {phrase}",
        "{phrase} imediatamente",
        "Você precisa {phrase}",
        "Atenção: {phrase}",
    ],
    "it": [
        "{phrase} subito",
        "Per favore, {phrase}",
        "{phrase} immediatamente",
        "Devi {phrase}",
        "Attenzione: {phrase}",
    ],
    "tr": [
        "{phrase} hemen şimdi",
        "Lütfen {phrase}",
        "{phrase} derhal",
        "Önemli: {phrase}",
        "Dikkat: {phrase}",
    ],
    "nl": [
        "{phrase} onmiddellijk",
        "Alsjeblieft, {phrase}",
        "{phrase} nu meteen",
        "Belangrijk: {phrase}",
        "Let op: {phrase}",
    ],
    "pl": [
        "{phrase} natychmiast",
        "Proszę, {phrase}",
        "{phrase} teraz",
        "Ważne: {phrase}",
        "Uwaga: {phrase}",
    ],
    "vi": [
        "{phrase} ngay bây giờ",
        "Xin hãy {phrase}",
        "{phrase} ngay lập tức",
        "Quan trọng: {phrase}",
        "Chú ý: {phrase}",
    ],
    "th": [
        "{phrase} ตอนนี้เลย",
        "กรุณา {phrase}",
        "{phrase} ทันที",
        "สำคัญ: {phrase}",
        "โปรดทราบ: {phrase}",
    ],
    "id": [
        "{phrase} sekarang juga",
        "Tolong {phrase}",
        "{phrase} segera",
        "Penting: {phrase}",
        "Perhatian: {phrase}",
    ],
    "sv": [
        "{phrase} omedelbart",
        "Snälla, {phrase}",
        "{phrase} genast",
        "Viktigt: {phrase}",
        "Obs: {phrase}",
    ],
    "cs": [
        "{phrase} ihned",
        "Prosím, {phrase}",
        "{phrase} okamžitě",
        "Důležité: {phrase}",
        "Pozor: {phrase}",
    ],
    "he": [
        "{phrase} עכשיו",
        "בבקשה, {phrase}",
        "{phrase} מיד",
        "חשוב: {phrase}",
        "שים לב: {phrase}",
    ],
    "romanized": [
        "{phrase} now",
        "Please {phrase}",
        "{phrase} immediately",
        "Important: {phrase}",
    ],
}

# Benign prefixes/suffixes to wrap around malicious phrases (evasion technique)
_BENIGN_WRAPPERS: Dict[str, List[Tuple[str, str]]] = {
    "fr": [("Bonjour, ", " Merci beaucoup."), ("", " C'est tout."), ("Excusez-moi, ", "")],
    "es": [("Hola, ", " Muchas gracias."), ("", " Eso es todo."), ("Disculpe, ", "")],
    "de": [("Hallo, ", " Vielen Dank."), ("", " Das ist alles."), ("Entschuldigung, ", "")],
    "zh": [("你好，", " 谢谢。"), ("", "就这些。"), ("请问，", "")],
    "ja": [("こんにちは、", " ありがとう。"), ("", "以上です。"), ("すみません、", "")],
    "ar": [("مرحباً، ", " شكراً جزيلاً."), ("", " هذا كل شيء."), ("عفواً، ", "")],
    "ko": [("안녕하세요, ", " 감사합니다."), ("", " 그게 전부입니다."), ("실례합니다, ", "")],
    "hi": [("नमस्ते, ", " बहुत धन्यवाद।"), ("", " बस इतना ही।"), ("माफ़ कीजिए, ", "")],
    "ru": [("Здравствуйте, ", " Спасибо."), ("", " Это всё."), ("Извините, ", "")],
    "pt": [("Olá, ", " Muito obrigado."), ("", " É tudo."), ("Desculpe, ", "")],
    "it": [("Ciao, ", " Grazie mille."), ("", " Questo è tutto."), ("Scusi, ", "")],
    "tr": [("Merhaba, ", " Teşekkür ederim."), ("", " Bu kadar."), ("Affedersiniz, ", "")],
    "nl": [("Hallo, ", " Hartelijk dank."), ("", " Dat is alles."), ("Pardon, ", "")],
    "pl": [("Cześć, ", " Dziękuję bardzo."), ("", " To wszystko."), ("Przepraszam, ", "")],
    "vi": [("Xin chào, ", " Cảm ơn rất nhiều."), ("", " Đó là tất cả."), ("Xin lỗi, ", "")],
    "th": [("สวัสดี ", " ขอบคุณมาก"), ("", " เท่านี้"), ("ขอโทษ ", "")],
    "id": [("Halo, ", " Terima kasih banyak."), ("", " Itu saja."), ("Maaf, ", "")],
    "sv": [("Hej, ", " Tack så mycket."), ("", " Det är allt."), ("Ursäkta, ", "")],
    "cs": [("Ahoj, ", " Děkuji moc."), ("", " To je vše."), ("Promiňte, ", "")],
    "he": [("שלום, ", " תודה רבה."), ("", " זה הכל."), ("סליחה, ", "")],
    "romanized": [("Hello, ", " Thanks."), ("", " That's all."), ("Sorry, ", "")],
}


def _perturb_case(text: str, rng: random.Random) -> str:
    """Random case perturbation: randomly upper/lower each character."""
    return ''.join(
        c.upper() if rng.random() < 0.3 else c.lower() if rng.random() < 0.3 else c
        for c in text
    )


def _char_perturb(text: str, rng: random.Random) -> str:
    """Light character perturbation: insert zero-width spaces, swap adjacent."""
    if len(text) < 4:
        return text
    result = list(text)
    # Randomly swap 1-2 adjacent character pairs
    for _ in range(rng.randint(0, 2)):
        idx = rng.randint(0, len(result) - 2)
        if result[idx].isalpha() and result[idx + 1].isalpha():
            result[idx], result[idx + 1] = result[idx + 1], result[idx]
    return ''.join(result)


def _generate_malicious(
    by_lang: Dict[str, List[Tuple[str, str, List[str]]]],
    target_per_lang: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Generate malicious samples from handler pattern seeds.

    Returns list of (text, "1") tuples.
    """
    samples = []
    for lang, patterns in by_lang.items():
        lang_samples = []

        # Collect all seed phrases for this language
        all_seeds = []
        for name, severity, literals in patterns:
            all_seeds.extend(literals)

        if not all_seeds:
            continue

        templates = _COMMAND_TEMPLATES.get(lang, ["{phrase}"])
        wrappers = _BENIGN_WRAPPERS.get(lang, [("", "")])

        # Strategy 1: Raw seeds
        for seed in all_seeds:
            lang_samples.append(seed)

        # Strategy 2: Case-perturbed seeds
        for seed in all_seeds:
            lang_samples.append(_perturb_case(seed, rng))

        # Strategy 3: Char-perturbed seeds
        for seed in all_seeds:
            lang_samples.append(_char_perturb(seed, rng))

        # Strategy 4: Template-wrapped seeds
        for seed in all_seeds:
            for tmpl in templates:
                lang_samples.append(tmpl.format(phrase=seed))

        # Strategy 5: Template + case perturbation
        for seed in all_seeds:
            tmpl = rng.choice(templates)
            lang_samples.append(tmpl.format(phrase=_perturb_case(seed, rng)))

        # Strategy 6: Benign-wrapped (evasion simulation)
        for seed in all_seeds:
            prefix, suffix = rng.choice(wrappers)
            lang_samples.append(f"{prefix}{seed}{suffix}")

        # Strategy 7: Combine two seeds from same language
        if len(all_seeds) >= 2:
            for _ in range(min(len(all_seeds) * 2, 50)):
                s1, s2 = rng.sample(all_seeds, 2)
                separator = rng.choice([". ", "; ", " — ", " ", "\n"])
                lang_samples.append(f"{s1}{separator}{s2}")

        # Deduplicate and sample to target
        seen = set()
        unique = []
        for s in lang_samples:
            s_stripped = s.strip()
            if s_stripped and s_stripped not in seen:
                seen.add(s_stripped)
                unique.append(s_stripped)

        if len(unique) > target_per_lang:
            unique = rng.sample(unique, target_per_lang)
        elif len(unique) < target_per_lang:
            # Oversample with additional perturbations
            extra_needed = target_per_lang - len(unique)
            extra = []
            for _ in range(extra_needed):
                base = rng.choice(list(seen))
                variant = _perturb_case(_char_perturb(base, rng), rng)
                if variant not in seen:
                    extra.append(variant)
                    seen.add(variant)
                else:
                    # Wrap in a template
                    tmpl = rng.choice(templates)
                    wrapped = tmpl.format(phrase=_perturb_case(base, rng))
                    extra.append(wrapped)
            unique.extend(extra[:extra_needed])

        samples.extend((text, "1") for text in unique)
    return samples


# ---------------------------------------------------------------------------
# 3. Safe sample generators
# ---------------------------------------------------------------------------

_SAFE_TEMPLATES: Dict[str, List[str]] = {
    "fr": [
        "Bonjour, comment allez-vous?",
        "Quelle est la capitale de la France?",
        "Pouvez-vous m'aider à traduire ce texte?",
        "Quel temps fait-il aujourd'hui?",
        "Merci beaucoup pour votre aide.",
        "Je voudrais réserver une table pour deux.",
        "Comment dit-on «hello» en français?",
        "Pouvez-vous me recommander un bon restaurant?",
        "Quelle heure est-il?",
        "J'aimerais en savoir plus sur l'histoire de Paris.",
        "Où se trouve la bibliothèque la plus proche?",
        "Pouvez-vous expliquer cette recette?",
        "Quels sont les horaires du musée?",
        "Je cherche un hôtel près de la gare.",
        "Comment fonctionne le métro?",
        "Quels livres me recommandez-vous?",
        "Pouvez-vous corriger ma grammaire?",
        "Quelle est la population de Lyon?",
        "Racontez-moi une blague amusante.",
        "Quels films français sont populaires?",
    ],
    "es": [
        "Hola, ¿cómo estás?",
        "¿Cuál es la capital de España?",
        "¿Puedes ayudarme a traducir este texto?",
        "¿Qué tiempo hace hoy?",
        "Muchas gracias por tu ayuda.",
        "Me gustaría reservar una mesa para dos.",
        "¿Cómo se dice «hello» en español?",
        "¿Puedes recomendarme un buen restaurante?",
        "¿Qué hora es?",
        "Me gustaría saber más sobre la historia de Madrid.",
        "¿Dónde está la biblioteca más cercana?",
        "¿Puedes explicar esta receta?",
        "¿Cuáles son los horarios del museo?",
        "Busco un hotel cerca de la estación.",
        "¿Cómo funciona el metro?",
        "¿Qué libros me recomiendas?",
        "¿Puedes corregir mi gramática?",
        "¿Cuál es la población de Barcelona?",
        "Cuéntame un chiste divertido.",
        "¿Cómo preparar una paella?",
    ],
    "de": [
        "Hallo, wie geht es Ihnen?",
        "Was ist die Hauptstadt von Deutschland?",
        "Können Sie mir helfen, diesen Text zu übersetzen?",
        "Wie ist das Wetter heute?",
        "Vielen Dank für Ihre Hilfe.",
        "Ich möchte einen Tisch für zwei reservieren.",
        "Wie sagt man «hello» auf Deutsch?",
        "Können Sie mir ein gutes Restaurant empfehlen?",
        "Wie spät ist es?",
        "Ich möchte mehr über die Geschichte Berlins erfahren.",
        "Wo ist die nächste Bibliothek?",
        "Können Sie dieses Rezept erklären?",
        "Welche Öffnungszeiten hat das Museum?",
        "Ich suche ein Hotel in der Nähe des Bahnhofs.",
        "Wie funktioniert die U-Bahn?",
        "Welche Bücher empfehlen Sie?",
        "Können Sie meine Grammatik korrigieren?",
        "Wie groß ist München?",
        "Erzählen Sie mir einen Witz.",
        "Was ist Oktoberfest?",
    ],
    "zh": [
        "你好，你怎么样？",
        "中国的首都是哪里？",
        "你能帮我翻译这段文字吗？",
        "今天天气怎么样？",
        "非常感谢你的帮助。",
        "我想订一张两人桌。",
        "你好用英语怎么说？",
        "你能推荐一家好餐厅吗？",
        "现在几点了？",
        "我想了解更多关于北京的历史。",
        "最近的图书馆在哪里？",
        "你能解释一下这个菜谱吗？",
        "博物馆的开放时间是什么？",
        "我在找火车站附近的酒店。",
        "地铁怎么坐？",
        "你推荐什么书？",
        "你能纠正我的语法吗？",
        "上海的人口有多少？",
        "给我讲个笑话。",
        "中国菜怎么做？",
    ],
    "ja": [
        "こんにちは、お元気ですか？",
        "日本の首都はどこですか？",
        "このテキストを翻訳してもらえますか？",
        "今日の天気はどうですか？",
        "お手伝いありがとうございます。",
        "二人用のテーブルを予約したいです。",
        "「hello」は日本語でなんと言いますか？",
        "おすすめのレストランを教えてください。",
        "今何時ですか？",
        "東京の歴史についてもっと知りたいです。",
        "一番近い図書館はどこですか？",
        "このレシピを説明してもらえますか？",
        "美術館の営業時間は？",
        "駅の近くのホテルを探しています。",
        "電車の乗り方を教えてください。",
        "おすすめの本はありますか？",
        "文法を直してもらえますか？",
        "大阪の人口はどれくらいですか？",
        "面白いジョークを教えてください。",
        "寿司の作り方を教えてください。",
    ],
    "ar": [
        "مرحباً، كيف حالك؟",
        "ما هي عاصمة مصر؟",
        "هل يمكنك مساعدتي في ترجمة هذا النص؟",
        "كيف الطقس اليوم؟",
        "شكراً جزيلاً على مساعدتك.",
        "أريد حجز طاولة لشخصين.",
        "كيف نقول «hello» بالعربية؟",
        "هل تنصحني بمطعم جيد؟",
        "كم الساعة الآن؟",
        "أريد معرفة المزيد عن تاريخ القاهرة.",
        "أين أقرب مكتبة؟",
        "هل يمكنك شرح هذه الوصفة؟",
        "ما مواعيد المتحف؟",
        "أبحث عن فندق قريب من المحطة.",
        "كيف يعمل المترو؟",
        "ما الكتب التي تنصحني بها؟",
        "هل يمكنك تصحيح قواعدي؟",
        "كم عدد سكان الرياض؟",
        "احكِ لي نكتة مضحكة.",
        "كيف أطبخ الكبسة؟",
    ],
    "ko": [
        "안녕하세요, 어떻게 지내세요?",
        "한국의 수도는 어디인가요?",
        "이 텍스트를 번역해 주실 수 있나요?",
        "오늘 날씨가 어떤가요?",
        "도움 주셔서 감사합니다.",
        "2인용 테이블을 예약하고 싶습니다.",
        "«hello»를 한국어로 어떻게 말하나요?",
        "좋은 레스토랑 추천해 주세요.",
        "지금 몇 시인가요?",
        "서울의 역사에 대해 더 알고 싶습니다.",
        "가장 가까운 도서관이 어디인가요?",
        "이 레시피를 설명해 주세요.",
        "박물관 운영 시간은 어떻게 되나요?",
        "역 근처 호텔을 찾고 있습니다.",
        "지하철은 어떻게 타나요?",
        "어떤 책을 추천하시나요?",
        "제 문법을 고쳐 주실 수 있나요?",
        "부산의 인구는 얼마나 되나요?",
        "재미있는 농담 하나 해주세요.",
        "김치 만드는 법을 알려주세요.",
    ],
    "hi": [
        "नमस्ते, आप कैसे हैं?",
        "भारत की राजधानी क्या है?",
        "क्या आप इस टेक्स्ट का अनुवाद करने में मदद कर सकते हैं?",
        "आज मौसम कैसा है?",
        "आपकी मदद के लिए बहुत धन्यवाद।",
        "मैं दो लोगों के लिए एक टेबल बुक करना चाहता हूँ।",
        "«hello» को हिंदी में क्या कहते हैं?",
        "क्या आप एक अच्छा रेस्तरां सुझा सकते हैं?",
        "अभी कितने बज रहे हैं?",
        "मुझे दिल्ली के इतिहास के बारे में और जानना है।",
        "सबसे नज़दीकी पुस्तकालय कहाँ है?",
        "क्या आप यह रेसिपी समझा सकते हैं?",
        "संग्रहालय के खुलने का समय क्या है?",
        "मैं स्टेशन के पास एक होटल ढूंढ रहा हूँ।",
        "मेट्रो कैसे काम करती है?",
        "आप कौन सी किताबें सुझाते हैं?",
        "क्या आप मेरी व्याकरण सही कर सकते हैं?",
        "मुंबई की जनसंख्या कितनी है?",
        "मुझे एक मज़ेदार चुटकुला सुनाओ।",
        "बिरयानी कैसे बनाते हैं?",
    ],
    "ru": [
        "Здравствуйте, как дела?",
        "Какая столица России?",
        "Можете помочь перевести этот текст?",
        "Какая сегодня погода?",
        "Большое спасибо за помощь.",
        "Я хотел бы забронировать столик на двоих.",
        "Как сказать «hello» по-русски?",
        "Можете порекомендовать хороший ресторан?",
        "Который сейчас час?",
        "Я хочу узнать больше об истории Москвы.",
        "Где ближайшая библиотека?",
        "Можете объяснить этот рецепт?",
        "Какие часы работы музея?",
        "Ищу гостиницу рядом с вокзалом.",
        "Как пользоваться метро?",
        "Какие книги вы рекомендуете?",
        "Можете исправить мою грамматику?",
        "Какое население Санкт-Петербурга?",
        "Расскажите мне смешную шутку.",
        "Как приготовить борщ?",
    ],
    "pt": [
        "Olá, como vai?",
        "Qual é a capital do Brasil?",
        "Pode me ajudar a traduzir este texto?",
        "Como está o tempo hoje?",
        "Muito obrigado pela ajuda.",
        "Gostaria de reservar uma mesa para dois.",
        "Como se diz «hello» em português?",
        "Pode recomendar um bom restaurante?",
        "Que horas são?",
        "Gostaria de saber mais sobre a história de Lisboa.",
        "Onde fica a biblioteca mais próxima?",
        "Pode explicar esta receita?",
        "Quais são os horários do museu?",
        "Estou procurando um hotel perto da estação.",
        "Como funciona o metrô?",
        "Que livros você recomenda?",
        "Pode corrigir minha gramática?",
        "Qual é a população de São Paulo?",
        "Me conte uma piada engraçada.",
        "Como preparar feijoada?",
    ],
    "it": [
        "Ciao, come stai?",
        "Qual è la capitale dell'Italia?",
        "Puoi aiutarmi a tradurre questo testo?",
        "Com'è il tempo oggi?",
        "Grazie mille per il tuo aiuto.",
        "Vorrei prenotare un tavolo per due.",
        "Come si dice «hello» in italiano?",
        "Puoi consigliarmi un buon ristorante?",
        "Che ore sono?",
        "Vorrei saperne di più sulla storia di Roma.",
        "Dov'è la biblioteca più vicina?",
        "Puoi spiegare questa ricetta?",
        "Quali sono gli orari del museo?",
        "Cerco un hotel vicino alla stazione.",
        "Come funziona la metropolitana?",
        "Che libri mi consigli?",
        "Puoi correggere la mia grammatica?",
        "Qual è la popolazione di Milano?",
        "Raccontami una barzelletta divertente.",
        "Come si prepara la pasta alla carbonara?",
    ],
    "tr": [
        "Merhaba, nasılsınız?",
        "Türkiye'nin başkenti neresidir?",
        "Bu metni çevirmeme yardımcı olabilir misiniz?",
        "Bugün hava nasıl?",
        "Yardımınız için çok teşekkür ederim.",
        "İki kişilik bir masa ayırtmak istiyorum.",
        "«Hello» Türkçe'de nasıl söylenir?",
        "Bana iyi bir restoran önerebilir misiniz?",
        "Saat kaç?",
        "İstanbul'un tarihi hakkında daha fazla bilgi edinmek istiyorum.",
        "En yakın kütüphane nerede?",
        "Bu tarifi açıklayabilir misiniz?",
        "Müzenin çalışma saatleri nedir?",
        "İstasyon yakınlarında bir otel arıyorum.",
        "Metro nasıl kullanılır?",
        "Hangi kitapları önerirsiniz?",
        "Dilbilgimi düzeltebilir misiniz?",
        "Ankara'nın nüfusu ne kadardır?",
        "Bana komik bir fıkra anlat.",
        "Kebap nasıl yapılır?",
    ],
    "nl": [
        "Hallo, hoe gaat het?",
        "Wat is de hoofdstad van Nederland?",
        "Kunt u mij helpen deze tekst te vertalen?",
        "Wat voor weer is het vandaag?",
        "Hartelijk dank voor uw hulp.",
        "Ik wil graag een tafel voor twee reserveren.",
        "Hoe zeg je «hello» in het Nederlands?",
        "Kunt u een goed restaurant aanbevelen?",
        "Hoe laat is het?",
        "Ik wil meer weten over de geschiedenis van Amsterdam.",
        "Waar is de dichtstbijzijnde bibliotheek?",
        "Kunt u dit recept uitleggen?",
        "Wat zijn de openingstijden van het museum?",
        "Ik zoek een hotel in de buurt van het station.",
        "Hoe werkt de metro?",
        "Welke boeken raadt u aan?",
        "Kunt u mijn grammatica corrigeren?",
        "Hoe groot is Rotterdam?",
        "Vertel me een leuke grap.",
        "Hoe maak je stroopwafels?",
    ],
    "pl": [
        "Cześć, jak się masz?",
        "Jaka jest stolica Polski?",
        "Czy możesz mi pomóc przetłumaczyć ten tekst?",
        "Jaka jest dzisiaj pogoda?",
        "Dziękuję bardzo za pomoc.",
        "Chciałbym zarezerwować stolik dla dwóch osób.",
        "Jak powiedzieć «hello» po polsku?",
        "Czy możesz polecić dobrą restaurację?",
        "Która jest godzina?",
        "Chciałbym dowiedzieć się więcej o historii Warszawy.",
        "Gdzie jest najbliższa biblioteka?",
        "Czy możesz wyjaśnić ten przepis?",
        "Jakie są godziny otwarcia muzeum?",
        "Szukam hotelu w pobliżu dworca.",
        "Jak działa metro?",
        "Jakie książki polecasz?",
        "Czy możesz poprawić moją gramatykę?",
        "Ile osób mieszka w Krakowie?",
        "Opowiedz mi śmieszny dowcip.",
        "Jak zrobić pierogi?",
    ],
    "vi": [
        "Xin chào, bạn khỏe không?",
        "Thủ đô của Việt Nam là gì?",
        "Bạn có thể giúp tôi dịch đoạn văn này không?",
        "Thời tiết hôm nay thế nào?",
        "Cảm ơn bạn rất nhiều vì đã giúp đỡ.",
        "Tôi muốn đặt bàn cho hai người.",
        "«Hello» trong tiếng Việt nói thế nào?",
        "Bạn có thể giới thiệu một nhà hàng ngon không?",
        "Bây giờ là mấy giờ?",
        "Tôi muốn tìm hiểu thêm về lịch sử Hà Nội.",
        "Thư viện gần nhất ở đâu?",
        "Bạn có thể giải thích công thức này không?",
        "Giờ mở cửa của bảo tàng là gì?",
        "Tôi đang tìm khách sạn gần ga tàu.",
        "Tàu điện ngầm hoạt động như thế nào?",
        "Bạn giới thiệu sách gì?",
        "Bạn có thể sửa ngữ pháp của tôi không?",
        "Dân số Thành phố Hồ Chí Minh là bao nhiêu?",
        "Kể cho tôi một câu chuyện cười.",
        "Phở nấu như thế nào?",
    ],
    "th": [
        "สวัสดี สบายดีไหม?",
        "เมืองหลวงของประเทศไทยคืออะไร?",
        "คุณช่วยแปลข้อความนี้ได้ไหม?",
        "วันนี้อากาศเป็นอย่างไร?",
        "ขอบคุณมากสำหรับความช่วยเหลือ",
        "ฉันต้องการจองโต๊ะสำหรับสองคน",
        "«hello» ในภาษาไทยพูดว่าอย่างไร?",
        "คุณแนะนำร้านอาหารดีๆได้ไหม?",
        "ตอนนี้กี่โมง?",
        "ฉันอยากรู้เพิ่มเติมเกี่ยวกับประวัติศาสตร์กรุงเทพ",
        "ห้องสมุดที่ใกล้ที่สุดอยู่ไหน?",
        "คุณอธิบายสูตรนี้ได้ไหม?",
        "เวลาเปิดทำการของพิพิธภัณฑ์คืออะไร?",
        "ฉันกำลังหาโรงแรมใกล้สถานีรถไฟ",
        "รถไฟฟ้าใช้งานอย่างไร?",
        "คุณแนะนำหนังสืออะไร?",
        "คุณช่วยแก้ไขไวยากรณ์ของฉันได้ไหม?",
        "เชียงใหม่มีประชากรเท่าไหร่?",
        "เล่าเรื่องตลกให้ฟังหน่อย",
        "ต้มยำกุ้งทำอย่างไร?",
    ],
    "id": [
        "Halo, apa kabar?",
        "Apa ibu kota Indonesia?",
        "Bisakah Anda membantu saya menerjemahkan teks ini?",
        "Bagaimana cuaca hari ini?",
        "Terima kasih banyak atas bantuan Anda.",
        "Saya ingin memesan meja untuk dua orang.",
        "Bagaimana cara mengatakan «hello» dalam bahasa Indonesia?",
        "Bisakah Anda merekomendasikan restoran yang bagus?",
        "Jam berapa sekarang?",
        "Saya ingin tahu lebih banyak tentang sejarah Jakarta.",
        "Di mana perpustakaan terdekat?",
        "Bisakah Anda menjelaskan resep ini?",
        "Apa jam buka museum?",
        "Saya mencari hotel dekat stasiun.",
        "Bagaimana cara naik kereta?",
        "Buku apa yang Anda rekomendasikan?",
        "Bisakah Anda memperbaiki tata bahasa saya?",
        "Berapa jumlah penduduk Surabaya?",
        "Ceritakan lelucon yang lucu.",
        "Bagaimana cara membuat nasi goreng?",
    ],
    "sv": [
        "Hej, hur mår du?",
        "Vad är Sveriges huvudstad?",
        "Kan du hjälpa mig att översätta den här texten?",
        "Hur är vädret idag?",
        "Tack så mycket för din hjälp.",
        "Jag vill boka ett bord för två.",
        "Hur säger man «hello» på svenska?",
        "Kan du rekommendera en bra restaurang?",
        "Vad är klockan?",
        "Jag vill veta mer om Stockholms historia.",
        "Var ligger närmaste bibliotek?",
        "Kan du förklara det här receptet?",
        "Vilka är museets öppettider?",
        "Jag letar efter ett hotell nära stationen.",
        "Hur fungerar tunnelbanan?",
        "Vilka böcker rekommenderar du?",
        "Kan du rätta min grammatik?",
        "Hur stor är Göteborg?",
        "Berätta en rolig historia.",
        "Hur lagar man köttbullar?",
    ],
    "cs": [
        "Ahoj, jak se máš?",
        "Jaké je hlavní město České republiky?",
        "Můžeš mi pomoct přeložit tento text?",
        "Jaké je dnes počasí?",
        "Mockrát děkuji za pomoc.",
        "Chtěl bych zarezervovat stůl pro dva.",
        "Jak se řekne «hello» česky?",
        "Můžeš doporučit dobrou restauraci?",
        "Kolik je hodin?",
        "Chtěl bych se dozvědět více o historii Prahy.",
        "Kde je nejbližší knihovna?",
        "Můžeš vysvětlit tento recept?",
        "Jaká je otevírací doba muzea?",
        "Hledám hotel blízko nádraží.",
        "Jak funguje metro?",
        "Jaké knihy doporučuješ?",
        "Můžeš opravit mou gramatiku?",
        "Kolik obyvatel má Brno?",
        "Řekni mi vtip.",
        "Jak se dělají knedlíky?",
    ],
    "he": [
        "שלום, מה שלומך?",
        "מה בירת ישראל?",
        "אתה יכול לעזור לי לתרגם את הטקסט הזה?",
        "איך מזג האוויר היום?",
        "תודה רבה על העזרה.",
        "אני רוצה להזמין שולחן לשניים.",
        "איך אומרים «hello» בעברית?",
        "אתה יכול להמליץ על מסעדה טובה?",
        "מה השעה?",
        "אני רוצה לדעת עוד על ההיסטוריה של ירושלים.",
        "איפה הספרייה הקרובה ביותר?",
        "אתה יכול להסביר את המתכון הזה?",
        "מה שעות הפתיחה של המוזיאון?",
        "אני מחפש מלון ליד התחנה.",
        "איך הרכבת הקלה עובדת?",
        "אילו ספרים אתה ממליץ?",
        "אתה יכול לתקן את הדקדוק שלי?",
        "מה אוכלוסיית תל אביב?",
        "ספר לי בדיחה מצחיקה.",
        "איך מכינים פלאפל?",
    ],
}

# Additional safe template patterns that can be parameterized
_SAFE_QUESTION_PATTERNS: Dict[str, List[str]] = {
    "fr": [
        "Qu'est-ce que {topic}?",
        "Comment fonctionne {topic}?",
        "Pourquoi {topic} est important?",
        "Expliquez-moi {topic} en termes simples.",
    ],
    "es": [
        "¿Qué es {topic}?",
        "¿Cómo funciona {topic}?",
        "¿Por qué {topic} es importante?",
        "Explícame {topic} en términos sencillos.",
    ],
    "de": [
        "Was ist {topic}?",
        "Wie funktioniert {topic}?",
        "Warum ist {topic} wichtig?",
        "Erklären Sie mir {topic} in einfachen Worten.",
    ],
    "zh": [
        "{topic}是什么？",
        "{topic}怎么运作？",
        "为什么{topic}很重要？",
        "用简单的话解释一下{topic}。",
    ],
    "ja": [
        "{topic}とは何ですか？",
        "{topic}はどのように機能しますか？",
        "なぜ{topic}は重要ですか？",
        "{topic}を簡単に説明してください。",
    ],
    "ar": [
        "ما هو {topic}؟",
        "كيف يعمل {topic}؟",
        "لماذا {topic} مهم؟",
        "اشرح لي {topic} بكلمات بسيطة.",
    ],
    "ko": [
        "{topic}이란 무엇인가요?",
        "{topic}은 어떻게 작동하나요?",
        "왜 {topic}이 중요한가요?",
        "{topic}을 쉬운 말로 설명해 주세요.",
    ],
    "hi": [
        "{topic} क्या है?",
        "{topic} कैसे काम करता है?",
        "{topic} क्यों महत्वपूर्ण है?",
        "{topic} को सरल शब्दों में समझाएं।",
    ],
    "ru": [
        "Что такое {topic}?",
        "Как работает {topic}?",
        "Почему {topic} важно?",
        "Объясните мне {topic} простыми словами.",
    ],
    "pt": [
        "O que é {topic}?",
        "Como funciona {topic}?",
        "Por que {topic} é importante?",
        "Explique-me {topic} em termos simples.",
    ],
    "it": [
        "Cos'è {topic}?",
        "Come funziona {topic}?",
        "Perché {topic} è importante?",
        "Spiegami {topic} in parole semplici.",
    ],
    "tr": [
        "{topic} nedir?",
        "{topic} nasıl çalışır?",
        "Neden {topic} önemlidir?",
        "{topic} konusunu basit kelimelerle açıkla.",
    ],
    "nl": [
        "Wat is {topic}?",
        "Hoe werkt {topic}?",
        "Waarom is {topic} belangrijk?",
        "Leg {topic} uit in eenvoudige woorden.",
    ],
    "pl": [
        "Co to jest {topic}?",
        "Jak działa {topic}?",
        "Dlaczego {topic} jest ważne?",
        "Wyjaśnij mi {topic} prostymi słowami.",
    ],
    "vi": [
        "{topic} là gì?",
        "{topic} hoạt động như thế nào?",
        "Tại sao {topic} quan trọng?",
        "Giải thích {topic} bằng từ ngữ đơn giản.",
    ],
    "th": [
        "{topic} คืออะไร?",
        "{topic} ทำงานอย่างไร?",
        "ทำไม {topic} จึงสำคัญ?",
        "อธิบาย {topic} ด้วยคำง่ายๆ",
    ],
    "id": [
        "Apa itu {topic}?",
        "Bagaimana {topic} bekerja?",
        "Mengapa {topic} penting?",
        "Jelaskan {topic} dengan kata-kata sederhana.",
    ],
    "sv": [
        "Vad är {topic}?",
        "Hur fungerar {topic}?",
        "Varför är {topic} viktigt?",
        "Förklara {topic} med enkla ord.",
    ],
    "cs": [
        "Co je {topic}?",
        "Jak funguje {topic}?",
        "Proč je {topic} důležité?",
        "Vysvětlete mi {topic} jednoduchými slovy.",
    ],
    "he": [
        "מה זה {topic}?",
        "איך {topic} עובד?",
        "למה {topic} חשוב?",
        "תסביר לי {topic} במילים פשוטות.",
    ],
}

# Neutral topics for safe question generation
_TOPICS = [
    "machine learning", "photosynthesis", "climate change", "blockchain",
    "quantum computing", "DNA", "solar energy", "recycling",
    "artificial intelligence", "the internet", "democracy", "gravity",
    "nutrition", "music theory", "economics", "architecture",
    "psychology", "astronomy", "medicine", "literature",
]


def _generate_safe(
    target_per_lang: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Generate safe multilingual samples.

    Returns list of (text, "0") tuples.
    """
    samples = []
    for lang, base_templates in _SAFE_TEMPLATES.items():
        lang_samples = list(base_templates)

        # Add question-pattern expansions
        q_patterns = _SAFE_QUESTION_PATTERNS.get(lang, [])
        for pattern in q_patterns:
            for topic in _TOPICS:
                lang_samples.append(pattern.format(topic=topic))

        # Add case-perturbed versions of base templates
        for tmpl in base_templates:
            lang_samples.append(_perturb_case(tmpl, rng))

        # Combine two safe sentences
        for _ in range(min(len(base_templates) * 3, 60)):
            s1 = rng.choice(base_templates)
            s2 = rng.choice(base_templates)
            if s1 != s2:
                lang_samples.append(f"{s1} {s2}")

        # Deduplicate
        seen = set()
        unique = []
        for s in lang_samples:
            s_stripped = s.strip()
            if s_stripped and s_stripped not in seen:
                seen.add(s_stripped)
                unique.append(s_stripped)

        if len(unique) > target_per_lang:
            unique = rng.sample(unique, target_per_lang)
        elif len(unique) < target_per_lang:
            # Oversample
            extra_needed = target_per_lang - len(unique)
            for _ in range(extra_needed):
                base = rng.choice(list(seen))
                variant = _perturb_case(base, rng)
                unique.append(variant)

        samples.extend((text, "0") for text in unique)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multilingual training data for prompt injection detection."
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target-per-lang", "-t",
        type=int,
        default=DEFAULT_TARGET_PER_LANG,
        help=f"Target samples per language per class (default: {DEFAULT_TARGET_PER_LANG})",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing files",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target = args.target_per_lang

    # Load patterns from multilingual_handler
    print("Loading patterns from multilingual_handler.py ...")
    by_lang = _load_handler_patterns()
    print(f"  Found {len(by_lang)} language groups with patterns")
    total_seeds = sum(len(phrases) for pats in by_lang.values() for _, _, phrases in pats)
    print(f"  Total seed phrases extracted: {total_seeds}")

    # Generate malicious samples
    print(f"\nGenerating ~{target} malicious samples per language ...")
    malicious = _generate_malicious(by_lang, target, rng)
    print(f"  Generated {len(malicious)} malicious samples")

    # Generate safe samples
    print(f"\nGenerating ~{target} safe samples per language ...")
    safe = _generate_safe(target, rng)
    print(f"  Generated {len(safe)} safe samples")

    # Combine and shuffle
    all_samples = malicious + safe
    rng.shuffle(all_samples)

    # Statistics
    from collections import Counter
    label_counts = Counter(label for _, label in all_samples)
    print(f"\n--- Summary ---")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Malicious (1): {label_counts['1']}")
    print(f"  Safe (0):      {label_counts['0']}")

    # Per-language stats (approximate via lang detection from handler keys)
    lang_keys_malicious = Counter()
    for lang in by_lang:
        # Count how many malicious samples were generated for this lang
        # (we track by generation, not by detection)
        lang_keys_malicious[lang] = min(target, target)  # approximate
    print(f"  Languages with malicious patterns: {sorted(by_lang.keys())}")
    print(f"  Languages with safe templates: {sorted(_SAFE_TEMPLATES.keys())}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        print(f"  Would write to: {args.output}")
        # Show a few sample rows
        print("\n  Sample rows:")
        for text, label in all_samples[:5]:
            display = text[:80] + ("..." if len(text) > 80 else "")
            print(f"    label={label}  {display}")
        return

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label"])
        for text, label in all_samples:
            writer.writerow([text, label])

    print(f"\nWrote {len(all_samples)} rows to {args.output}")


if __name__ == "__main__":
    main()
