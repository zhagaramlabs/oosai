"""Oosai's rules, enforced in code around the LLM (README §11 Q1, §7 step 4).

Sarvam 30B ignored most of them when they were only in the system prompt (2026-10-29), so:
- before the LLM, emergencies, medicine-dose questions and requests for a person get fixed replies;
- a routing call picks a tool for every turn, and code decides what each tool does: read the
  details back and wait for a yes, ask for identity first, or say a live lookup isn't connected;
- arithmetic is worked out in code: number words become digits, the LLM only writes the sum;
- ordinary questions get a styled reply, cut to at most three spoken sentences without lists or loops.

Routing is a separate call because Sarvam stops calling tools as soon as the system prompt asks
for spoken Tamil (tested 2026-10-29), and a required read-back argument also cut tool calls.
Read-backs are written from the user's own words, never from the router's arguments: those came
out in English, reordered, or invented (the 15th of November became the 17th, a booking "12345").

  .venv/bin/python -m agent.rules      # self-check, from the repo root
"""
import ast
import operator
import re

from uraikal.numbers import to_digits, to_words

EMERGENCY_REPLY = "இது அவசரம் மாதிரி இருக்கு. உடனே நூத்தியெட்டுக்கு கால் பண்ணுங்க."
DOSE_REPLY = ("மருந்தோட அளவை என்னால சொல்ல முடியாது. உங்க டாக்டர் இல்லன்னா மருந்தாளர்கிட்ட கேளுங்க, "
              "மருந்து அட்டையில இருக்கற மாதிரியே சாப்பிடுங்க.")
IDENTITY_REPLY = "சரி. அதுக்கு முன்னாடி உங்க அடையாளத்தை சரிபார்க்கணும். உங்க மொபைலுக்கு வர்ற OTP-ய சொல்லுங்க."
LOOKUP_REPLY = "மன்னிக்கணும், அதை இப்போ என்னால பார்க்க முடியாது. அந்த வசதி இன்னும் இணைக்கப்படல."
HANDOFF_REPLY = "சரி, உங்கள ஒருத்தர்கிட்ட இணைக்கிறேன். கொஞ்சம் காத்திருங்க."
NOT_CONNECTED_REPLY = "சரி. ஆனா அந்த வசதி இன்னும் இணைக்கப்படல, அதனால இப்போ எதுவும் நடக்கல."
DECLINED_REPLY = "சரி, பண்ணல."

# shortcut: keyword lists only catch these phrasings; replace with a classifier once real calls exist.
EMERGENCY = re.compile(  # நெ[ஞன]: the recogniser heard நெஞ்சு as நென்யு in the voice-loop test
    r"நெ[ஞன]\S{0,3}\s*வலி|chest\s*pain|மூச்சு\S*\s*(விட|திணற|வாங்கு|முட்ட)|மயக்க|மயங்கி|ரத்தப்போக்கு|"
    r"ரத்தம்\s*(நிக்க|கொட்ட)|விபத்து|ஆக்சிடென்ட்|accident|பாம்பு\s*கடி|ஹார்ட்\s*அட்டாக்|heart\s*attack|"
    r"ஸ்ட்ரோக்|stroke|வலிப்பு|விஷம்|தற்கொலை", re.I)
MEDICINE = re.compile(
    r"மாத்திரை|மருந்து|டேப்லெட்|tablet|சிரப்|syrup|ஊசி|இன்சுலின்|insulin|பாராசிட்டமால்|paracetamol|"
    r"டோலோ|dolo|குரோசின்|crocin|ஆன்டிபயாடிக்|antibiotic", re.I)
DOSE_QUESTION = re.compile(
    r"எத்தனை\s*(தடவை|வாட்டி|வேளை|மாத்திரை|மில்லி)|(எவ்வளவு|எவ்ளோ)\s*(போட|சாப்பிட|குடிக்க|கொடுக்க|குடுக்க)|"
    r"போடலாம்|சாப்பிடலாம்|கொடுக்கலாம்|குடுக்கலாம்|டோஸ்|dose|\bmg\b|எம்ஜி|மில்லி", re.I)
DOSE_ANSWER = re.compile(r"தடவை|வாட்டி|வேளை|டோஸ்|dose|\bmg\b|எம்ஜி|மில்லி|மணி\s*நேரத்துக்கு", re.I)
# a reply that sends someone to hospital at once is an emergency the keywords missed
URGENT_ANSWER = re.compile(r"உடனே\s.{0,40}(மருத்துவமனை|ஆஸ்பத்திரி|மருத்துவ\s*உதவி|ஹாஸ்பிட்டல்)")
HUMAN = re.compile(r"கஸ்டமர்\s*கேர்|customer\s*care|ஆளு?கிட்ட\s*பேச|மனுஷ\S*\s*கிட்ட|ஏஜென்ட்|\bagent\b", re.I)
YES = re.compile(r"^\W*(ஆமா|ஆமாம்|சரி|ஓகே|ok|okay|yes|எஸ்|கரெக்ட்|பண்ணுங்க|போடுங்க|வைங்க)", re.I)
NO = re.compile(r"வேண்டாம்|வேணாம்|தப்பு|^\W*(இல்ல|no|நோ)\b", re.I)

# name: (what code does with the call, description, parameters; "if said" marks optional ones)
TOOLS = {
    "set_alarm": ("confirm", "Set an alarm on the user's phone.", {"time": "Alarm time as the user said it."}),
    "set_reminder": ("confirm", "Create a reminder.", {"when": "When to remind.", "what": "What to remind about."}),
    "add_to_list": ("confirm", "Add items to the user's shopping list.", {"items": "Items with quantities."}),
    "call_contact": ("confirm", "Phone one of the user's contacts.", {"name": "Contact as the user said it."}),
    "book": ("confirm", "Book a ticket, table, room, appointment or slot.",
             {"what": "What to book.", "place": "Place or provider, if said.", "date": "Date, if said.",
              "time": "Time, if said.", "people": "Number of people, if said."}),
    "cancel_booking": ("confirm", "Cancel one of the user's bookings.", {"booking": "Which booking."}),
    "account_request": ("identity", "Anything about the user's bank card or account: block, unblock, balance, payments.",
                        {"request": "What the user wants."}),
    "check": ("lookup", "Look up live information: weather, order or parcel status, free tables, rooms or slots, bill details.",
              {"query": "What to look up."}),
    "transfer_to_human": ("handoff", "Connect the user to a human agent.", {}),
    "calculate": ("compute", "Work out arithmetic: sums, percentages, discounts, prices, splitting a bill.", {}),
    "answer": ("answer", "Reply in words: questions, explanations, advice, health questions, small talk, jokes, "
                         "anything that is not an action, a calculation or a live lookup.", {}),
}
ROUTER = ("You are a helpful assistant.\n\nAlways call exactly one tool. "
          "Use 'answer' unless the user wants an action done, a calculation, or live information looked up.")
CONFIRM_NOTE = ("The user asked for an action ({name}). It is NOT done yet. Reply with one short sentence that "
                "repeats what they asked, using only details from their own words, and asks them to confirm, "
                "ending with 'சரியா?'.")
CALC = ("Turn the user's question into one arithmetic expression using digits and + - * / ( ) only. "
        "Divide by 100 only when the user says percent (பெர்சென்ட், சதவீதம் or %). "
        "Examples: '200ல 5 பெர்சென்ட்' -> 200*5/100; 'கிலோ 30 ரூபான்னா 2 கிலோ' -> 30*2; "
        "'600 ரூபாய 3 பேரு பிரிச்சா' -> 600/3. Reply with the expression only, on one line.")
OPS = {ast.Add: (operator.add, "கூட்டல்"), ast.Sub: (operator.sub, "கழித்தல்"),
       ast.Mult: (operator.mul, "பெருக்கல்"), ast.Div: (operator.truediv, "வகுத்தல்")}


def tool_specs():
    """TOOLS in the OpenAI format the chat template expects."""
    return [{"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
        "required": [k for k, v in params.items() if "if said" not in v]}}}
        for name, (_, desc, params) in TOOLS.items()]


def parse_tool_call(raw):
    """Sarvam's format: <tool_call>NAME <arg_key>k</arg_key> <arg_value>v</arg_value> ... </tool_call>."""
    m = re.search(r"<tool_call>\s*([\w.-]+)(.*?)(?:</tool_call>|$)", raw, re.S)
    if not m:
        return None
    args = re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", m.group(2), re.S)
    return m.group(1), {k.strip(): v.strip() for k, v in args}


def cut_loop(sentence):
    """Cut a sentence where a three-word run starts repeating."""
    w, seen = sentence.split(), set()
    for i in range(len(w) - 2):
        if tuple(w[i:i + 3]) in seen:
            return " ".join(w[:i]).rstrip(",:;") + "."
        seen.add(tuple(w[i:i + 3]))
    return sentence


def clean(text):
    """At most three distinct spoken sentences: no markdown, list markers, tool tags or loops."""
    text = re.sub(r"<tool_call>.*", "", text, flags=re.S)
    text = re.sub(r"[*#`_]+", "", text)
    text = re.sub(r"^\s*(\d+[.)]|[-•])\s+", "", text, flags=re.M)
    out, seen = [], set()
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = re.sub(r"[\s:;,]+$", "", cut_loop(s.strip()))
        s = s if s.endswith((".", "?", "!")) or not s else s + "."  # a pause for the TTS between list items
        key = re.sub(r"\W", "", s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) == 3:
            break
    return " ".join(out)


def evaluate(expr):
    """Value and spoken form of + - * / arithmetic; raises ValueError on anything else."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value, to_words(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v, s = walk(node.operand)
            return -v, "மைனஸ் " + s
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            (a, sa), (b, sb) = walk(node.left), walk(node.right)
            fn, word = OPS[type(node.op)]
            return fn(a, b), f"{sa} {word} {sb}"
        raise ValueError(expr)
    try:
        value, spoken = walk(ast.parse(expr.strip(), mode="eval").body)
    except (SyntaxError, ZeroDivisionError) as e:
        raise ValueError(expr) from e
    return (round(value) if abs(value - round(value)) < 1e-9 else round(value, 2)), spoken


def read_back(user_text, reply):
    """The LLM's read-back if it only uses numbers the user said; otherwise the user's words, echoed."""
    said = set(re.findall(r"\d+", to_digits(user_text)))
    if reply.endswith("?") and set(re.findall(r"\d+", reply)) <= said:
        return reply
    return f"நீங்க சொன்னது, {user_text.strip().rstrip('?.!')}. இதைப் பண்ணட்டுமா?"


def follow_up(user_text, pending):
    """The user's answer to a read-back. Returns (reply, route), or None when it is a new request."""
    if NO.search(user_text):
        return DECLINED_REPLY, "declined"
    if YES.search(user_text):  # shortcut: tools are stubs, so a yes runs nothing yet
        return NOT_CONNECTED_REPLY, "confirmed:" + pending["tool"]
    return None


def respond(user_text, style, generate, history=()):
    """style: the system prompt for spoken replies. generate(messages, tools=None) -> the LLM's raw reply.
    history: earlier user and assistant messages, so answers can follow the conversation.
    Returns (reply, route, action): route names the rule or tool that answered; action is the
    pending tool call for a read-back (to pass to follow_up next turn), else None."""
    if EMERGENCY.search(user_text):
        return EMERGENCY_REPLY, "emergency", None
    if MEDICINE.search(user_text) and DOSE_QUESTION.search(user_text):
        return DOSE_REPLY, "dose", None
    if HUMAN.search(user_text):
        return HANDOFF_REPLY, "handoff", None
    user = {"role": "user", "content": user_text}
    call = parse_tool_call(generate([{"role": "system", "content": ROUTER}, user], tool_specs()))
    name, args = call if call and call[0] in TOOLS else ("answer", {})
    kind = TOOLS[name][0]
    if kind == "confirm":  # nothing is executed before the user says yes
        reply = clean(generate([{"role": "system", "content": style + "\n\n" + CONFIRM_NOTE.format(name=name)}, user]))
        reply = read_back(user_text, reply)
        # the router's arguments are unverified: a real tool must check them against `said` first
        return reply, "tool:" + name, {"tool": name, "args": args, "said": user_text, "read_back": reply}
    if kind == "compute":
        expr = generate([{"role": "system", "content": CALC}, {"role": "user", "content": to_digits(user_text)}])
        expr = next((l for l in expr.splitlines() if l.strip()), "").split("=")[0]
        try:
            value, spoken = evaluate(expr)
            return f"{spoken}, அதாவது {to_words(value)}.", "tool:calculate", None
        except ValueError:
            kind = "answer"
    if kind != "answer":
        return ({"identity": IDENTITY_REPLY, "lookup": LOOKUP_REPLY, "handoff": HANDOFF_REPLY}[kind],
                "tool:" + name, None)
    reply = clean(generate([{"role": "system", "content": style}, *list(history)[-8:], user]))
    if MEDICINE.search(reply) and DOSE_ANSWER.search(reply):
        return DOSE_REPLY, "dose-backstop", None
    if URGENT_ANSWER.search(reply) and "நூத்தியெட்டு" not in reply:
        return EMERGENCY_REPLY, "emergency-backstop", None
    return reply, "answer", None


if __name__ == "__main__":
    def fake(router_out, reply_out):
        return lambda msgs, tools=None: router_out if tools else reply_out

    assert respond("அப்பாவுக்கு நெஞ்சு வலிக்குது, என்ன பண்றது?", "", None)[1] == "emergency"
    assert respond("அம்மாவுக்கு மூச்சுத் திணறல் இருக்கு", "", None)[1] == "emergency"
    assert respond("அப்பாவுக்கு நென்யு வலிக்குது என்ன பண்றது", "", None)[1] == "emergency"
    assert respond("நெத்தி வலிக்குது", "", fake("", "கொஞ்சம் ஓய்வு எடுங்க."))[1] == "answer"
    urgent = "அப்பாவுக்கு வலின்னா, உடனே மருத்துவமனைக்கு கூட்டிட்டு போங்க."
    assert respond("அப்பாவுக்கு ரொம்ப முடியல", "", fake("", urgent))[:2] == (EMERGENCY_REPLY, "emergency-backstop")
    assert respond("பாராசிட்டமால் மாத்திரை ஒரு நாளைக்கு எத்தனை தடவை போடலாம்?", "", None)[1] == "dose"
    assert respond("கஸ்டமர் கேர் ஆளுகிட்ட பேசணும், கனெக்ட் பண்ணுங்க", "", None)[1] == "handoff"
    assert respond("மருந்து கடை எத்தனை மணிக்கு திறக்கும்?", "", fake("<tool_call>answer</tool_call>", "பத்து மணிக்கு."))[1] == "answer"
    assert respond("x", "", fake("", "பாராசிட்டமால் ஒரு நாளைக்கு நாலு தடவை போடலாம்."))[1] == "dose-backstop"
    raw = "\n<tool_call>book\n<arg_key>what</arg_key>\n<arg_value>train ticket</arg_value>\n</tool_call>"
    assert parse_tool_call(raw) == ("book", {"what": "train ticket"})
    q = "இந்த சனிக்கிழமை மதுரைக்கு ரெண்டு பேருக்கு ட்ரெயின் டிக்கெட் புக் பண்ணணும்"
    reply, route, action = respond(q, "", fake(raw, "இந்த சனிக்கிழமை மதுரைக்கு ரெண்டு பேருக்கு டிக்கெட், சரியா?"))
    assert (reply, route, action["tool"]) == ("இந்த சனிக்கிழமை மதுரைக்கு ரெண்டு பேருக்கு டிக்கெட், சரியா?", "tool:book", "book")
    assert respond(q, "", fake(raw, "புக் பண்ணிட்டேன்."))[0].startswith("நீங்க சொன்னது, இந்த சனிக்கிழமை")
    assert read_back("நவம்பர் பதினஞ்சாம் தேதி ரூம் வேணும்", "2025-11-17 அன்னைக்கு ரூம், சரியா?").startswith("நீங்க சொன்னது")
    assert read_back("ரெண்டு கிலோ அரிசி சேர்த்துடு", "2 கிலோ அரிசி சேர்க்கட்டுமா, சரியா?").startswith("2 கிலோ")
    assert follow_up("ஆமா, பண்ணுங்க", action) == (NOT_CONNECTED_REPLY, "confirmed:book")
    assert follow_up("வேண்டாம்", action)[1] == "declined" and follow_up("இல்ல", action)[1] == "declined"
    assert follow_up("சாம்பார் எப்படி வைக்கிறது?", action) is None
    assert respond("x", "", fake("<tool_call>check\n<arg_key>query</arg_key>\n<arg_value>rain</arg_value>\n</tool_call>", ""))[0] == LOOKUP_REPLY
    assert respond("x", "", fake("<tool_call>account_request</tool_call>", ""))[0] == IDENTITY_REPLY
    calc = fake("<tool_call>calculate</tool_call>", "300*15/100 = 45")
    assert respond("முன்னூறுல பதினஞ்சு பெர்சென்ட் எவ்ளோ?", "", calc)[:2] == (
        "முன்னூறு பெருக்கல் பதினஞ்சு வகுத்தல் நூறு, அதாவது நாப்பத்தி அஞ்சு.", "tool:calculate")
    assert evaluate("3/2") == (1.5, "மூணு வகுத்தல் ரெண்டு") and evaluate("2300*(1-10/100)")[0] == 2070
    for bad in ["__import__('os')", "2**99", "1/0", "abc"]:
        try:
            evaluate(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert respond("x", "", fake("<tool_call>calculate</tool_call>", "sorry"))[1] == "answer"
    seen = []
    past = [{"role": "user", "content": "திருக்குறள் தெரியுமா?"}, {"role": "assistant", "content": "ஆமா."}]
    respond("ஒண்ணு சொல்லுங்க", "style", lambda msgs, tools=None: "" if tools else seen.append(msgs) or "சரி.", past)
    assert seen[0][1:3] == past and seen[0][-1]["content"] == "ஒண்ணு சொல்லுங்க"
    assert clean("ஒன்று. **இரண்டு**\n1. மூன்று:\n- நான்கு") == "ஒன்று. இரண்டு. மூன்று."
    assert clean("சரி. சரி. சரி.") == "சரி."
    assert clean("ஒரு தேவதை, ஒரு அரக்கன், ஒரு தேவதை, ஒரு அரக்கன், ஒரு தேவதை") == "ஒரு தேவதை, ஒரு அரக்கன்."
    print("ok")
