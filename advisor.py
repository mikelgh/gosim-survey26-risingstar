"""The model-driven stages of the pro agent. Two calls start at the beginning of every night:

1. night_plan   (natural-language understanding + plan adaptation): reads tonight's forecast and the current
   bulletin and decides whether tonight is a bad night for faint must-observe targets and which compass
   sectors to keep away from. The planner uses both answers for the whole night.
2. fault_review (data parsing + action decision): reads the agent's own hour-by-hour quality table of the last
   nights and judges how likely an unannounced instrument fault is. The answer sets how readily the agent
   reports (probes) a fault tonight.

A third, occasional call confirms a paid fault report before it is sent.

Calls run in the background (llm_client.Call); the agent waits for them only as long as the wall clock allows
and keeps planning otherwise. Every answer is validated; a missing or invalid answer leaves the rule-based
value in place for that night.
"""
from __future__ import annotations

import re
import time

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}

# v20i: keep every dialogue line that may carry handover intel, plus one neighbour on each side
# (rephrasing: "接上", "刚才那段的意思是", "以此为准").  A plain prefix cut (800 chars) drops the
# guide-camera maintenance windows, which sit in the middle/end of the 2400-4400-char notes.
_REASON_KEYWORDS = [
    "暖气", "咖啡机", "门禁卡", "打印机", "关顶", "下雨", "薄云", "薄霾", "雨",
    "遮挡", "度", "山", "往上", "照拍", "拍不了",
    "导星", "报修", "维护", "修",
    "平场", "镜盖", "测试",
    "摩斯", "唱名", "凯撒", "偏移", "密码", "英文",
    "整体偏差", "中规中矩", "平平", "一般", "长期预报", "评级",
    "风大", "风限", "超速",
    "预警", "GRB", "IceCube", "LVK", "follow-up", "发射窗口", "发射",
    "楼", "晃动", "吞吐", "计数", "抖动",
    "月亮", "月相", "涂层", "点检",
]


def extract_reason(reason: str, max_chars: int = 2000, radius: int = 1) -> str:
    """Compress a handover-note reason to the informative lines only."""
    lines = [l for l in (reason or "").splitlines() if l.strip()]
    if not lines:
        return ""
    keep = [False] * len(lines)
    for i, ln in enumerate(lines):
        if any(kw in ln for kw in _REASON_KEYWORDS):
            for j in range(max(0, i - radius), min(len(lines), i + radius + 1)):
                keep[j] = True
    out = [lines[i] for i in range(len(lines)) if keep[i]]
    s = "\n".join(out)
    if len(s) > max_chars:
        s = s[: max_chars * 3 // 4] + "\n...[截断]...\n" + s[-max_chars // 4:]
    return s


# --- v20i2: deterministic decoders for the coded intel inside handover notes. ---
# The notes carry (a) English notices Caesar-shifted by the day-of-month, and
# (b) clock times hummed as solfege (do=1..si=7, high do=8, high re=9, rest=0).
# Both are deterministic, so decode them with rules instead of the LLM and feed
# the plaintext back to the model; lines about the *neighbouring* station are
# tagged so the model never mistakes them for our sky.

_CAESAR_COMMON = [
    "update", "latest", "camera", "guider", "report", "maintenance", "service",
    "window", "tonight", "rain", "cloud", "flat", "lamp", "test", "data",
    "waste", "fault", "repair", "time", "local", "notes", "tonight", "tonight",
    "after", "before", "this", "that", "when", "will", "until", "because",
    "overcast", "shoot", "schedule", "launch", "observe", "observer",
]
_CAESAR_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _caesar(word: str, shift: int) -> str:
    return "".join(
        _CAESAR_ALPHA[(_CAESAR_ALPHA.index(c) + shift) % 26] if c in _CAESAR_ALPHA else c
        for c in word
    )


def _caesar_decrypt(text: str) -> str | None:
    """Brute-force the day-of-month shift (1..31); return plaintext when >=2 common
    words appear. Input is already lowercased ASCII words."""
    words = re.findall(r"[a-z]{2,}", text)
    if len(words) < 2:
        return None
    best, best_hits = None, 0
    for shift in range(1, 32):
        plain = [_caesar(w, shift) for w in words]
        hits = sum(1 for w in plain if len(w) >= 3 and w in _CAESAR_COMMON)
        if hits > best_hits:
            best, best_hits = plain, hits
    if best_hits >= 2:
        # rebuild every alphabetic word with the decoded form, keep punctuation
        pos = 0
        out = []
        for token in re.split(r"([a-z]{2,})", text):
            if token and token[0] in _CAESAR_ALPHA and len(token) >= 2:
                out.append(best[pos])
                pos += 1
            else:
                out.append(token)
        return "".join(out)
    return None


_SOLFEGE = {"do": 1, "re": 2, "mi": 3, "fa": 4, "sol": 5, "la": 6, "si": 7,
            "高音do": 8, "高音re": 9, "休止": 0}
_SOL_RE = re.compile(r"(高音\s*)?(do|re|mi|fa|sol|la|si|休止)")


def _solfege_clock(text: str) -> list[str]:
    """Decode 're re : fa sol' = 22:45, '休止 : fa sol' = 0:45, '高音do : 休止 休止' = 8:00."""
    out = []
    for m in re.finditer(r"([^。；\n]*?(?::|：)[^。；\n]*)", text):
        seg = m.group(1)
        parts = re.split(r":|：", seg)
        if len(parts) != 2:
            continue
        digits = []
        ok = True
        for part in parts:
            toks = _SOL_RE.findall(part.lower())
            if not toks:
                ok = False
                break
            vals = []
            for hi, name in toks:
                key = ("高音" + name) if hi else name
                key = key.strip()
                if key not in _SOLFEGE:
                    ok = False
                    break
                vals.append(_SOLFEGE[key])
            if not ok:
                break
            digits.append(vals)
        if not ok or len(digits) != 2:
            continue
        hh = "".join(str(v) for v in digits[0])
        mm = "".join(str(v) for v in digits[1])
        if hh and mm:
            out.append(f"{hh}:{mm}")
    return out


_NONLOCAL = ("隔壁", "另一座", "别站", "他们那边", "山谷里", "旁边", "邻站", "另一座山")


_CIPHER_BAD = ("暖气又坏", "打印机卡纸")     # all-day rain dome-closed / thick cloud cannot shoot
_CIPHER_OK = ("咖啡机没水", "咖啡机缺水")     # thin cloud, still shootable
_CIPHER_WIND = "门禁卡"                        # that direction windy, dome closed
_FLAT_KWS = ("平场", "镜盖", "测试", "flat", "Flat")
_DIR_MAP = {"东": "E", "西": "W", "南": "S", "北": "N", "东南": "SE", "西南": "SW", "东北": "NE", "西北": "NW"}


def station_cipher(reason: str) -> dict:
    """v20i6: deterministic decode of the station's coded handover notes. The LLM plan stays
    probabilistic; these ciphers are ground truth for bad_night / avoid / flat junk windows."""
    out = {"bad": None, "avoid": set(), "flat": [], "note": []}
    if not reason:
        return out
    for kw in _CIPHER_BAD:
        if kw in reason:
            out["bad"] = True
            out["note"].append(f"{kw}=全天下雨/厚云关顶")
    for kw in _CIPHER_OK:
        if kw in reason:
            out["note"].append(f"{kw}=薄云照拍(可观测)")
    if _CIPHER_WIND in reason:
        idx = reason.index(_CIPHER_WIND)
        seg = reason[max(0, idx - 40): idx + 80]
        for d in ("东南", "西南", "东北", "西北", "东", "西", "南", "北", "NE", "NW", "SE", "SW", "N", "E", "S", "W"):
            if d in seg:
                out["avoid"].add(_DIR_MAP.get(d, d))
                out["note"].append(f"{_CIPHER_WIND}={d}向风大关")
                break
    for m in re.finditer(r"(\d{1,2}:\d{2})\s*[-–—~至]\s*(\d{1,2}:\d{2})", reason):
        if any(k in reason for k in _FLAT_KWS):
            out["flat"].append(f"{m.group(1)}-{m.group(2)}")
    return out


def decode_intel(reason: str) -> str:
    """Decode the deterministic codes in a handover note; tag non-local lines."""
    lines = [l for l in (reason or "").splitlines() if l.strip()]
    dec = []
    for ln in lines:
        tag = ""
        if any(k in ln for k in _NONLOCAL):
            tag = "[非本站]"
        c = _caesar_decrypt(ln.lower())
        if c:
            dec.append(f"{tag}[凯撒] {c}")
            continue
        clocks = _solfege_clock(ln)
        if clocks:
            dec.append(f"{tag}[唱名时间] " + ", ".join(clocks))
            continue
        # a line that merely *defines* the codebook should not be decoded
        if tag and any(kw in ln for kw in _REASON_KEYWORDS):
            dec.append(tag + ln.strip()[:120])
    return "\n".join(dec) if dec else ""


NIGHT_PLAN_SYSTEM = (
    "You plan one night of a robotic spectroscopic survey. Input: tonight's forecast notices, the current bulletin, "
    "a survey summary, and the handover-note excerpts for the observation requests (coded Chinese notes). Reply with "
    "one JSON object only.\n"
    "Fields:\n"
    "- bad_night: true when tonight is effectively unobservable (all-sky rain/storm/thick cloud, or the notes say so "
    "in code: \"暖气又坏了\"=all-day rain dome-closed, \"打印机卡纸\"=thick cloud cannot shoot). \"咖啡机没水\"=thin "
    "cloud, still shootable → false. Only judge THIS night, not future nights the notes also mention.\n"
    "- avoid_directions: sectors with rain/storm/overcast/haze/cold_snap tonight, plus sectors the notes say are "
    "closed by strong wind (\"门禁卡刷不开\"=that direction windy, closed). Never list a sector nothing names.\n"
    "- terrain: per-sector mountain blocking elevation from the notes (degrees), e.g. {\"W\": 37, \"SW\": 33}; use "
    "the corrected/latest numbers (\"以此为准\"); null when the notes give none. The planner uses it to skip targets "
    "below the line; do NOT list such sectors in avoid_directions.\n"
    "- guider_report: \"report\" when the notes say the guide camera will be serviced tonight and must be reported at "
    "the given time (\"到点报修/请报修/直接报修\"); \"no_report\" when the notes say efficiency drop is normal "
    "(\"不是天的问题\"); null otherwise.\n"
    "- guider_time: the maintenance clock \"HH:MM\" (e.g. \"22:00\") when guider_report is \"report\"; null otherwise.\n"
    "- flat_windows: tonight's flat-lamp/lid-test junk windows as [\"HH:MM-HH:MM\", ...] (data is wasted, not a "
    "fault, never report); empty when none.\n"
    "- rating: \"bad\" when the long-range forecast for tonight is overall-biased/unreliable (\"整体偏差\", important "
    "targets should not be bet on this night); \"normal\" for 中规中矩/平平/一般; null when absent.\n"
    "- future_good_nights: how many GOOD (shootable) nights remain in the NEXT TWO nights after tonight "
    "(0/1/2). Read the long-range forecast in the notes (e.g. \"明晚雨\" → only 1 more good night after tonight; "
    "\"后天下雨/未来两天不可观测\" → 0). 2 when the notes give no usable long-range forecast. This is a FACT "
    "about the sky, not a suggestion.\n"
    "- duration_cap: longest exposure in seconds (1800/2400/3600). 1800 when must-observe targets are far behind "
    "and the sky is clear (short exposures finish more before deadline); 3600 when on track or sky poor.\n"
    "- reason: <12 words.\n"
    "Decoded intel lines in the notes (prefix [凯撒] / [唱名时间]) are deterministic decodes — trust them "
    "directly. Lines tagged [非本站] are about the NEIGHBOURING station, not ours: never use them for "
    "bad_night/avoid_directions.\n"
    "Reply: {\"bad_night\": bool, \"avoid_directions\": [..], \"terrain\": {...}|null, \"guider_report\": \"report\"|\"no_report\"|null, "
    "\"guider_time\": \"HH:MM\"|null, \"flat_windows\": [\"HH:MM-HH:MM\"]|[], \"rating\": \"bad\"|\"normal\"|null, "
    "\"future_good_nights\": 0|1|2, \"duration_cap\": 2400, \"reason\": \"..\"}"
)

FAULT_REVIEW_SYSTEM = (
    "You watch the data quality of a robotic telescope. An instrument fault is never announced: it lowers the "
    "instrument efficiency, and so the quality of every exposure, until someone reports it; a correct report "
    "repairs it at once. An earthquake (it appears in the bulletin) also lowers instrument efficiency, and that loss "
    "fades night by night; a report does not repair it. Weather lowers quality too, but it also lowers the program "
    "band, which the instrument does not affect.\n"
    "Columns per hour: E = measured quality / quality the program bands allow (about 1 when healthy; low when the "
    "instrument is the cause; in a very clear sky the bands bound it only loosely, so it can stay near 1), scale = "
    "measured sky quality relative to the clear-sky model, ref = the usual scale since the last repair. "
    "notices_now lists the current bulletin.\n"
    "Signs of a fault: quality that drops and stays down without recovering, E low for many hours across nights, "
    "not explained by announced weather or by a recent earthquake whose effect is fading.\n"
    "Reporting: a correct report earns 100 and repairs the instrument; false reports are free while "
    "free_false_reports_left > 0, afterwards each costs 150.\n"
    'Reply with one JSON object only: {"fault_likely": <0..1>, "reason": "<15 words"}'
)

CONFIRM_SYSTEM = (
    "You check the evidence for an unannounced instrument fault on a robotic telescope before a paid report. A false "
    "report costs 150 points; a correct one earns 100 and repairs the instrument. E per hour = measured quality / "
    "quality the program bands allow: about 1 when healthy, low while the instrument is the cause. Weather lowers both "
    "quality and band; an earthquake lowers instrument efficiency in a way that fades night by night and that a "
    "report does not repair.\n"
    'Reply with one JSON object only: {"report": true|false, "reason": "<15 words"}'
)


class Advisor:
    def __init__(self, client, log=lambda text: None):
        self.client = client
        self.log = log
        self.plan_call = None
        self.fault_call = None
        self.plan_applied = True
        self.fault_applied = True
        self.night_date = None
        self.announced: set = set()

    # --- night start ---------------------------------------------------------------------------------

    def start_night(self, night_date: str, tonight: list, bulletin: list, fault_table: dict, wallclock_left: float,
                    wait_seconds: float, summary: dict | None = None, requests_summary: str = ""):
        """Submit both calls; wait up to wait_seconds for them. Returns (plan, fault) answers that are ready."""
        self.night_date = night_date
        self.announced = {n.get("direction") for n in tonight + bulletin if n.get("event_kind") in WEATHER_KINDS}
        notices = {"night": night_date,
                   "forecast_tonight": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in tonight],
                   "bulletin_now": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in bulletin]}
        if summary:
            notices["survey_summary"] = summary
        if requests_summary:
            notices["observation_requests"] = requests_summary
        self.plan_call = self.client.submit("night_plan", NIGHT_PLAN_SYSTEM, notices, wallclock_left)
        self.fault_call = self.client.submit("fault_review", FAULT_REVIEW_SYSTEM, fault_table, wallclock_left)
        self.plan_applied = self.plan_call is None
        self.fault_applied = self.fault_call is None
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for call in (self.plan_call, self.fault_call):
            if call is not None:
                call.wait(deadline - time.monotonic())
        return self.poll()

    def poll(self):
        """(plan, fault) answers that arrived since the last poll; None for each one not (newly) available."""
        plan = fault = None
        if not self.plan_applied and self.plan_call.done():
            self.plan_applied = True
            plan = self._valid_plan(self.client.collect(self.plan_call))
        if not self.fault_applied and self.fault_call.done():
            self.fault_applied = True
            fault = self._valid_fault(self.client.collect(self.fault_call))
        return plan, fault

    def _valid_plan(self, answer):
        if not isinstance(answer, dict) or not isinstance(answer.get("bad_night"), bool):
            return None
        avoid = answer.get("avoid_directions", [])
        if not isinstance(avoid, list):
            return None
        # the model may rank announced weather; it may not close sky that nothing announced
        avoid = sorted({str(d).upper() for d in avoid if str(d).upper() in DIRECTIONS and str(d).upper() in self.announced})
        cap = answer.get("duration_cap")
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = 3600
        if cap not in (1800, 2400, 3600):
            cap = 3600
        terrain = answer.get("terrain")
        if not isinstance(terrain, dict) or not terrain:
            terrain = None
        else:
            terrain = {str(k).upper(): v for k, v in terrain.items()
                       if str(k).upper() in DIRECTIONS and isinstance(v, (int, float))}
            terrain = terrain or None
        gr = str(answer.get("guider_report") or "").lower()
        if gr not in ("report", "no_report"):
            gr = None
        gt = answer.get("guider_time")
        gt = str(gt).strip() if gt is not None else None
        flat = answer.get("flat_windows")
        if isinstance(flat, list):
            flat = [str(w).strip() for w in flat if isinstance(w, str) and "-" in str(w)]
        else:
            flat = []
        rating = str(answer.get("rating") or "").lower()
        if rating not in ("bad", "normal"):
            rating = None
        fgn = answer.get("future_good_nights")
        try:
            fgn = int(fgn)
        except (TypeError, ValueError):
            fgn = 2
        if fgn not in (0, 1, 2):
            fgn = 2
        return {"bad_night": answer["bad_night"], "avoid_directions": avoid,
                "terrain": terrain, "guider_report": gr, "guider_time": gt,
                "flat_windows": flat, "rating": rating, "future_good_nights": fgn,
                "duration_cap": cap, "reason": str(answer.get("reason", ""))[:80]}

    @staticmethod
    def _valid_fault(answer):
        if not isinstance(answer, dict):
            return None
        try:
            p = float(answer.get("fault_likely"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= p <= 1.0:
            return None
        return {"fault_likely": p, "reason": str(answer.get("reason", ""))[:80]}

    # --- paid report confirmation ----------------------------------------------------------------------

    def confirm_report(self, evidence: dict, wallclock_left: float, wait_seconds: float):
        """True / False from the model, or None (no answer in time: the rule decides)."""
        call = self.client.submit("confirm_report", CONFIRM_SYSTEM, evidence, wallclock_left)
        if call is None:
            return None
        call.wait(wait_seconds)
        answer = self.client.collect(call)
        if isinstance(answer, dict) and isinstance(answer.get("report"), bool):
            return answer["report"]
        return None
