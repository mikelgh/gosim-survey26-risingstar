#!/usr/bin/env python3
"""Agent Observer v4 reference agent ("pro"). Standard library only, participant-agent-protocol-v4.

One JSON object per line on stdin, one per line on stdout, logs on stderr.

What it does (details in README.md and planner.py):

1. Planning: every decision picks pointing, fibre assignment, duration and program together, maximising
   expected gain minus a price for telescope time (gain - lambda * T). Required targets and observation
   requests enter as probability-weighted bonuses.
2. Program: the band level is fitted to saturated hits, which show the program multiplier exactly.
3. Instrument faults: E = (quality level) / (band level). Weather moves both, a fault only the first;
   when E stays low the agent reports. Free false reports are spent early; each low episode is probed
   once; paid probes need two low nights.
4. Pace: the search level adapts to the measured cost per decision so a 4-month card fits the wall clock.
5. Model (advisor.py): at every night start a night plan (forecast + bulletin -> bad night, sectors to avoid)
   and a fault review (own quality table -> how likely a fault is, which gates paid reports); before a paid
   report the model confirms or vetoes. Calls run in the background; without an API key the agent exits,
   except when the platform sets OBSERVER_MODEL_DISABLED=1 (an evaluation without a model): then rules only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

from advisor import Advisor, decode_intel, extract_reason, station_cipher
from llm_client import LLMClient, api_key, load_dotenv
from planner import Planner
from skymath import format_utc, parse_utc

WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}
BAD_KINDS = {"rain", "storm", "overcast", "haze"}
PROTOCOL = "participant-agent-protocol-v4"


def _env(name: str, default: float) -> float:
    return type(default)(os.environ.get(f"PRO_{name}", default))


# --- fault reporting (see _fault_verdict) ---
E_LOW_FREE = _env("E_LOW_FREE", 0.9)      # free probe: E below this in 3 of the last 4 hours
E_LOW_FREE2 = _env("E_LOW_FREE2", 0.9)    # ... threshold while both free probes are left
E_FREE_HOURS = _env("E_FREE_HOURS", 4)
E_LOW = _env("E_LOW", 0.85)               # paid probe: low hours must span two nights ...
E_PAID_LOW = _env("E_PAID_LOW", 0.75)     # ... the median of the last 12 hourly E below this ...
E_PAID_HOURS = _env("E_PAID_HOURS", 12)
E_RECOVER = _env("E_RECOVER", 0.95)       # after a false probe, wait until E is back above this
MAX_PAID_FALSE = _env("MAX_PAID", 6)
PERSIST_NIGHTS = _env("PERSIST_NIGHTS", 3)
E_PAID_STEP = _env("E_PAID_STEP", 0.05)   # ... minus this per paid false probe so far
MAX_FALSE_REPORTS = 8
# The participant guide: an earthquake (announced in the bulletin) lowers instrument efficiency, the loss fades
# night by night, and a report does not repair it. So E drops right after an earthquake are not reportable, and
# while its effect may last only a new step down in E (a fresh drop from the preceding hours) is fault evidence.
QUAKE_HOLD_HOURS = _env("QUAKE_HOLD_HOURS", 12.0)   # no probes this long after an earthquake notice appears
QUAKE_STEP = _env("QUAKE_STEP", 0.8)                # step: median E of the last 3 rows < this x the 9 rows before
QUAKE_TAIL_HOURS = _env("QUAKE_TAIL_HOURS", 24.0)   # the earthquake period lasts this long after its last notice
PAID_SPACING_HOURS = 20.0
MIN_REPORT_SPACING_HOURS = 2.0
# --- pace ---
PACE_SAFETY = _env("PACE_SAFETY", 0.75)
# --- model ---
MODEL_WAIT_MAX = _env("MODEL_WAIT_MAX", 20.0)   # longest wait for the night's model answers (s)
MODEL_FAULT_HIGH = _env("MODEL_FAULT_HIGH", 0.6)  # fault review at or above this: report more readily tonight
MODEL_FAULT_LOW = _env("MODEL_FAULT_LOW", 0.15)   # ... at or below this: paid reports need the strongest evidence
SCALE_STEP = _env("SCALE_STEP", 0.7)
MODEL_FREE_PROBE = _env("MODEL_FREE_PROBE", 0)   # 1: a high fault review may also spend a free probe on a low scale            # with a likely fault: report when scale stays below this x ref
FIXED_LEVEL = _env("FIXED_LEVEL", -1)     # development only: pin the search level (deterministic runs)   # spend at most this share of the remaining wall clock


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class RulesOnly:
    """The advisor's interface without a model (OBSERVER_MODEL_DISABLED=1): every rule default stands."""
    night_date = None

    def start_night(self, night_date, *_args):
        self.night_date = night_date
        return None, None

    def poll(self):
        return None, None

    def confirm_report(self, *_args):
        return None


def model_disabled() -> bool:
    """The platform sets OBSERVER_MODEL_DISABLED=1 for an evaluation started with 「本次不提供模型」 / --no-model."""
    return os.environ.get("OBSERVER_MODEL_DISABLED") == "1"


class ObserverAgent:
    def __init__(self, init: dict, rules_only: bool = False):
        started = time.monotonic()
        self.planner = Planner(init, log=log)
        self.rules_only = rules_only
        self.client = None if rules_only else LLMClient(log=log)
        self.advisor = RulesOnly() if rules_only else Advisor(self.client, log=log)
        self.model_wait = 0.0                    # wall seconds spent waiting for the model (not planning cost)
        self.fault_likely = None                 # tonight's model estimate that an instrument fault is active
        self.scale_hours: dict = {}              # hour -> [planner.scale samples] (for the model's fault table)
        self.start = parse_utc(init["survey"]["start_utc"])
        self.forecast_notices: list = []
        self.night_seen = None
        self.req_history: list = []          # (night_index, required done) for the model's survey summary
        self.observes = 0
        # fault reporting state
        reporting = init["scoring"].get("reporting", {})
        self.free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0
        self.false_since_correct = 0
        self.paid_false = 0
        self.last_report_hours = -1e9
        self.ref_from_hours = -1e9
        self.episode_blocked = False
        self.blocked_at_hour = -1
        self.quake_on = False
        self.quake_onset_hours = -1e9
        self.quake_last_hours = -1e9
        # v20i: guide-camera maintenance from the handover notes
        self.guider_pending: str | None = None
        self.guider_reported = False
        # pace state
        self.cost_ema = [0.0, 0.0, 0.0, 0.0]     # CPU seconds per observe decision at each search level
        self.wall_ema = [0.0, 0.0, 0.0, 0.0]     # real seconds per observe decision (own turn, model waits excluded)
        self.turn_end = None
        self.decisions = 0
        self.engine_ema = None
        self.sim_step_ema = None
        self.last_now = None
        log(f"pro: {len(self.planner.ids)} targets, {sum(self.planner.required)} required, "
            f"{len(self.planner.nights)} nights; init {time.monotonic() - started:.2f}s; model {self.client.model if self.client else 'none (rules only)'}")

    # --- decision loop ------------------------------------------------------------------------------

    def respond(self, payload: dict) -> dict:
        started = time.monotonic()
        cpu_started = time.process_time()
        if self.turn_end is not None:   # engine time between our turns (charged only by the old real-time clock)
            gap = started - self.turn_end
            if 0.0 <= gap < 5.0:
                self.engine_ema = gap if self.engine_ema is None else 0.95 * self.engine_ema + 0.05 * gap
        model_before = self.model_wait
        level = self.planner.fast_level
        action = self._respond(payload)
        # the platform charges CPU time inside our turns; waiting for the model is free of CPU, so keep it
        # out of the real-time estimate as well
        cpu = time.process_time() - cpu_started
        wall = time.monotonic() - started - (self.model_wait - model_before)
        if action.get("action") == "observe" and level < 4:
            for ema, cost in ((self.cost_ema, cpu), (self.wall_ema, wall)):
                c = ema[level]
                ema[level] = cost if c == 0.0 else 0.9 * c + 0.1 * cost
                for k in range(level + 1, 4):   # cheaper levels not measured yet: a third of the level above
                    if ema[k] == 0.0 or ema[k] > ema[k - 1]:
                        ema[k] = ema[k - 1] / 3.0
        self.turn_end = time.monotonic()
        return action

    def _respond(self, payload: dict) -> dict:
        now = parse_utc(payload["now_utc"])
        if self.last_now is not None and self.planner.current_night(now) is not None:
            step = (now - self.last_now).total_seconds()
            if 0 < step <= 3600:
                self.sim_step_ema = step if self.sim_step_ema is None else 0.95 * self.sim_step_ema + 0.05 * step
        self.last_now = now
        hours = (now - self.start).total_seconds() / 3600.0
        planner = self.planner
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self.forecast_notices = message.get("notices", [])
        last = payload.get("last_result") or {}
        if last.get("action") == "report":
            self._on_report_result(last, hours)
        planner.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        quake = any(kind == "earthquake" for kind, _ in planner.notices)
        if quake and not self.quake_on:
            self.quake_onset_hours = hours
            log(f"pro: earthquake notice at {payload['now_utc']}")
        self.quake_on = quake
        if quake:
            self.quake_last_hours = hours
        planner.on_requests(payload.get("active_requests", []))
        planner.on_result(payload.get("last_result"), now, hours)
        self._pace(payload, now)

        night = planner.current_night(now)
        if night is None:
            nxt = planner.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night
        if self.night_seen != night_index:
            self.night_seen = night_index
            self._night_advice(night_start, payload, hours)
        else:
            self._apply_advice(*self.advisor.poll())
        self.scale_hours.setdefault(int(hours), []).append(self.planner.scale)
        if (night_end - now).total_seconds() < planner.min_exposure:
            nxt = planner.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}
        if planner.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}
        report = self._maybe_guider_report(now)
        if report is not None:
            return report
        report = self._maybe_report(hours, payload)
        if report is not None:
            return report
        action = planner.plan(now, night_end, night_index, hours)
        if action is None:
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start), "reason": "nothing useful is up"}
        self.observes += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.planner.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into)))

    # --- pace ---------------------------------------------------------------------------------------

    def _clock(self, payload: dict):
        """(CPU seconds left, real seconds left, fair clock?) from the request's wallclock block.

        Fair clock (current platform): the budget is normalized CPU time inside our turns, and
        remaining_real_cpu_seconds converts it to this machine's CPU seconds; a separate real-time cap
        (wall_remaining_seconds) only guards against runaway runs. Older runners count real time only."""
        wall = payload.get("wallclock") or {}
        if "remaining_real_cpu_seconds" in wall:
            return (float(wall["remaining_real_cpu_seconds"]), float(wall.get("wall_remaining_seconds", 1e9)), True)
        remaining = float(wall.get("remaining_seconds", 1e9))
        return remaining, remaining, False

    def _decisions_left(self, now) -> float:
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in self.planner.nights if end > now)
        return max(1.0, night_seconds / (self.sim_step_ema or 900.0))   # daytime waits cost nothing

    def _pace(self, payload: dict, now) -> None:
        """Pick the search level from the measured cost per decision and the decisions still to come."""
        cpu_left, wall_left, fair = self._clock(payload)
        decisions_left = self._decisions_left(now)
        engine = self.engine_ema or 0.0
        cpu_budget = PACE_SAFETY * cpu_left / decisions_left if fair else 1e9
        wall_budget = PACE_SAFETY * wall_left / decisions_left - engine
        # estimates of levels not used for a while decay, so the agent climbs back up and re-measures them
        self.decisions += 1
        if self.decisions % 50 == 0:
            for k in range(4):
                if k != self.planner.fast_level:
                    self.cost_ema[k] *= 0.85
                    self.wall_ema[k] *= 0.85
        if FIXED_LEVEL >= 0:
            self.planner.fast_level = FIXED_LEVEL
            return
        level = 0
        while level < 3 and (self.cost_ema[level] > cpu_budget or self.wall_ema[level] > wall_budget):
            level += 1
        if min(cpu_left, wall_left) < 15.0:
            level = 4
        if level != self.planner.fast_level:
            log(f"pro: pace level {level} (cpu budget {min(cpu_budget, 99) * 1000:.0f} ms, wall budget {wall_budget * 1000:.0f} ms, "
                f"cpu costs {[round(c * 1000) for c in self.cost_ema]} ms, {decisions_left:.0f} decisions left)")
            self.planner.fast_level = level

    # --- model stages (advisor.py): night plan and fault review at every night start --------------------

    def _night_advice(self, night_start, payload: dict, hours: float) -> None:
        """Night start: rule defaults first, then the two model calls (night plan, fault review)."""
        night_date = (night_start - timedelta(hours=12)).date().isoformat()
        tonight = [n for n in self.forecast_notices if night_date in n.get("nights", [])]
        bulletin = (payload.get("latest_bulletin") or {}).get("notices", [])
        # rule defaults, kept when the model gives no valid answer
        self.planner.bad_forecast = any(n.get("direction") == "ALL" and n.get("event_kind") in BAD_KINDS for n in tonight)
        self.planner.extra_avoid = set()
        self.planner.duration_cap = 3600
        self.fault_likely = None
        req_total = sum(1 for i in range(len(self.planner.required)) if self.planner.required[i])
        req_done = sum(1 for i in range(len(self.planner.required)) if self.planner.required[i] and self.planner.factor[i] >= 0.5)
        self.req_history.append((self.night_seen, req_done))
        summary = {"nights_elapsed": int(self.night_seen or 0) + 1, "required_done": req_done,
                   "required_total": req_total, "recent_required_done": [d for _, d in self.req_history[-3:]]}
        # v20g: feed the observation-request notes to the model. They carry coded weather ("暖气又坏了" = all-sky
        # rain), blocked sectors, guide-camera maintenance windows and flat-lamp flat tests the structured
        # notices do not contain; the plan model reads them to correct bad_night / avoid_directions.
        raw_reqs = payload.get("active_requests", [])
        if not isinstance(raw_reqs, list):
            raw_reqs = []
        req_texts = []
        ciph = {"bad": None, "avoid": set(), "flat": [], "note": []}
        for r in raw_reqs:
            if not isinstance(r, dict) or not r.get("reason"):
                continue
            # v20i: handover notes are 300-4400 chars of coded dialogue; a prefix cut drops the
            # guide-camera maintenance and flat-test windows that sit in the middle/end. Extract
            # the informative lines instead of slicing the first 800 chars.
            txt = extract_reason(str(r["reason"]), max_chars=2000)
            # v20i2: deterministic Caesar/solfege decodes, tagged non-local lines.
            dcd = decode_intel(str(r["reason"]))
            if dcd:
                txt += "\n  解码情报:\n" + "\n".join("   " + dl for dl in dcd.splitlines())
            # v20i7: deterministic weather ciphers (ground truth), independent of the LLM.
            c = station_cipher(str(r["reason"]))
            if c["bad"]:
                ciph["bad"] = True
            ciph["avoid"] |= c["avoid"]
            ciph["flat"] += c["flat"]
            ciph["note"] += c["note"]
            req_texts.append(f"[{r.get('request_id', '?')} dl={r.get('deadline_utc', '?')}] {txt}")
        if ciph["note"]:
            requests_summary_prefix = "确定性暗号解码: " + "; ".join(sorted(set(ciph["note"]))) + "\n"
        else:
            requests_summary_prefix = ""
        requests_summary = requests_summary_prefix + "\n".join(req_texts)
        try:
            with open("v20g_req_debug.log", "a", encoding="utf-8") as f:
                f.write(f"active={len(raw_reqs)} with_reason={len(req_texts)} "
                        f"sumlen={len(requests_summary)} head={requests_summary[:120]!r}\n")
        except OSError:
            pass
        left = self._clock(payload)[1]
        started = time.monotonic()
        answers = self.advisor.start_night(night_date, tonight, bulletin, self._fault_table(hours), left,
                                           self._model_wait_budget(payload), summary, requests_summary)
        self.model_wait += time.monotonic() - started
        self._apply_advice(*answers, ciph)

    def _model_wait_budget(self, payload: dict) -> float:
        """How long a night start may wait for the model. Waiting costs no CPU budget, only real time: use half
        of the real time the planner and the engine will not need, spread over the nights left."""
        _, wall_left, _ = self._clock(payload)
        now = self.last_now
        nights_left = max(1, sum(1 for _, end in self.planner.nights if end > now))
        per_decision = max(self.wall_ema[self.planner.fast_level], 0.05) + (self.engine_ema or 0.02)
        spare = wall_left - 1.5 * self._decisions_left(now) * per_decision - 60.0
        return max(0.0, min(MODEL_WAIT_MAX, 0.5 * spare / nights_left))

    def _apply_advice(self, plan, fault, ciph=None) -> None:
        if plan is not None:
            # v20i7: deterministic station ciphers override the probabilistic LLM judgement.
            bad = plan["bad_night"]
            if ciph and ciph.get("bad"):
                bad = True
            avoid = set(plan["avoid_directions"])
            if ciph and ciph.get("avoid"):
                avoid |= ciph["avoid"]
            self.planner.bad_forecast = bad
            self.planner.extra_avoid = avoid
            self.planner.duration_cap = int(plan.get("duration_cap", 3600))
            # v20i8 P2: cross-night sprint — how many good nights remain in the next two.
            self.planner.future_good_nights = int(plan.get("future_good_nights", 2))
            # v20i: handover intel: per-sector blocking elevation, flat junk windows, guide-camera report clock.
            flat = list(plan.get("flat_windows") or [])
            if ciph and ciph.get("flat"):
                for f in ciph["flat"]:
                    if f not in flat:
                        flat.append(f)
            self.planner.set_night_intel(terrain=plan.get("terrain"), flat=flat,
                                         guider_time=plan.get("guider_time"))
            if plan.get("guider_report") == "report" and plan.get("guider_time"):
                self.guider_pending = plan["guider_time"]
            elif plan.get("guider_report") != "report":
                self.guider_pending = None
            log(f"llm night plan {self.advisor.night_date}: bad={bad} avoid={sorted(avoid)} "
                f"terrain={plan.get('terrain')} guider={plan.get('guider_report')}/{plan.get('guider_time')} "
                f"flat={flat} rating={plan.get('rating')} cap={plan.get('duration_cap')} ({plan['reason']})")
        if fault is not None:
            self.fault_likely = fault["fault_likely"]
            log(f"llm fault review {self.advisor.night_date}: fault_likely={fault['fault_likely']:.2f} ({fault['reason']})")

    def _maybe_guider_report(self, now) -> dict | None:
        """v20i: the handover notes say the guide camera is serviced at a clock time tonight and must be
        reported at that moment (until reported, every frame is junk). Fire one report when the clock matches."""
        if not self.guider_pending or self.guider_reported:
            return None
        hh, mm = self.guider_pending.split(":")
        target = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        if abs((now - target).total_seconds()) <= 900:
            self.guider_reported = True
            self.guider_pending = None
            log(f"pro: guide-camera maintenance window {self.guider_pending} reached -> report")
            return {"action": "report", "reason": "guide camera maintenance window: frames junk until reported"}
        return None

    def _scale_ref(self) -> float:
        """Usual clear-sky scale since the last repair: 75th percentile of the hourly medians."""
        values = sorted(sorted(v)[len(v) // 2] for h, v in self.scale_hours.items() if h >= self.ref_from_hours and v)
        return values[(3 * len(values)) // 4] if len(values) >= 4 else 1.0

    def _fault_table(self, hours: float) -> dict:
        """The evidence the fault review reads: the last ~30 observed hours."""
        e_by_hour = {hour: sorted(v)[len(v) // 2] for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours}
        rows = []
        for hour in sorted(self.scale_hours)[-30:]:
            if hour < self.ref_from_hours:
                continue
            v = sorted(self.scale_hours[hour])
            stamp = (self.start + timedelta(hours=hour)).strftime("%m-%dT%H")
            rows.append([stamp, round(e_by_hour[hour], 2) if hour in e_by_hour else None, round(v[len(v) // 2], 2)])
        notices = sorted({f"{kind} {direction}" for kind, direction in self.planner.notices})
        return {"columns": ["utc_hour", "E", "scale"], "rows": rows, "ref": round(self._scale_ref(), 2),
                "notices_now": notices,
                "hours_since_earthquake_notice_began": None if self.quake_onset_hours < -1e8 else round(hours - self.quake_onset_hours, 1),
                "free_false_reports_left": max(0, self.free_left()), "paid_false_reports_so_far": self.paid_false,
                "correct_reports_so_far": self.correct_reports,
                "hours_since_last_report": None if self.reports == 0 else round(hours - self.last_report_hours, 1)}

    # --- instrument faults ---------------------------------------------------------------------------

    def _maybe_report(self, hours: float, payload: dict):
        """Report (probe) when the quality level stays below what the program bands allow.

        A report costs no time, its answer arrives at once, and the first false reports after each correct
        one are free: spend free probes readily, paid ones only on strong, lasting evidence."""
        if self.false_reports >= MAX_FALSE_REPORTS or hours - self.last_report_hours < MIN_REPORT_SPACING_HOURS:
            return None
        if hours - self.quake_onset_hours < QUAKE_HOLD_HOURS:
            return None   # the earthquake explains the drop; a report would not repair it
        if self._fault_verdict(hours, payload) and self._model_agrees(hours, payload):
            self.last_report_hours = hours
            self.reports += 1
            log(f"pro: report at {payload['now_utc']} (quality below what the program bands allow), free left {self.free_left()}")
            return {"action": "report", "reason": "quality level below what the program bands allow"}
        return None

    def _fault_verdict(self, hours: float, payload: dict) -> bool:
        """Hourly E = quality level / band level (planner.e_hours). 1 = consistent; a fault keeps E low."""
        rows = [(hour, night, sorted(v)[len(v) // 2]) for hour, night, v in self.planner.e_hours if hour >= self.ref_from_hours]
        if rows and int(hours) != getattr(self, "_logged_hour", None):
            self._logged_hour = int(hours)
            log(f"pro: E {payload['now_utc']} {rows[-1][2]:.2f} scale {self.planner.scale:.3f} band {self.planner.band_level or 0:.3f}")
        if len(rows) < E_FREE_HOURS or rows[-1][0] < int(hours) - 1:
            return False
        if QUAKE_STEP > 0 and hours - self.quake_last_hours < QUAKE_TAIL_HOURS:
            if len(rows) < 8:
                return False
            last = sorted(e for _, _, e in rows[-3:])[1]
            prev = sorted(e for _, _, e in rows[-12:-3])
            if not (last < QUAKE_STEP * prev[len(prev) // 2] and rows[-3][0] >= int(hours) - 4):
                return False
            if self.episode_blocked and rows[-3][0] <= self.blocked_at_hour:
                return False   # the step that was already probed, not a new one
            self.episode_blocked = False   # a new step is a new episode
        if self.episode_blocked:
            # this low episode was probed already and was not a fault: wait for a recovery first
            last4 = rows[-4:]
            if sum(1 for _, _, e in last4 if e >= E_RECOVER) >= 3 and rows[-1][0] > self.blocked_at_hour:
                self.episode_blocked = False
                log(f"pro: quality recovered at {payload['now_utc']}; probing re-armed")
            else:
                return False
        likely = self.fault_likely
        if MODEL_FREE_PROBE and likely is not None and likely >= MODEL_FAULT_HIGH and self.free_left() > 0 and self._scale_step(hours):
            # off by default: on the practice cards it spent free probes on unannounced weather
            log(f"pro: model-flagged fault (likely {likely:.2f}) and scale below {SCALE_STEP} x ref")
            return True
        if self.free_left() > 0:
            last = rows[-E_FREE_HOURS:]
            threshold = E_LOW_FREE2 if self.free_left() >= 2 else E_LOW_FREE
            low = sum(1 for _, _, e in last if e < threshold)
            return low >= E_FREE_HOURS - 1 and last[-1][2] < threshold and last[-1][0] - last[0][0] <= E_FREE_HOURS + 3
        if self.paid_false >= MAX_PAID_FALSE or hours - self.last_report_hours < PAID_SPACING_HOURS:
            return False
        last = rows[-E_PAID_HOURS:]
        values = sorted(e for _, _, e in last)
        nights = {night for _, night, e in last if e < E_LOW}
        # a fault never goes away on its own: three low nights in a row justify a probe whatever the bar
        by_night: dict = {}
        for _, night, e in rows:
            by_night.setdefault(night, []).append(e)
        nights_seq = sorted(by_night)[-PERSIST_NIGHTS:]
        if (len(nights_seq) == PERSIST_NIGHTS and nights_seq[-1] - nights_seq[0] <= PERSIST_NIGHTS
                and all(len(by_night[n]) >= 3 and sorted(by_night[n])[len(by_night[n]) // 2] < E_LOW for n in nights_seq)
                and hours - self.last_report_hours >= 40.0):
            return True
        if likely is not None and likely <= MODEL_FAULT_LOW:
            return False   # the fault review sees weather, not a fault: only the persistence rule above may report
        # each paid false probe raises the bar for the next one
        paid_low = E_PAID_LOW - E_PAID_STEP * self.paid_false
        return (len(last) == E_PAID_HOURS and values[len(values) // 2] < paid_low and len(nights) >= 2
                and all(e < E_LOW for _, _, e in last[-3:]))

    def _scale_step(self, hours: float) -> bool:
        """The last 3 observed hours all sit below SCALE_STEP x the usual clear-sky scale."""
        recent = [sorted(v)[len(v) // 2] for h, v in sorted(self.scale_hours.items()) if h >= self.ref_from_hours and v][-3:]
        return len(recent) == 3 and max(recent) < SCALE_STEP * self._scale_ref()

    def _model_agrees(self, hours: float, payload: dict) -> bool:
        """Paid probes only: the model looks at the evidence first and may veto. Free probes cost nothing, so they
        never wait for it. No answer in time: the rule's decision stands."""
        if self.free_left() > 0:
            return True
        rows = [(hour, sorted(v)[len(v) // 2]) for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours][-24:]
        evidence = {"hourly_E_last_24h": [round(e, 2) for _, e in rows], "fault_table": self._fault_table(hours),
                    "paid_false_reports_so_far": self.paid_false, "correct_reports_so_far": self.correct_reports}
        started = time.monotonic()
        left = self._clock(payload)[1]
        verdict = self.advisor.confirm_report(evidence, left, min(30.0, 2.0 * self._model_wait_budget(payload)))
        self.model_wait += time.monotonic() - started
        if verdict is False:
            log(f"pro: model vetoed a paid report at {payload['now_utc']}")
            self.last_report_hours = hours
            return False
        return True

    def free_left(self) -> int:
        return self.free_allowance - self.false_since_correct

    def _on_report_result(self, result: dict, hours: float) -> None:
        if result.get("correct"):
            log(f"pro: report correct, fault repaired (delta {result.get('score_delta')})")
            self.correct_reports += 1
            self.false_since_correct = 0
            self.planner.forget_quality_history()
            self.ref_from_hours = hours
        else:
            self.false_since_correct += 1
            self.false_reports += 1
            self.episode_blocked = True
            self.blocked_at_hour = int(hours)
            if self.false_since_correct > self.free_allowance:
                self.paid_false += 1
            log(f"pro: report false (delta {result.get('score_delta')}); free left {self.free_left()}")


def main() -> int:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    rules_only = model_disabled()
    if rules_only:
        log("pro: OBSERVER_MODEL_DISABLED=1, running rules only (no model calls)")
    elif not api_key():
        print("missing API key: set OPENAI_API_KEY", file=sys.stderr, flush=True)
        return 2
    agent = None
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("message_type")
        if message.get("protocol_version") != PROTOCOL:
            log(f"pro: unexpected protocol {message.get('protocol_version')!r}")
        if kind == "initialize":
            agent = ObserverAgent(message["payload"], rules_only=rules_only)
        elif kind == "decision_request":
            try:
                action = agent.respond(message["payload"])
            except Exception as exc:  # noqa: BLE001 - never crash the run: wait one slot instead
                log(f"pro: error {type(exc).__name__}: {exc}; waiting one slot")
                action = {"action": "wait", "duration_seconds": 900, "reason": "internal error"}
            action.setdefault("decision_source", "rules" if rules_only else "llm-advised")
            print(json.dumps({"protocol_version": PROTOCOL, "message_type": "decision_response",
                              "decision_sequence": message["decision_sequence"], **action}, separators=(",", ":")), flush=True)
        elif kind == "finish":
            payload = message.get("payload", {})
            log(f"pro finished: termination_reason={payload.get('termination_reason')} "
                f"observes={agent.observes if agent else 0} reports={agent.reports if agent else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
