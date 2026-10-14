# gosim-survey26-risingstar

**Risingstar team's final agent for [GOSIM Survey26](https://create.gosim.org/survey26/) — the Agentic Observer Hackathon.**

This repository contains the exact version we submitted as our **final entry** (evaluation id `c34edba0`, locked before the deadline), published as-is for the open-source showcase.

The task: build a fully autonomous agent that runs a 4-month spectroscopic survey on a simulated
telescope (16 fibres, VLT-like site at Paranal). The agent talks to the platform over a JSONL
protocol (`participant-agent-protocol-v4`), receives one `decision_request` per observation slot,
and must decide **pointing, fibre assignment, exposure duration and observing program** — every
night, under weather, instrument faults, earthquakes, and hidden events announced only through
operator hand-over notes. Standard library only; no third-party packages.

## Results

| Stage | Rank | Score |
|---|---|---|
| Final (4 hidden cards E–H, 3 runs averaged per card) | **#48 / 103** | **27,090** |
| Online phase (public cards A–D) | #59 / 104 | 29,512.86 |
| Official baseline example | #75 / 103 | 24,134 (**we are +12%**) |

Per-card finals: **E 33,696 · F 25,758 · G 19,059 · H 29,847** (top team: 33,576).

Honest post-mortem: our generic engine is competitive (E 33,696 was near the first tier), but
*required-target-dense* cards punished us hard — on card G we left 95 required targets uncovered
(−4,750, our single biggest loss). The gap to the podium is required-target coverage, not
exposure efficiency.

## Architecture

```
agent.py        protocol loop: JSONL in/out, night lifecycle, fault reporting state machine
planner.py      core search: pointing + fibres + duration + program in ONE optimisation
advisor.py      LLM-driven stages: night plan, fault review, paid-report confirmation
llm_client.py   OpenAI-compatible chat client (background calls, JSON mode, timeouts)
skymath.py      astronomy helpers: sidereal math, altaz, hour angle, fibre grid
observer.project.json   platform manifest (entry point + environment)
```

### Planner (planner.py)

Every decision is one joint optimisation, not a sequence of greedy picks:

1. Visible targets with remaining gain are shortlisted by a cheap proxy ranking.
2. The gain of exposing target *i* for duration *T* is
   `weight * (reach(T) * program multiplier - best so far)`
   `+ REQUIRED_BONUS * P(required target reaches factor 0.5)`
   `+ request bonus * P(request target reaches its threshold)`,
   with an urgency factor for targets near the end of their visibility window.
3. Candidate fields: the best targets centred on each of the 16 fibres plus the densest patches
   of remaining sky. The winning field maximises **total gain − λ·T**, where λ is the price of
   telescope time, scaled by how scarce the card's remaining time is.
4. Programs are chosen from a band level *fitted to saturated hits* — the engine's own
   saturated observations reveal the program multiplier exactly, so the agent uses its own
   results as a sensor instead of hard-coding program values.
5. On hard-mode cards the command is corrected by a pointing offset learned from hit/miss patterns.

The planner only uses the public catalogue, the public score formula, bulletins and its own
observation history.

### LLM usage (advisor.py)

The rules require LLM-driven decision stages; we use three, all as **background advisors with
rule-based fallback** (an invalid or late answer never blocks planning — the rule value stays):

1. **night_plan** (natural-language understanding + plan adaptation): reads tonight's forecast
   and the current bulletin, decides whether tonight is a bad night for faint must-observe
   targets and which sky sectors to avoid.
2. **fault_review** (data parsing + action decision): reads the agent's own hour-by-hour
   quality table and judges how likely an unannounced instrument fault is; the answer sets how
   readily the agent reports a fault tonight.
3. **report confirmation**: before spending a *paid* fault report, the model confirms or vetoes.

Calls run in the background (`llm_client.Call`); the agent waits only as long as the wall clock
allows. Any OpenAI-compatible chat endpoint works (`OPENAI_BASE_URL` / `OPENAI_API_KEY` /
`OPENAI_MODEL`; the final run used `qwen-plus`). With no API key the agent exits — unless the
platform sets `OBSERVER_MODEL_DISABLED=1`, in which case it runs on rules only.

## Cracking the hand-over-note ciphers

On the A1–D1 expansion cards, each observation request carries a
Chinese "operator hand-over note" whose key information is **not plaintext** — it is a codebook
of euphemisms and two real ciphers:

| Code in the note | Actual meaning |
|---|---|
| 「暖气又坏了」("the heater broke again") | all-day rain, dome closed — bad night |
| 「打印机卡纸」("printer jam") | thick cloud, cannot shoot |
| 「咖啡机没水」("coffee machine is empty") | thin cloud, still shootable |
| 「门禁卡」("access card") + a direction | that compass sector is wind-closed tonight |
| Caesar cipher, shift = the note's date | encrypted notices (e.g. guide-camera maintenance windows) |
| Solfège encoding (do=1 … rest=0) | clock times hidden as musical notes |
| flat-field / dome-cover time slots | junk frames, but *not* reportable — don't waste a report |

No ground truth was given. We cracked it the boring way: aligning real notes with real
observation outcomes (empirical bad / wind / flat classification), brute-forcing the Caesar
shift against the note's date, and a lookup table for solfège. End state: **9/12 Caesar notices
decoded, 26/26 weather euphemisms correct**.

The final version uses a **two-layer** design (advisor.py):

1. **Deterministic decode wins over the model.** `station_cipher` / `decode_intel` turn weather
   euphemisms into hard bad-night / avoid-sector / flat-window facts that **override** the LLM's
   answer, decode Caesar + solfège into maintenance clocks, and tag lines that belong to *other*
   stations. Rationale: we once caught the LLM hallucinating a 105-minute flat window that idled
   the planner all night — anything decodable by rules is not left to the model.
2. **The LLM still helps where rules can't.** It reads the decoded excerpts (codebook included
   in its system prompt) for the probabilistic parts of the night plan, e.g. long-range quality
   ratings that exist only inside the notes.

Measured effect: decoding + tagging was worth **≈ +615** across cards A–D; acting correctly on
the guide-camera maintenance windows (reporting the fault) was worth **≈ +1,000** on card D1;
the deterministic override removed the LLM's randomness on three decision channels and made the
final version stable across repeated evaluations.

Honest note: decoding teaches the agent "don't shoot junk, report real faults" — it does not
help push faint required targets over the scoring line, which remained our biggest gap.

## Lessons learned

What worked:

- **One joint optimisation beats staged greed.** Choosing pointing, fibres, duration and program
  against a time price (λ) in a single objective was the single biggest score jump.
- **Use the engine as a sensor.** Fitting the program multiplier from our own saturated hits and
  the pointing offset from hit/miss patterns removed a whole class of hard-coded assumptions.
- **LLM as background advisor, not pilot.** Concurrent calls with strict timeouts and rule
  fallbacks gave us LLM upside with zero catastrophic downside across ~100 evaluations.
- **Spend free probes early, price paid ones.** The fault-report state machine (free probes,
  two-night paid threshold, earthquake hold-off) survived every hidden card.

What didn't (yet):

- **Required-target-dense cards.** "Crossing-the-line" frame allocation for dense required
  clusters was the one lever we never cracked; the same failure recurred from practice card B1
  to final card G. Coverage-first heuristics we tried traded too much best-frame score for it.
- **Parameter search on public cards can overfit.** The hidden finals rewarded the strategies
  that generalised (dynamic forecast/bulletin reasoning) over the ones tuned to specific cards.

## Running locally

The agent targets the GOSIM Survey26 platform runtime. To experiment locally you need the
official starter kit / examples (simulator + local scorer) from the
[GOSIM Survey26 examples repository](https://github.com/gosimfoundation/hackathon-survey26),
then point it at this directory. Configure the model via a `.env` file next to the sources
(see `.env.example`), or set `OBSERVER_MODEL_DISABLED=1` to run rules-only.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- The GOSIM Survey26 organisers, for the starter kit, public cards and evaluation infrastructure.
- The official Python reference example, which we used as our first baseline opponent.
