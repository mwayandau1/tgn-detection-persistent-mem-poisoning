# Candidate venues (brief v2, Section 3), compiled 14 October 2026

Every date was looked up on 14 October 2026; none is from memory. **Official** means the
venue's own page was read. **Aggregator** means a third-party listing, to be confirmed on the
official page before planning around it. Fit is judged against the current evidence: one
threat model, single seed, and a detector that ties engineered-feature oracles on AUROC.

## Open now, in deadline order

| Venue | Deadline | Source | Length | Review | Template / citations | Artefacts | Fit |
|---|---|---|---|---|---|---|---|
| WISTP 2027 (Djerba, 19 to 21 Mar 2027; Springer LNCS, hybrid) | 9 Dec 2026, 23:59 AoE; one listing says 2 Jan 2027 | Aggregators disagree; official site not found | Full 16 pp LNCS; short 8 pp; SoK accepted | Double-blind | Springer LNCS (splncs04), numbered | Not stated | Good: 2027 edition emphasises LLM and agentic-AI security, including memory poisoning. Smaller venue; suits the current evidence. |
| ACM AsiaCCS 2027, round 2 (Macau, 12 to 16 Jul 2027) | 11 Dec 2026 AoE; early reject 5 Feb, notification 31 Mar 2027 | Official CFP | 12 pp, plus up to 10 pp of bibliography and appendices | Double-blind; author list frozen at submission | ACM sigconf two-column; style unspecified (ACM numbered by default) | Required half-page Open Science appendix with an anonymous artefact link; Ethics appendix if relevant; ORCIDs | Stronger venue (AI Security track). Competitive only with E2 seeds and ideally a second threat model (E5). |
| ICLR 2027 workshops (San Francisco) | Suggested contribution deadline 1 Feb 2027; accepted workshops announced 29 Nov 2026 | Official ICLR call for workshops | Per workshop (typically 4 to 9 pp) | Usually double-blind on OpenReview | ICLR template | Per workshop | Good if an agent-memory or agent-security workshop is accepted (ICLR 2026 had MemAgents). Usually non-archival, so the work can go to a journal afterwards. Check the list after 29 Nov. |
| IEEE Communications Magazine, special issue "Memory Poisoning Across Networked Learning-Enabled Agents" | 15 Mar 2027; publication Oct 2027 | Official CFP | At most 5,500 words and 7 pages; at most 6 figures and tables combined; at most 15 references | Not stated | Two-column IEEE; numbered references; no footnote references; equations not used (up to three simple ones if justified) | Not stated | Topically closest (detection techniques; benchmarks), and the no-equation, short-form format matches the thesis's style. Its emphasis is multi-agent, so the single-agent framing must be argued. |

## Not yet announced (pattern from the previous edition)

| Venue | Expected | Basis |
|---|---|---|
| ESORICS 2027 (Springer LNCS, double-blind) | Winter cycle about 9 to 10 Jan 2027; spring cycle about 21 to 22 Apr 2027 | 2025 and 2026 calls (official-style listings) |
| DIMVA 2027 | Cycle 1 about early Dec 2026; cycle 2 about mid Feb 2027 | 2026: 10 Dec 2025 and 11 Feb 2026 (ieee-security.org listing) |
| PAKDD 2027 (Wellington, 29 Jun to 2 Jul 2027; special track on LLMs and agentic AI) | Probably mid to late Nov 2026 | PAKDD 2026 listed 15 or 25 Nov 2025 (sources disagree) |
| RAID 2027 | About mid Apr 2027 | RAID 2026: 16 Apr 2026 (aggregator) |
| ACSAC 2027 | About late May 2027 | ACSAC 2026: 26 May 2026 (aggregator) |
| AISec 2027 (ACM CCS workshop) | Usually summer 2027 | Not found for 2026 either |

## Passed or not recommended

* IEEE SaTML 2027: 30 Sep 2026 (aggregator). Passed.
* IEEE S&P 2027: abstract 10 Nov, paper 17 Nov 2026 (aggregators). Top tier; not recommended
  with one threat model and an oracle tie.
* EuroS&P 2027: no call found; the 2026 deadline was 13 Nov 2025.
* Journals with rolling submission (Computers & Security; IEEE TIFS; IEEE TDSC): no deadline;
  confirm each journal's review model and format in its own guide for authors before choosing.

## Suggested route

1. **Benchmark-validity framing** (brief Section 3, option a) to WISTP 2027 if its 9 December
   date is confirmed, or to an ICLR 2027 workshop on 1 February. The label-leakage diagnosis,
   the four-control validation protocol, the duplication confound, and the matched-control
   construction are contributions that stand regardless of the oracle tie.
2. **AsiaCCS round 2 (11 December)** only if E2 is finished and the tie is resolved, or E5
   adds a second threat model. Eight weeks is tight for E5.
3. **Communications Magazine (15 March)** as a short, equation-free article on detecting
   memory poisoning by trajectory, with the validation protocol as its methodological core.

Sources:
[AsiaCCS 2027 CFP](https://asiaccs2027.cityu.edu.mo/call-for-papers/) ·
[ICLR call for workshops](https://iclr.cc/Conferences/current/CallForWorkshops) ·
[IEEE ComMag special issue](https://www.comsoc.org/publications/magazines/ieee-communications-magazine/cfp/memory-poisoning-across-networked-learning-enabled-agents-threats-detection-and-defense) ·
[IEEE ComMag policies and guidelines](https://www.comsoc.org/publications/magazines/ieee-communications-magazine/policies-guidelines) ·
[WISTP 2027 listing (myhuiban)](https://www.myhuiban.com/conference/785.md) ·
[WISTP 2027 listing (infosec-conferences)](https://infosec-conferences.com/event/20270319-international-conference-on-information-security-theory-and-practice/) ·
[ESORICS 2026 CFP](https://www.ieee-security.org/Calendar/cfps/cfp-ESORICS2026.html) ·
[DIMVA 2026 CFP](https://www.ieee-security.org/Calendar/cfps/cfp-DIMVA2026.html) ·
[PAKDD deadlines](https://mldeadlines.com/conference/pakdd/) ·
[SaTML deadlines](https://mldeadlines.com/conference/satml/) ·
[S&P 2027 listing](https://www.getpaperpilot.com/deadlines/s-p-2027.html)
