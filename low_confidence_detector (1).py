"""
Low Confidence Words Detector — FIXED VERSION
===============================================
For Waymo Triage Caption Quality Check (Checklist Item 10)

Fixes from v1:
  - Added "look like" (not just "looks like")
  - Added Grammar Fix category (It's, It is, there's)
  - Added Spatial Precision category (in the front, in front of me)
  - Added Action Precision category (taking a turn)
  - Improved auto-fix to handle sentence restructuring
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# 1. ALL PATTERN CATEGORIES
# ─────────────────────────────────────────────────────────────

LOW_CONFIDENCE_PATTERNS = [

    # ── VISUAL UNCERTAINTY ────────────────────────────────────
    {
        "category": "Visual Uncertainty",
        "patterns": [
            (r"\blooks?\s+like\b",   "Avoid visual guessing — state facts",        "is"),
            (r"\bappears to be\b",   "State what it is, not guesses",              "is"),
            (r"\bappears\b",         "Use factual observation",                    "is"),
            (r"\bseems to be\b",     "Avoid assumption language",                  "is"),
            (r"\bseems\b",           "Avoid assumption language",                  "is"),
            (r"\bI think\b",         "Remove subjective language",                 ""),
            (r"\bI believe\b",       "Remove subjective language",                 ""),
            (r"\bI feel\b",          "Remove subjective language",                 ""),
            (r"\bpossibly\b",        "Use certain language",                       ""),
            (r"\bprobably\b",        "Use certain language",                       ""),
            (r"\bperhaps\b",         "Use certain language",                       ""),
            (r"\bmaybe\b",           "Use certain language",                       ""),
        ]
    },

    # ── MODAL UNCERTAINTY ─────────────────────────────────────
    {
        "category": "Modal Uncertainty",
        "patterns": [
            (r"\bmight be\b",        "Use definitive statement",                   "is"),
            (r"\bmight\b",           "Use definitive statement",                   "will"),
            (r"\bcould be\b",        "Use definitive statement",                   "is"),
            (r"\bwould be\b",        "Use definitive statement",                   "is"),
            (r"\bshould be\b",       "Use factual observation",                    "is"),
            (r"\bmay be\b",          "Avoid hedging language",                     "is"),
            (r"\bmay\b",             "Avoid hedging language",                     "will"),
        ]
    },

    # ── APPROXIMATION ─────────────────────────────────────────
    {
        "category": "Approximation",
        "patterns": [
            (r"\babout\b",           "Use exact value if known",                   "approximately"),
            (r"\broughly\b",         "Use exact value if known",                   "approximately"),
            (r"\bsomething like\b",  "Be specific",                                ""),
            (r"\bkind of\b",         "Remove filler phrase",                       ""),
            (r"\bsort of\b",         "Remove filler phrase",                       ""),
            (r"\bsomewhat\b",        "Use precise description",                    ""),
        ]
    },

    # ── VISIBILITY DOUBT ──────────────────────────────────────
    {
        "category": "Visibility Doubt",
        "patterns": [
            (r"\bnot\s+sure\b",           "Remove uncertainty expression",         ""),
            (r"\buncertain\b",            "Remove uncertainty expression",         ""),
            (r"\bhard to (tell|say|see)\b","If unclear, remove the object",        ""),
            (r"\bI cannot (tell|say|confirm)\b", "Remove unverifiable claims",     ""),
            (r"\bI('m| am) not sure\b",   "Remove subjective uncertainty",         ""),
            (r"\bnot (entirely|fully|completely) (clear|visible|sure)\b",
                                           "Clarify or remove if unseen",          ""),
        ]
    },

    # ── WEAK / HEDGED ACTIONS ─────────────────────────────────
    {
        "category": "Weak Actions",
        "patterns": [
            (r"\btrying to\b",       "State the action directly",                  ""),
            (r"\battempting to\b",   "State the action directly",                  ""),
            (r"\bexpected to\b",     "Use factual description",                    "will"),
            (r"\blikely to\b",       "Use factual description",                    "will"),
            (r"\bintend(s|ing)? to\b","Use direct action statement",               "will"),
            (r"\bplan(s|ning)? to\b","Use direct action statement",                "will"),
        ]
    },

    # ── PARTIAL OBSERVATIONS ──────────────────────────────────
    {
        "category": "Partial Observations",
        "patterns": [
            (r"\bhard to see\b",               "Remove if object not visible",     ""),
            (r"\bdifficult to (see|determine|tell)\b",
                                               "Remove unverifiable claims",        ""),
            (r"\bnot clearly visible\b",       "Omit if not clearly visible",      ""),
            (r"\bI can'?t (see|tell|confirm)\b","Remove unverifiable claims",      ""),
            (r"\bpartially visible\b",         "Describe what IS visible",         ""),
        ]
    },

    # ── GRAMMAR FIX (NEW) ─────────────────────────────────────
    {
        "category": "Grammar Fix",
        "patterns": [
            (r"\bIt'?s\s+look\b",    "Should be 'It looks' or remove entirely",   "The"),
            (r"\bIt'?s\s+looks\b",   "Should be 'It looks' — remove 'It's'",      "It"),
            (r"\bIt'?s\b",           "Avoid 'It's' — use subject noun instead",   "The"),
            (r"\bThere'?s\b",        "Avoid 'There's' — be specific",             "A"),
            (r"\bthey'?re\b",        "Avoid contraction — use 'they are'",        "they are"),
            (r"\bwe'?re\b",          "Avoid contraction — use 'we are'",          "we are"),
            (r"\bdon'?t\b",          "Avoid contraction — use 'do not'",          "do not"),
            (r"\bcan'?t\b",          "Avoid contraction — use 'cannot'",          "cannot"),
            (r"\bwon'?t\b",          "Avoid contraction — use 'will not'",        "will not"),
        ]
    },

    # ── SPATIAL PRECISION (NEW) ───────────────────────────────
    {
        "category": "Spatial Precision",
        "patterns": [
            (r"\bin the front\b",    "Use 'ahead' for forward direction",          "ahead"),
            (r"\bin front of me\b",  "Use 'ahead' for AV captions",               "ahead"),
            (r"\bup ahead\b",        "Use 'ahead' — remove redundant 'up'",       "ahead"),
            (r"\bover there\b",      "Be specific about direction/distance",       ""),
            (r"\bnearby\b",          "Specify exact distance or position",         ""),
            (r"\bsomewhere\b",       "Be specific about location",                 ""),
            (r"\bfar away\b",        "Use exact distance in meters",              ""),
            (r"\bclose by\b",        "Use exact distance in meters",              ""),
        ]
    },

    # ── ACTION PRECISION (NEW) ────────────────────────────────
    {
        "category": "Action Precision",
        "patterns": [
            (r"\btaking a right turn\b",  "Use 'turning right'",                  "turning right"),
            (r"\btaking a left turn\b",   "Use 'turning left'",                   "turning left"),
            (r"\btaking a turn\b",        "Specify direction: turning right/left", "turning"),
            (r"\bmaking a right turn\b",  "Use 'turning right'",                  "turning right"),
            (r"\bmaking a left turn\b",   "Use 'turning left'",                   "turning left"),
            (r"\bgoing straight\b",       "Use 'continuing straight'",            "continuing straight"),
            (r"\bslowing down\b",         "Use 'decelerating'",                   "decelerating"),
            (r"\bspeeding up\b",          "Use 'accelerating'",                   "accelerating"),
            (r"\bstopping\b",             "Use 'coming to a stop'",               "coming to a stop"),
        ]
    },
]


# ─────────────────────────────────────────────────────────────
# 2. DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class DetectedIssue:
    word_found:  str
    category:    str
    reason:      str
    suggestion:  str
    position:    int
    sentence:    str


@dataclass
class DetectionResult:
    original_text:  str
    issues:         List[DetectedIssue] = field(default_factory=list)
    cleaned_text:   str = ""
    total_found:    int = 0
    categories_hit: List[str] = field(default_factory=list)
    passed:         bool = False

    def summary(self) -> str:
        if self.passed:
            return "✅ PASSED — No low confidence words found"
        return (
            f"❌ FAILED — {self.total_found} issue(s) found "
            f"in {len(self.categories_hit)} category(ies): "
            f"{', '.join(self.categories_hit)}"
        )


# ─────────────────────────────────────────────────────────────
# 3. DETECTOR CLASS
# ─────────────────────────────────────────────────────────────

class LowConfidenceDetector:

    def __init__(self):
        self._compiled = []
        for cat_block in LOW_CONFIDENCE_PATTERNS:
            cat = cat_block["category"]
            for (pattern, reason, suggestion) in cat_block["patterns"]:
                self._compiled.append({
                    "regex":      re.compile(pattern, re.IGNORECASE),
                    "pattern":    pattern,
                    "category":   cat,
                    "reason":     reason,
                    "suggestion": suggestion,
                })

    def _sentences(self, text: str) -> List[Tuple[int, str]]:
        results = []
        for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
            results.append((m.start(), m.group().strip()))
        return results

    def analyze(self, text: str) -> DetectionResult:
        result = DetectionResult(original_text=text)
        sentences = self._sentences(text)

        for entry in self._compiled:
            for match in entry["regex"].finditer(text):
                match_pos = match.start()
                sentence = text
                for (start, sent) in sentences:
                    end = start + len(sent)
                    if start <= match_pos <= end:
                        sentence = sent
                        break

                result.issues.append(DetectedIssue(
                    word_found = match.group(),
                    category   = entry["category"],
                    reason     = entry["reason"],
                    suggestion = entry["suggestion"],
                    position   = match_pos,
                    sentence   = sentence,
                ))

        # Deduplicate overlapping matches — keep first hit per position range
        seen = set()
        unique = []
        for issue in sorted(result.issues, key=lambda x: x.position):
            key = (issue.position, issue.word_found.lower())
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        result.issues         = unique
        result.total_found    = len(unique)
        result.categories_hit = list({i.category for i in unique})
        result.passed         = result.total_found == 0
        result.cleaned_text   = self._auto_fix(text, unique)
        return result

    def _auto_fix(self, text: str, issues: List[DetectedIssue]) -> str:
        cleaned = text
        for issue in sorted(issues, key=lambda x: x.position, reverse=True):
            pat = re.compile(re.escape(issue.word_found), re.IGNORECASE)
            if issue.suggestion:
                cleaned = pat.sub(issue.suggestion, cleaned, count=1)
            else:
                cleaned = pat.sub(" ", cleaned, count=1)
        cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
        return cleaned

    def report(self, result: DetectionResult) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  LOW CONFIDENCE WORDS — DETECTION REPORT")
        lines.append("=" * 65)
        lines.append(f"\n  Status  : {result.summary()}")
        lines.append(f"  Total   : {result.total_found} issue(s) found")
        if result.categories_hit:
            lines.append(f"  In      : {', '.join(result.categories_hit)}")

        if result.issues:
            lines.append("\n" + "-" * 65)
            for i, issue in enumerate(result.issues, 1):
                fix = f"→ '{issue.suggestion}'" if issue.suggestion else "→ REMOVE"
                lines.append(f"\n  #{i}  [{issue.category}]")
                lines.append(f"      Found    : '{issue.word_found}'")
                lines.append(f"      Reason   : {issue.reason}")
                lines.append(f"      Fix      : {fix}")
                lines.append(f"      Sentence : {issue.sentence[:80]}")

        lines.append("\n" + "-" * 65)
        lines.append(f"  BEFORE : {result.original_text}")
        lines.append(f"  AFTER  : {result.cleaned_text}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 4. TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    detector = LowConfidenceDetector()

    tests = [
        (
            "YOUR EXACT SENTENCE",
            "It's look like that vehicle in the front is taking a right turn"
        ),
        (
            "WAYMO CAPTION WITH ISSUES",
            "There's a vehicle that looks like it might be slowing down. "
            "It's probably going to stop somewhere nearby. I think it's "
            "taking a left turn up ahead."
        ),
        (
            "CLEAN CAPTION — SHOULD PASS",
            "A white SUV is parked on the right side, partially blocking the lane. "
            "I will continue straight at 0.8 m/s, navigating carefully. "
            "In 205 meters I will turn right as per route plan."
        ),
    ]

    for (label, text) in tests:
        print(f"\n\n{'#'*65}")
        print(f"  TEST: {label}")
        print(f"{'#'*65}")
        result = detector.analyze(text)
        print(detector.report(result))
