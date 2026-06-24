
   
import re
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
 
 
# ─────────────────────────────────────────────────────────────
# 1. ALL PATTERN CATEGORIES (unchanged from v2)
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
            (r"\bnot\s+sure\b",                "Remove uncertainty expression",    ""),
            (r"\buncertain\b",                 "Remove uncertainty expression",    ""),
            (r"\bhard to (tell|say|see)\b",    "If unclear, remove the object",   ""),
            (r"\bI cannot (tell|say|confirm)\b","Remove unverifiable claims",      ""),
            (r"\bI('m| am) not sure\b",        "Remove subjective uncertainty",   ""),
            (r"\bnot (entirely|fully|completely) (clear|visible|sure)\b",
                                               "Clarify or remove if unseen",      ""),
        ]
    },
 
    # ── WEAK / HEDGED ACTIONS ─────────────────────────────────
    {
        "category": "Weak Actions",
        "patterns": [
            (r"\btrying to\b",        "State the action directly",                 ""),
            (r"\battempting to\b",    "State the action directly",                 ""),
            (r"\bexpected to\b",      "Use factual description",                   "will"),
            (r"\blikely to\b",        "Use factual description",                   "will"),
            (r"\bintend(s|ing)? to\b","Use direct action statement",               "will"),
            (r"\bplan(s|ning)? to\b", "Use direct action statement",               "will"),
        ]
    },
 
    # ── PARTIAL OBSERVATIONS ──────────────────────────────────
    {
        "category": "Partial Observations",
        "patterns": [
            (r"\bhard to see\b",                "Remove if object not visible",    ""),
            (r"\bdifficult to (see|determine|tell)\b",
                                                "Remove unverifiable claims",       ""),
            (r"\bnot clearly visible\b",        "Omit if not clearly visible",     ""),
            (r"\bI can'?t (see|tell|confirm)\b","Remove unverifiable claims",      ""),
            (r"\bpartially visible\b",          "Describe what IS visible",        ""),
        ]
    },
 
    # ── GRAMMAR FIX ───────────────────────────────────────────
    {
        "category": "Grammar Fix",
        "patterns": [
            (r"\bIt'?s\s+look\b",  "Should be 'It looks' or remove entirely",     "The"),
            (r"\bIt'?s\s+looks\b", "Should be 'It looks' — remove 'It's'",        "It"),
            (r"\bIt'?s\b",         "Avoid 'It's' — use subject noun instead",     "The"),
            (r"\bThere'?s\b",      "Avoid 'There's' — be specific",               "A"),
            (r"\bthey'?re\b",      "Avoid contraction — use 'they are'",          "they are"),
            (r"\bwe'?re\b",        "Avoid contraction — use 'we are'",            "we are"),
            (r"\bdon'?t\b",        "Avoid contraction — use 'do not'",            "do not"),
            (r"\bcan'?t\b",        "Avoid contraction — use 'cannot'",            "cannot"),
            (r"\bwon'?t\b",        "Avoid contraction — use 'will not'",          "will not"),
        ]
    },
 
    # ── SPATIAL PRECISION ─────────────────────────────────────
    {
        "category": "Spatial Precision",
        "patterns": [
            (r"\bin the front\b",   "Use 'ahead' for forward direction",           "ahead"),
            (r"\bin front of me\b", "Use 'ahead' for AV captions",                "ahead"),
            (r"\bup ahead\b",       "Use 'ahead' — remove redundant 'up'",        "ahead"),
            (r"\bover there\b",     "Be specific about direction/distance",        ""),
            (r"\bnearby\b",         "Specify exact distance or position",          ""),
            (r"\bsomewhere\b",      "Be specific about location",                  ""),
            (r"\bfar away\b",       "Use exact distance in meters",               ""),
            (r"\bclose by\b",       "Use exact distance in meters",               ""),
        ]
    },
 
    # ── ACTION PRECISION ──────────────────────────────────────
    {
        "category": "Action Precision",
        "patterns": [
            (r"\btaking a right turn\b", "Use 'turning right'",                   "turning right"),
            (r"\btaking a left turn\b",  "Use 'turning left'",                    "turning left"),
            (r"\btaking a turn\b",       "Specify direction: turning right/left",  "turning"),
            (r"\bmaking a right turn\b", "Use 'turning right'",                   "turning right"),
            (r"\bmaking a left turn\b",  "Use 'turning left'",                    "turning left"),
            (r"\bgoing straight\b",      "Use 'continuing straight'",             "continuing straight"),
            (r"\bslowing down\b",        "Use 'decelerating'",                    "decelerating"),
            (r"\bspeeding up\b",         "Use 'accelerating'",                    "accelerating"),
            (r"\bstopping\b",            "Use 'coming to a stop'",                "coming to a stop"),
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
    source_file:    str = "manual input"
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
# 3. FILE INPUT HANDLERS (NEW)
# ─────────────────────────────────────────────────────────────
 
class FileInputHandler:
    """
    Handles reading text from:
      - PDF files  (.pdf)
      - Text files (.txt)
      - Direct string input
    """
 
    # ── Read from PDF ──────────────────────────────────────────
    @staticmethod
    def read_pdf(filepath: str) -> str:
        """
        Extract all text from a PDF file (multi-page supported).
 
        Args:
            filepath: path to the .pdf file
 
        Returns:
            Full extracted text as a single string
        """
        try:
            import pypdf
        except ImportError:
            print("❌ pypdf not installed. Run: pip install pypdf")
            sys.exit(1)
 
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
 
        if not filepath.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {filepath}")
 
        print(f"\n📄 Reading PDF: {filepath}")
 
        text_pages = []
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            print(f"   Total pages: {total_pages}")
 
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_pages.append(page_text.strip())
                    print(f"   Page {i+1}: {len(page_text)} characters extracted")
                else:
                    print(f"   Page {i+1}: ⚠️  No text found (may be scanned image)")
 
        if not text_pages:
            raise ValueError(
                "No text could be extracted from the PDF.\n"
                "The PDF may be a scanned image — use OCR first."
            )
 
        full_text = "\n\n".join(text_pages)
        print(f"\n✅ PDF read successfully — {len(full_text)} total characters")
        return full_text
 
    # ── Read from TXT ──────────────────────────────────────────
    @staticmethod
    def read_txt(filepath: str) -> str:
        """
        Read text from a .txt file.
 
        Args:
            filepath: path to the .txt file
 
        Returns:
            File contents as a string
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
 
        if not filepath.lower().endswith(".txt"):
            raise ValueError(f"File is not a .txt file: {filepath}")
 
        print(f"\n📝 Reading text file: {filepath}")
 
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    text = f.read()
                print(f"✅ File read successfully ({enc}) — {len(text)} characters")
                return text
            except UnicodeDecodeError:
                continue
 
        raise ValueError(f"Could not read file with any known encoding: {filepath}")
 
    # ── Read from manual input ─────────────────────────────────
    @staticmethod
    def read_manual() -> str:
        """
        Accept multi-line text input from the user in terminal.
        Type text, then press Enter twice to finish.
        """
        print("\n✏️  Enter your text below.")
        print("    (Press ENTER twice when done)\n")
        lines = []
        empty_count = 0
        while True:
            try:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                    lines.append(line)
            except EOFError:
                break
 
        text = "\n".join(lines).strip()
        if not text:
            raise ValueError("No text was entered.")
        print(f"\n✅ Input received — {len(text)} characters")
        return text
 
    # ── Auto-detect and read ───────────────────────────────────
    @staticmethod
    def read_file(filepath: str) -> str:
        """
        Auto-detect file type and read accordingly.
 
        Args:
            filepath: path to PDF or TXT file
 
        Returns:
            Extracted text string
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return FileInputHandler.read_pdf(filepath)
        elif ext == ".txt":
            return FileInputHandler.read_txt(filepath)
        else:
            raise ValueError(
                f"Unsupported file type: '{ext}'\n"
                f"Supported types: .pdf, .txt"
            )
 
 
# ─────────────────────────────────────────────────────────────
# 4. DETECTOR CLASS
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
 
    def analyze(self, text: str, source_file: str = "manual input") -> DetectionResult:
        result = DetectionResult(
            original_text=text,
            source_file=source_file
        )
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
 
        # Deduplicate
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
        lines.append(f"\n  Source  : {result.source_file}")
        lines.append(f"  Status  : {result.summary()}")
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
        lines.append(f"  BEFORE :\n  {result.original_text[:300]}"
                     + ("..." if len(result.original_text) > 300 else ""))
        lines.append(f"\n  AFTER  :\n  {result.cleaned_text[:300]}"
                     + ("..." if len(result.cleaned_text) > 300 else ""))
        lines.append("=" * 65)
        return "\n".join(lines)
 
    def save_report(self, result: DetectionResult, output_path: str = None):
        """Save the full report + corrected text to a .txt file"""
        if output_path is None:
            base = os.path.splitext(result.source_file)[0]
            base = os.path.basename(base)
            output_path = f"{base}_report.txt"
 
        report_text = self.report(result)
 
        # Also save full corrected text (not truncated)
        full_output = (
            report_text
            + "\n\n" + "=" * 65
            + "\n  FULL CORRECTED TEXT\n"
            + "=" * 65
            + f"\n\n{result.cleaned_text}\n"
        )
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_output)
 
        print(f"\n💾 Report saved to: {output_path}")
        return output_path
 
 
# ─────────────────────────────────────────────────────────────
# 5. INTERACTIVE MENU (NEW)
# ─────────────────────────────────────────────────────────────
 
def interactive_menu():
    """
    Interactive terminal menu — runs when no arguments are given.
    User picks input type, file path, and whether to save report.
    """
    print("\n" + "=" * 65)
    print("  LOW CONFIDENCE WORDS DETECTOR — v3")
    print("  Waymo Triage Caption Quality Check (Item 10)")
    print("=" * 65)
    print("\n  Select input type:")
    print("  1. PDF file (.pdf)")
    print("  2. Text file (.txt)")
    print("  3. Type text manually")
    print("  4. Run built-in tests")
    print("  0. Exit")
    print()
 
    choice = input("  Enter choice (0-4): ").strip()
 
    detector  = LowConfidenceDetector()
    handler   = FileInputHandler()
    text      = None
    source    = "manual input"
 
    if choice == "0":
        print("Exiting.")
        sys.exit(0)
 
    elif choice == "1":
        filepath = input("\n  Enter PDF file path: ").strip().strip('"')
        text   = handler.read_pdf(filepath)
        source = filepath
 
    elif choice == "2":
        filepath = input("\n  Enter .txt file path: ").strip().strip('"')
        text   = handler.read_txt(filepath)
        source = filepath
 
    elif choice == "3":
        text   = handler.read_manual()
        source = "manual input"
 
    elif choice == "4":
        run_builtin_tests()
        return
 
    else:
        print("❌ Invalid choice. Exiting.")
        sys.exit(1)
 
    # Run analysis
    print("\n⚙️  Running analysis...")
    result = detector.analyze(text, source_file=source)
    print(detector.report(result))
 
    # Ask to save
    save = input("\n  Save report to file? (y/n): ").strip().lower()
    if save == "y":
        out = input("  Output filename (press Enter for auto): ").strip()
        detector.save_report(result, out if out else None)
 
 
# ─────────────────────────────────────────────────────────────
# 6. BUILT-IN TESTS
# ─────────────────────────────────────────────────────────────
 
def run_builtin_tests():
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
        result = detector.analyze(text, source_file="built-in test")
        print(detector.report(result))
 
 
# ─────────────────────────────────────────────────────────────
# 7. CLI ARGUMENT PARSER (NEW)
# ─────────────────────────────────────────────────────────────
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Low Confidence Words Detector — Waymo Triage Caption Check"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdf",
        metavar="FILE.pdf",
        help="Path to a PDF file to analyze"
    )
    group.add_argument(
        "--txt",
        metavar="FILE.txt",
        help="Path to a .txt file to analyze"
    )
    group.add_argument(
        "--text",
        metavar="'your text here'",
        help="Direct text string to analyze"
    )
    group.add_argument(
        "--test",
        action="store_true",
        help="Run built-in test cases"
    )
    parser.add_argument(
        "--save",
        metavar="output.txt",
        help="Save report to this file (optional)",
        default=None
    )
    return parser.parse_args()
 
 
# ─────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    args = parse_args()
    detector = LowConfidenceDetector()
    handler  = FileInputHandler()
 
    # ── No arguments → interactive menu ───────────────────────
    if len(sys.argv) == 1:
        interactive_menu()
        sys.exit(0)
 
    # ── Run built-in tests ─────────────────────────────────────
    if args.test:
        run_builtin_tests()
        sys.exit(0)
 
    # ── PDF input ──────────────────────────────────────────────
    if args.pdf:
        text   = handler.read_pdf(args.pdf)
        source = args.pdf
 
    # ── TXT input ──────────────────────────────────────────────
    elif args.txt:
        text   = handler.read_txt(args.txt)
        source = args.txt
 
    # ── Direct text input ──────────────────────────────────────
    elif args.text:
        text   = args.text
        source = "command line"
 
    else:
        interactive_menu()
        sys.exit(0)
 
    # ── Analyze & print ────────────────────────────────────────
    print("\n⚙️  Running analysis...")
    result = detector.analyze(text, source_file=source)
    print(detector.report(result))
 
    # ── Save if requested ──────────────────────────────────────
    if args.save:
        detector.save_report(result, args.save)
