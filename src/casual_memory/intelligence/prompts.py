"""
System prompts for intelligence components.

These prompts are used with LLM-based verifiers and detectors for
memory classification, conflict detection, and duplicate detection.
"""

# Conflict detection prompt (based on research with 96.2% accuracy)
CONFLICT_DETECTION_PROMPT = """Do these two statements contradict each other?

Statement A: "{statement_a}"
Statement B: "{statement_b}"

Consider:
- Direct contradictions: "I live in X" vs "I live in Y" → YES
- Refinements: "I work as engineer" vs "I work as software engineer at Google" → NO
- Temporal changes: "I used to X" vs "I quit X 5 years ago" → NO (both true at different times)
- Synonyms: "software developer" vs "software engineer" → NO
- Unrelated facts: Different topics entirely → NO

Respond with ONLY one word: YES or NO

Answer:"""


# Alternative conflict detection prompt with more detailed reasoning
CONFLICT_DETECTION_PROMPT_DETAILED = """Analyze if these two statements contradict each other.

Statement A: "{statement_a}"
Statement B: "{statement_b}"

Classification Guidelines:

**CONTRADICTION (YES):**
- Same topic but incompatible values: "I live in Paris" vs "I live in London"
- Direct negation: "I like coffee" vs "I hate coffee"
- Mutually exclusive states: "I am vegetarian" vs "I eat meat daily"

**NOT A CONTRADICTION (NO):**
- Refinements: "I work as engineer" vs "I work as senior software engineer at Google"
- Temporal progression: "I lived in Paris" vs "I now live in London"
- Different time periods: "I used to smoke" vs "I quit smoking 5 years ago"
- Synonyms or paraphrases: "software developer" vs "software engineer"
- Unrelated topics: "I like pizza" vs "I live in Paris"
- Compatible facts: "I have a dog" vs "I have a cat"

**Edge Cases:**
- Intensity differences are NOT contradictions: "I like X" vs "I love X"
- Related jobs are NOT contradictions: "frontend developer" vs "backend developer"

Respond with ONLY one word: YES or NO

Answer:"""


# Duplicate detection prompt for same/refinement detection
DUPLICATE_DETECTION_PROMPT = """Are these two statements the same fact or is one a more detailed version of the other?

Statement A: "{statement_a}"
Statement B: "{statement_b}"

Consider:

**SAME (YES):**
- Exact duplicates: "I live in Paris" vs "I live in Paris"
- Paraphrases: "I work as a software engineer" vs "I'm employed as a software developer"
- Intensity variations: "I like coffee" vs "I love coffee"
- Refinements: "I work as engineer" vs "I work as senior software engineer at Google"
  (One statement adds detail but doesn't contradict the core fact)

**DIFFERENT (NO):**
- Distinct facts: "I live in Paris" vs "I work in London"
- Contradictions: "I live in Paris" vs "I live in London"
- Temporal changes: "I lived in Paris" vs "I now live in London"
- Unrelated statements: "I like pizza" vs "The sky is blue"

Respond with ONLY one word: YES or NO

Answer:"""
