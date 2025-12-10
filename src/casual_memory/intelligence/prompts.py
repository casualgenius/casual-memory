"""
System prompts for intelligence components.

These prompts are used with LLM-based verifiers and detectors for
memory classification, conflict detection, and duplicate detection.
"""

# Conflict detection prompt (enhanced version)
CONFLICT_DETECTION_SYSTEM_PROMPT_OLD = """Your job is to determine if two statements contradict each other?

Key rule: If one statement adds MORE DETAIL to the other without changing the core fact, it's NOT a contradiction.

Consider:
- Direct contradictions: "I live in Paris" vs "I live in London" → YES (different cities)
- Job contradictions: "I work as a teacher" vs "I work as a doctor" → YES (different jobs)
- Job refinement: "I work as engineer" vs "I work as senior software engineer at Google" → NO (added detail)
- Location refinement: "I live in France" vs "I live in Paris, France" → NO (added detail)
- Temporal progression: "I was a student" vs "I now work at Google" → NO (past vs present)
- Temporal changes: "I used to drink coffee" vs "I stopped drinking coffee" → NO (both true at different times)
- Incompatible skill levels: "I'm a beginner at piano" vs "I'm an expert pianist" → YES (can't be both)
- Synonyms: "software developer" vs "software engineer" → NO
- Unrelated facts: Different topics entirely → NO

Respond with ONLY one word: YES or NO
"""

# Conflict detection prompt (enhanced version)
CONFLICT_DETECTION_SYSTEM_PROMPT = """Your job is to determine if two statements contradict each other.

Consider:
- Direct contradictions: "I live in Berlin" vs "I live in Tokyo" → YES
- Job contradictions: "I work as a nurse" vs "I work as a lawyer" → YES
- Refinements: "I work as a manager" vs "I work as a senior marketing manager at Microsoft" → NO
- Location refinements: "I live in Canada" vs "I live in Toronto, Canada" → NO
- Temporal progression: "I was in college" vs "I now work at Amazon" → NO (past vs present)
- Temporal changes: "I used to eat meat" vs "I became vegan last year" → NO (both true at different times)
- Incompatible skill levels: "I'm learning Spanish" vs "I'm fluent in Spanish" → YES (contradictory levels)
- Compatible facts: "I have a bird" vs "I have a fish" → NO (can have multiple pets)
- Compatible activities: "I enjoy hiking" vs "I enjoy reading" → NO (can have multiple hobbies)
- Synonyms: "programmer" vs "coder" → NO
- Unrelated facts: Different topics entirely → NO

Respond with ONLY one word: YES or NO
"""

# Alternative conflict detection prompt with more detailed reasoning
CONFLICT_DETECTION_SYSTEM_PROMPT_DETAILED = """Your job is to analyze if these two statements contradict each other.

Classification Guidelines:

**CONTRADICTION (YES):**
- Same topic but incompatible values: "I live in Berlin" vs "I live in Tokyo"
- Job contradictions: "I work as a nurse" vs "I work as a lawyer" (mutually exclusive current roles)
- Direct negation: "I enjoy reading" vs "I hate reading"
- Mutually exclusive states: "I am vegan" vs "I eat chicken regularly"
- Incompatible skill levels: "I can't swim" vs "I'm a competitive swimmer"

**NOT A CONTRADICTION (NO):**
- Refinements: "I'm an engineer" vs "I'm a senior backend engineer at Amazon"
- Temporal progression: "I studied at MIT" vs "I now work at SpaceX" (past → present)
- Different time periods: "I played soccer" vs "I stopped playing soccer in 2020"
- Synonyms or paraphrases: "programmer" vs "coder"
- Unrelated topics: "I enjoy hiking" vs "I live in Seattle"
- Compatible facts: "I play guitar" vs "I play drums"

**Edge Cases:**
- Intensity differences are NOT contradictions: "I enjoy coffee" vs "I'm obsessed with coffee"
- Multiple skills/roles can coexist: "data scientist" vs "machine learning engineer"

Respond with ONLY one word: YES or NO
"""

# Aliases for backward compatibility
CONFLICT_DETECTION_PROMPT = CONFLICT_DETECTION_SYSTEM_PROMPT
CONFLICT_DETECTION_PROMPT_DETAILED = CONFLICT_DETECTION_SYSTEM_PROMPT_DETAILED

# Duplicate detection prompt for same/refinement detection
DUPLICATE_DETECTION_SYSTEM_PROMPT = """Your job is to determine if two statements describing the same fact (one being a refinement or duplicate of the other) or are they distinct pieces of information?Are these two statements describing the same fact (one being a refinement or duplicate of the other) or are they distinct pieces of information?

Consider:

**SAME FACT (duplicate/refinement):**
- Exact duplicates: "I live in Paris" vs "I live in Paris"
- Paraphrases: "I work as a software engineer" vs "I'm employed as a software developer"
- Intensity variations: "I like coffee" vs "I love coffee"
- Location refinements: "I live in London" vs "I live in Central London"
- Job refinements: "I work as engineer" vs "I work as senior software engineer at Google"
- Relationship clarifications: "My girlfriend is Bow" vs "My girlfriend's name is Bow"

**DISTINCT FACTS (different information):**
- Different attributes: "I live in Bangkok" vs "I work in Bangkok" (residence vs employment)
- Different preferences: "I like coffee" vs "I like tea"
- Different personal info: "My name is Alex" vs "My age is 30"
- Different skills: "I know Python" vs "I know JavaScript"
- Contradictions: "I live in Paris" vs "I live in London"
- Temporal changes: "I lived in Paris" vs "I now live in London"

Respond with ONLY one word: SAME or DISTINCT
"""
