# Memory Extraction Comparison
Generated on: 2025-12-09 04:22:35

## Extraction System Prompt

You are an expert memory extraction agent. Your purpose is to identify and extract significant pieces of information from a conversation that should be remembered.

You will be given a conversation as a list of JSON messages.

### Instructions
1.  **Analyze the Conversation**: Carefully read the entire conversation to understand the context.
2.  **Extract Memories**: Identify any facts, preferences, goals, or significant events. Do NOT extract conversational filler (e.g., "Okay, got it", "That's great!").
3.  **Format Output**: Return a single JSON object with a top-level key "memories", containing a list of memory objects. Do not include any other text, explanations, or markdown.

For each memory you find, return a JSON object with the following fields:
- `text` (string): A concise, self-contained statement.
- `type` (string): The category of the memory. Must be one of:
    - `fact`: Objective, verifiable information (e.g., name, location, job, medical conditions, allergies, physical attributes)
    - `preference`: Subjective likes/dislikes or opinions (e.g., favorite food, enjoyment of activities, distaste for crowds)
    - `goal`: Intentions or tasks to accomplish (e.g., reminders, learning objectives, future aspirations)
    - `event`: Specific occurrences with dates (e.g., appointments, trips, birthdays, meetings)
- `tags` (list of strings): A list of relevant lowercase keywords.
- `importance` (float): A score from 0.0 to 1.0 indicating how crucial the memory is. Medical information and safety-critical facts (allergies) should be 0.9-1.0.
- `source` (string): The role of the message originator: "user", "assistant", or "tool".
- `valid_until` (string, optional): An ISO8601 timestamp for temporary memories. Today is {today_natural} (ISO: {isonow}).
    - Use this for reminders, appointments, or information that expires (e.g., "remind me tomorrow morning").

### CRITICAL RULES
- **First-Person Perspective**: For memories about the user (facts, preferences, goals), the `text` MUST be in the first person (e.g., "My name is Alex", "I am learning guitar"). Do NOT use "The user is...".
- **Self-Contained Memories**: The `text` should be understandable without the context of the conversation.
    - BAD: "Yes, that's my name."
    - GOOD: "My name is Alex."
    - BAD: "but I can't stand crowded places." (sentence fragment with "but")
    - GOOD: "I can't stand crowded places."
- **Atomic Facts**: Split compound statements into separate, focused memories:
    - If a sentence contains multiple independent facts, extract each as a separate memory
    - Example: "I'm allergic to peanuts and shellfish" → TWO memories (peanut allergy + shellfish allergy)
    - Example: "I work as a software engineer at Google" → Multiple memories (job title + employer)
    - Medical/safety information (allergies) is especially important to split for precise matching
- **State Changes**: When extracting information about changes over time, focus on the CURRENT state:
    - "I used to smoke but I quit 5 years ago" → Extract BOTH: current state ("I don't smoke") AND the change event ("I quit smoking 5 years ago")
    - The current state is usually more important than the past behavior
- **Sentiment Extraction**: When users express strong feelings (love, hate, enjoy, can't stand), extract these as separate preference memories:
    - "I went to Paris and absolutely loved it" → event ("I visited Paris") + preference ("I loved visiting Paris")
    - "I enjoy hiking but hate crowds" → TWO preferences (positive + negative)
- **Time References**: Extract temporal information as mentioned by the user, preserving times:
    - "I have a dentist appointment tomorrow at 2pm" → "I have a dentist appointment tomorrow at 2pm"
    - "I have a team meeting on Friday at 10am" → "I have a team meeting on Friday at 10am"
    - Keep relative dates ("tomorrow", "Friday", "in 2 days") - the system will normalize them
    - For `valid_until`: Leave as null - the system will calculate expiry times for temporal memories
- **No Redundancy**: Do not extract memories that are already present or are minor variations of other extracted memories in the same turn.
- **Always Return a List**: Even if you find only one memory, it MUST be inside the "memories" list.

### Examples

Example 1 - If the user says, "Pick up my prescription at the pharmacy on Thursday at 3pm.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I need to pick up my prescription at the pharmacy on Thursday at 3pm.",
      "type": "goal",
      "tags": ["reminder", "pharmacy", "prescription"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 2 - If the user says, "I have a haircut appointment next Monday at 11am and dinner plans on Saturday evening.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I have a haircut appointment next Monday at 11am.",
      "type": "event",
      "tags": ["appointment", "haircut"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I have dinner plans on Saturday in the evening.",
      "type": "event",
      "tags": ["dinner", "plans"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 3 - If the user says, "I graduated from MIT in 2020 with a degree in computer science.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I graduated from MIT in 2020.",
      "type": "fact",
      "tags": ["education", "mit", "graduation"],
      "importance": 0.9,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I have a degree in computer science.",
      "type": "fact",
      "tags": ["education", "degree", "computer science"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 4 - If the user says, "I'm allergic to dairy and tree nuts.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I am allergic to dairy.",
      "type": "fact",
      "tags": ["allergy", "dairy", "health"],
      "importance": 1.0,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I am allergic to tree nuts.",
      "type": "fact",
      "tags": ["allergy", "tree nuts", "health"],
      "importance": 1.0,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 5 - If the user says, "I used to drink coffee every day but I stopped 3 months ago.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I don't drink coffee.",
      "type": "fact",
      "tags": ["coffee", "beverage", "habit"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I stopped drinking coffee 3 months ago.",
      "type": "event",
      "tags": ["coffee", "stopped", "habit"],
      "importance": 0.5,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 6 - If the user says, "I visited Tokyo last year and had an amazing time.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I visited Tokyo last year.",
      "type": "event",
      "tags": ["travel", "tokyo", "vacation"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I had an amazing time in Tokyo.",
      "type": "preference",
      "tags": ["tokyo", "travel", "enjoyment"],
      "importance": 0.6,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 7 - If the user says, "My birthday is July 23rd.", you should return:
```json
{{
  "memories": [
    {{
      "text": "My birthday is on July 23rd.",
      "type": "fact",
      "tags": ["birthday", "personal"],
      "importance": 0.9,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Note: Birthdays are recurring annual events, so they don't expire - use `type: "fact"` not `"event"` and `valid_until: null`.

Example 8 - If the user says, "Remind me to call my mom next Tuesday.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I need to call my mom next Tuesday.",
      "type": "goal",
      "tags": ["reminder", "call", "mom"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Even if there is only ONE memory, it MUST be inside the "memories" array.


## Conversation 1

```
- user: I live in London and I prefer to get weather updates in Celsius.
- assistant: Okay, I'll remember that you live in London and prefer Celsius for weather forecasts.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I live in London. | fact | location, london | 0.8 | user | None | 2.14s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I prefer to get weather updates in Celsius. | preference | weather, temperature, celsius | 0.7 | user | None |  |
| qwen2.5:7b-instruct | I live in London. | fact | location, home | 0.8 | user | None | 8.25s |
| qwen2.5:7b-instruct | I prefer to get weather updates in Celsius. | preference | weather, updates, Celsius | 1.0 | user | None |  |

---

## Conversation 2

```
- user: What's my name?
- assistant: You haven't told me your name yet.
- user: Oh, right. My name is Alex.
- assistant: Got it. I'll remember that your name is Alex.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | My name is Alex. | fact | name, personal | 0.9 | user | None | 1.16s |
| qwen2.5:7b-instruct | My name is Alex. | fact | name, personal | 0.9 | user | None | 1.32s |

---

## Conversation 3

```
- user: Can you remind me to buy milk tomorrow morning?
- assistant: Sure, I'll remind you to buy milk tomorrow morning.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I need to buy milk on December 10. | goal | reminder, buy, milk | 0.7 | user | 2025-12-10T23:59:59 | 0.90s |
| qwen2.5:7b-instruct | I need to buy milk on December 10. | goal | reminder, milk | 0.8 | user | 2025-12-10T23:59:59 | 1.37s |

---

## Conversation 4

```
- user: I have a doctor's appointment in 2 days
- assistant: Oh, I hope everything is ok.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I have a doctor's appointment on December 11. | event | appointment, doctor, health | 0.8 | user | 2025-12-11T23:59:59 | 1.25s |
| qwen2.5:7b-instruct | I have a doctor's appointment on December 11. | event | appointment, doctor | 0.8 | user | 2025-12-11T23:59:59 | 1.40s |

---

## Conversation 5

```
- user: I'm trying to learn how to play the guitar.
- assistant: That's great! It's a wonderful instrument. I'll make a note that you're learning to play the guitar.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I am learning to play the guitar. | goal | guitar, learning, music | 0.7 | user | None | 3.06s |
| qwen2.5:7b-instruct | I am trying to learn how to play the guitar. | goal | learning, guitar | 0.8 | user | None | 1.40s |

---

## Conversation 6

```
- user: I went to Paris last summer and absolutely loved it.
- assistant: That sounds wonderful! I'll remember that you visited Paris.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I visited Paris last summer. | event | travel, paris, vacation | 0.7 | user | None | 1.30s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I loved visiting Paris. | preference | paris, travel, enjoyment | 0.6 | user | None |  |
| qwen2.5:7b-instruct | I visited Paris last summer. | event | travel, paris, vacation | 0.7 | user | None | 2.09s |
| qwen2.5:7b-instruct | I loved visiting Paris. | preference | paris, enjoyment | 0.6 | user | None |  |

---

## Conversation 7

```
- user: I have a dentist appointment tomorrow at 2pm and a team meeting on Friday at 10am.
- assistant: Got it, I've noted both appointments.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I have a dentist appointment on December 10 at 2pm. | event | appointment, dentist | 0.7 | user | 2025-12-10T23:59:59 | 1.14s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I have a team meeting on December 12 10am. | event | meeting, team | 0.7 | user | 2025-12-12T23:59:59 |  |
| qwen2.5:7b-instruct | I have a dentist appointment on December 10 at 2pm. | event | appointment, dentist | 0.7 | user | 2025-12-10T23:59:59 | 2.16s |
| qwen2.5:7b-instruct | I have a team meeting on December 12 10am. | event | meeting, team | 0.8 | user | 2025-12-12T23:59:59 |  |

---

## Conversation 8

```
- user: I want to visit Japan someday.
- assistant: That would be amazing! I'll remember that goal.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I want to visit Japan someday. | goal | travel, japan, someday | 0.6 | user | None | 0.83s |
| qwen2.5:7b-instruct | I want to visit Japan someday. | goal | travel, Japan, someday | 0.8 | user | None | 1.41s |

---

## Conversation 9

```
- user: I work as a software engineer at Google in London.
- assistant: That's great! I'll make a note of that.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I work as a software engineer. | fact | job, software engineer, profession | 0.8 | user | None | 1.28s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I work at Google. | fact | employer, google, company | 0.8 | user | None |  |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I work in London. | fact | location, london, workplace | 0.7 | user | None |  |
| qwen2.5:7b-instruct | I work as a software engineer at Google. | fact | job, software engineering, google | 0.9 | user | None | 2.13s |
| qwen2.5:7b-instruct | I am based in London. | fact | location, based, london | 0.8 | user | None |  |

---

## Conversation 10

```
- user: I'm allergic to peanuts and shellfish.
- assistant: Important to know, I'll remember that.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I am allergic to peanuts. | fact | allergy, peanuts, health | 1.0 | user | None | 1.16s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I am allergic to shellfish. | fact | allergy, shellfish, health | 1.0 | user | None |  |
| qwen2.5:7b-instruct | I am allergic to peanuts. | fact | allergy, peanuts, health | 1.0 | user | None | 2.12s |
| qwen2.5:7b-instruct | I am allergic to shellfish. | fact | allergy, shellfish, health | 1.0 | user | None |  |

---

## Conversation 11

```
- user: I really enjoy hiking on weekends, but I can't stand crowded places.
- assistant: Got it, I'll keep that in mind.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I really enjoy hiking on weekends. | preference | hiking, weekends, enjoy | 0.7 | user | None | 1.64s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I can't stand crowded places. | preference | crowds, distaste, personal space | 0.8 | user | None |  |
| qwen2.5:7b-instruct | I enjoy hiking on weekends. | preference | hiking, weekends, enjoyment | 0.8 | user | None | 2.15s |
| qwen2.5:7b-instruct | I can't stand crowded places. | preference | crowded places, dislike | 1.0 | user | None |  |

---

## Conversation 12

```
- user: My birthday is next Tuesday.
- assistant: I'll make a note of that!
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | My birthday is on December 16. | fact | birthday, personal | 0.9 | user | None | 0.97s |
| qwen2.5:7b-instruct | My birthday is on December 16. | fact | birthday, personal | 0.9 | user | None | 1.32s |

---

## Conversation 13

```
- user: I used to smoke but I quit 5 years ago.
- assistant: Good for you! That's an important change.
```

| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-next-80b-a3b-instruct-2509 | I don't smoke. | fact | smoking, quit, health | 0.9 | user | None | 3.04s |
| qwen/qwen3-next-80b-a3b-instruct-2509 | I quit smoking 5 years ago. | event | smoking, quit, habit | 0.7 | user | None |  |
| qwen2.5:7b-instruct | I don't smoke. | fact | smoking, health | 0.9 | user | None | 2.04s |
| qwen2.5:7b-instruct | I quit smoking 5 years ago. | event | quit, smoking, history | 0.8 | user | None |  |

---

