# Extraction Model Refactoring Summary

**Date**: 2025-12-31
**Objective**: Separate LLM extraction concerns from system-managed fields

## Problem Identified

The `MemoryFact` model was being used as both:
1. **Extraction model** - JSON schema sent to LLM
2. **Storage model** - Full model with system-managed fields

This caused several issues:
- **Schema bloat**: LLM saw unnecessary fields (user_id, confidence, mention_count, archived, etc.)
- **Token waste**: Extra fields in JSON schema sent to LLM
- **Confusion**: System-managed fields visible to LLM could cause confusion
- **Tight coupling**: Extraction interface coupled to storage concerns
- **Hidden issues**: The 80B importance scoring bug was hidden by Optional defaults

## Solution Implemented

Created a clean separation between extraction and storage models:

### New Model: `MemoryFactExtraction`

**Purpose**: ONLY fields the LLM should populate
**Location**: `src/casual_memory/models.py`

```python
class MemoryFactExtraction(BaseModel):
    """Model for LLM memory extraction - contains ONLY fields the LLM should populate."""
    text: str = Field(..., description="...")
    type: Literal["fact", "preference", "event", "goal", "weather"] = Field(...)
    tags: list[str] = Field(...)
    importance: float = Field(..., ge=0.0, le=1.0, ...)  # REQUIRED (no default!)
    valid_until: Optional[str | None] = Field(default=None, ...)
```

**Fields NOT included** (compared to MemoryFact):
- ❌ `source` - System-managed, inferred from message role
- ❌ `user_id` - System-managed
- ❌ `confidence` - System-calculated from mentions
- ❌ `mention_count` - System-managed
- ❌ `first_seen`, `last_seen` - System-managed timestamps
- ❌ `archived`, `archived_at`, `superseded_by` - System lifecycle fields

### Updated: `MemoryExtractionResponse`

Changed from using `MemoryFact` to `MemoryFactExtraction`:

```python
class MemoryExtractionResponse(BaseModel):
    memories: list[MemoryFactExtraction] = Field(...)  # Was: list[MemoryFact]
```

### Updated: `LLMMemoryExtracter`

Converts extraction results to full `MemoryFact` instances:

```python
# Parse LLM response into MemoryFactExtraction
extraction_response = MemoryExtractionResponse.model_validate_json(content)

# Convert each extraction to full MemoryFact (adds system fields with defaults)
for memory_extraction in extraction_response.memories:
    memory_dict = memory_extraction.model_dump()
    normalized_dict = normalize_memory_dates(memory_dict, now)
    normalized_memory = MemoryFact(**normalized_dict)  # System fields get defaults
```

## Benefits Achieved

### 1. **Cleaner Separation of Concerns**
- Extraction model = what LLM provides
- MemoryFact = full model with system + extracted fields
- Clear boundary between extraction and management

### 2. **Smaller JSON Schema**
- Before: 13 fields in schema sent to LLM
- After: 5 fields in schema sent to LLM
- **62% reduction** in schema complexity

### 3. **Better Validation**
- Making `importance` required (not Optional) exposed the 80B model bug immediately
- No silent defaults masking missing fields
- Explicit about what LLM must provide

### 4. **System Fields Protected**
- LLM cannot accidentally set `confidence`, `mention_count`, etc.
- Clear that these are system-calculated/managed
- Prevents confusion about field ownership

### 5. **Easier Testing**
- Test mock responses only include extraction fields
- System fields tested separately
- Clear test assertions about what LLM provides vs system manages

## Files Modified

### Core Models
- ✅ `src/casual_memory/models.py` - Added `MemoryFactExtraction`
- ✅ `src/casual_memory/__init__.py` - Exported new model

### Extraction
- ✅ `src/casual_memory/extractors/models.py` - Updated to use `MemoryFactExtraction`
- ✅ `src/casual_memory/extractors/llm_extractor.py` - Convert extraction to MemoryFact

### Tests
- ✅ `tests/extractors/test_llm_extractor.py` - Updated all tests
  - Removed `source` from mock JSON responses (18 occurrences)
  - Updated assertions to expect `source=None` (system-managed)
  - Fixed `test_extract_with_defaults` to include required `importance`
  - All 12 tests passing ✅

## Test Results

**Before**: 3 failures (source field issues, validation errors)
**After**: 12/12 tests passing ✅

**Comparison tool**: Still works correctly ✅

## Example JSON Schema Difference

### Before (MemoryFact - 13 fields):
```json
{
  "text": "...",
  "type": "...",
  "tags": [...],
  "importance": 0.5,  // Optional with default
  "source": null,     // System field visible to LLM
  "valid_until": null,
  "user_id": null,    // System field visible to LLM
  "confidence": 0.5,  // System field visible to LLM
  "mention_count": 1, // System field visible to LLM
  "first_seen": null,
  "last_seen": null,
  "archived": false,
  "archived_at": null,
  "superseded_by": null
}
```

### After (MemoryFactExtraction - 5 fields):
```json
{
  "text": "...",          // Required
  "type": "...",          // Required
  "tags": [...],          // Required
  "importance": 0.9,      // REQUIRED (no default!)
  "valid_until": null     // Optional
}
```

## Key Architectural Decision

**Extraction model is minimal and focused**:
- Only includes what LLM should decide
- System-managed fields are hidden from LLM
- Conversion happens in extractor layer
- Storage model (`MemoryFact`) remains unchanged for compatibility

This follows the **Single Responsibility Principle**: each model has one clear purpose.

## Impact on Other Components

- ✅ **No breaking changes** - MemoryFact API unchanged
- ✅ **Storage layer** - No changes needed
- ✅ **Classification pipeline** - No changes needed
- ✅ **Memory service** - No changes needed
- ✅ **Backwards compatible** - Existing code continues to work

## Related Issues Fixed

This refactoring also helped expose and fix:
- ❌ **80B model importance scoring bug** - Making `importance` required forced LLM to provide it
- ✅ **Clear field ownership** - System knows exactly which fields it manages
- ✅ **Better error messages** - Validation errors point to exact missing fields

## Lessons Learned

1. **JSON schema describes structure, not semantics** - Field descriptions matter!
2. **Optional with defaults can hide bugs** - Required fields force correct behavior
3. **Separation of concerns is critical** - Extraction ≠ Storage
4. **Test what the LLM should produce, not what the system adds**

## Next Steps

Phase 1 complete! Ready for:
- ✅ Phase 2: Implement importance-weighted auto-resolution
- ✅ Phase 3: Apply validated prompts to production
- ✅ Future: Entity extraction (transform tags → entity graphs)
