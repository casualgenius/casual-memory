# SQLAlchemy Conflict Store

The `SQLAlchemyConflictStore` provides a unified storage backend that works with any SQLAlchemy-compatible database (PostgreSQL, SQLite, MySQL, MariaDB, Oracle, etc.).

## Benefits

- **Single implementation** - One codebase for all database backends
- **Database agnostic** - Works with PostgreSQL, SQLite, MySQL, and more
- **Migration support** - Use Alembic for schema versioning
- **Type safety** - Uses SQLAlchemy ORM with Pydantic validation
- **Transaction support** - Automatic commit/rollback on errors
- **No duplication** - Replaces separate PostgreSQL and SQLite implementations

## Installation

```bash
# For SQLite (no additional dependencies)
pip install casual-memory[sqlalchemy]

# For PostgreSQL
pip install casual-memory[postgres]  # Includes SQLAlchemy + psycopg2

# For MySQL
pip install casual-memory[sqlalchemy] pymysql

# For all storage backends
pip install casual-memory[all]
```

## Usage

### Basic Usage (SQLite)

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore
from casual_memory.models import MemoryConflict

# Create engine
engine = create_engine("sqlite:///conflicts.db")

# Create store and initialize tables
store = SQLAlchemyConflictStore(engine)
store.create_tables()

# Use the store
conflict = MemoryConflict(
    id="conflict_123",
    user_id="user1",
    memory_a_id="mem_a",
    memory_b_id="mem_b",
    category="location",
    similarity_score=0.91,
    avg_importance=0.8,
    clarification_hint="Which location is correct?"
)

conflict_id = store.add_conflict(conflict)
retrieved = store.get_conflict(conflict_id)
```

### PostgreSQL Usage

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

# Create PostgreSQL engine
engine = create_engine(
    "postgresql://user:password@localhost:5432/memory_db",
    pool_size=10,
    max_overflow=20
)

# Create store (tables can be managed via Alembic migrations)
store = SQLAlchemyConflictStore(engine)
# store.create_tables()  # Or use Alembic migrations
```

### MySQL Usage

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

# Create MySQL engine
engine = create_engine(
    "mysql+pymysql://user:password@localhost:3306/memory_db",
    pool_pre_ping=True
)

store = SQLAlchemyConflictStore(engine)
store.create_tables()
```

### In-Memory Testing

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

# In-memory SQLite for testing
engine = create_engine("sqlite:///:memory:")
store = SQLAlchemyConflictStore(engine)
store.create_tables()
```

## Migration from Old Implementations

### From PostgresConflictStore

**Before:**
```python
from casual_memory.storage.conflicts.postgres import PostgresConflictStore

store = PostgresConflictStore(
    host="localhost",
    port=5432,
    database="memory_db",
    user="user",
    password="password"
)
```

**After:**
```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

engine = create_engine("postgresql://user:password@localhost:5432/memory_db")
store = SQLAlchemyConflictStore(engine)
```

### From SQLiteConflictStore

**Before:**
```python
from casual_memory.storage.conflicts.sqlite import SQLiteConflictStore

store = SQLiteConflictStore(db_path="conflicts.db")
```

**After:**
```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

engine = create_engine("sqlite:///conflicts.db")
store = SQLAlchemyConflictStore(engine)
store.create_tables()
```

## Integration with memory-store-service

The SQLAlchemy conflict store is designed to work seamlessly with your existing memory-store-service:

```python
# In memory-store-service/app/main.py or similar
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore
from app.config import settings

# Create engine from existing database config
engine = create_engine(
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

# Use the same store implementation for all databases
conflict_store = SQLAlchemyConflictStore(engine)
```

This allows you to:
- Use the same conflict store implementation across development (SQLite) and production (PostgreSQL)
- Manage schema migrations with Alembic
- Share the database engine with other parts of your application

## Schema Management

### Option 1: Auto-create (Simple Projects)

```python
store.create_tables()  # Creates tables if they don't exist
```

### Option 2: Alembic Migrations (Production)

```bash
# In your service directory
alembic init alembic
```

Create a migration:
```python
# alembic/versions/001_create_conflicts_table.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'conflicts',
        sa.Column('id', sa.String, primary_key=True),
        sa.Column('user_id', sa.String, nullable=False, index=True),
        sa.Column('memory_a_id', sa.String, nullable=False),
        sa.Column('memory_b_id', sa.String, nullable=False),
        sa.Column('category', sa.String, nullable=False),
        sa.Column('is_singleton_category', sa.Boolean, nullable=False, default=False),
        sa.Column('similarity_score', sa.Float, nullable=False),
        sa.Column('status', sa.String, nullable=False, default='pending', index=True),
        sa.Column('avg_importance', sa.Float, nullable=False),
        sa.Column('clarification_hint', sa.Text, nullable=False),
        sa.Column('resolution_type', sa.String, nullable=True),
        sa.Column('winning_memory_id', sa.String, nullable=True),
        sa.Column('resolution_attempts', sa.Integer, nullable=False, default=0),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('resolved_at', sa.DateTime, nullable=True),
        sa.Column('metadata_json', sa.Text, nullable=False, default='{}'),
    )

    op.create_index('idx_conflicts_user_status', 'conflicts', ['user_id', 'status'])

def downgrade():
    op.drop_table('conflicts')
```

Apply migrations:
```bash
alembic upgrade head
```

## API Reference

The `SQLAlchemyConflictStore` implements the `ConflictStore` protocol with these methods:

- `add_conflict(conflict: MemoryConflict) -> str` - Store a new conflict
- `get_conflict(conflict_id: str) -> Optional[MemoryConflict]` - Retrieve by ID
- `get_pending_conflicts(user_id: str, limit: Optional[int] = None) -> List[MemoryConflict]` - Get pending conflicts
- `resolve_conflict(conflict_id: str, resolution: ConflictResolution) -> bool` - Resolve a conflict
- `get_conflict_count(user_id: str, status: Optional[str] = None) -> int` - Count conflicts
- `escalate_conflict(conflict_id: str) -> bool` - Escalate unresolvable conflicts
- `clear_user_conflicts(user_id: str, status: Optional[str] = None) -> int` - Clear conflicts

## Database Schema

The conflicts table schema:

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(PK) | Unique conflict identifier |
| user_id | VARCHAR(indexed) | User this conflict belongs to |
| memory_a_id | VARCHAR | ID of first conflicting memory |
| memory_b_id | VARCHAR | ID of second conflicting memory |
| category | VARCHAR | Conflict category (location, job, etc.) |
| is_singleton_category | BOOLEAN | Whether only one memory allowed |
| similarity_score | FLOAT | Vector similarity score (0.0-1.0) |
| status | VARCHAR(indexed) | Status: pending/resolved/escalated |
| avg_importance | FLOAT | Average importance (0.0-1.0) |
| clarification_hint | TEXT | Suggested clarification question |
| resolution_type | VARCHAR | How resolved: manual/automated/conversational |
| winning_memory_id | VARCHAR | ID of kept memory after resolution |
| resolution_attempts | INTEGER | Number of resolution attempts |
| created_at | DATETIME | When conflict was detected |
| resolved_at | DATETIME | When conflict was resolved |
| metadata_json | TEXT | Additional metadata (JSON) |

## Performance Considerations

### Connection Pooling

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=10,        # Number of connections to keep open
    max_overflow=20,     # Additional connections when pool exhausted
    pool_pre_ping=True   # Verify connections before using
)
```

### Query Optimization

The store automatically creates indexes for:
- `user_id` - Fast user-specific queries
- `status` - Fast status filtering
- `(user_id, status)` - Composite index for combined queries

### Transaction Management

Transactions are automatically managed with proper rollback on errors:

```python
# If any error occurs, the transaction is automatically rolled back
try:
    store.add_conflict(conflict)
except Exception as e:
    # Transaction already rolled back, database state is consistent
    logger.error(f"Failed to add conflict: {e}")
```

## Testing

The SQLAlchemy conflict store includes comprehensive tests that work across all database backends:

```bash
# Run all storage tests
pytest tests/storage/

# Run only SQLAlchemy tests
pytest tests/storage/conflicts/test_sqlalchemy_conflict_store.py -v

# Test with different databases
# SQLite
DATABASE_URL=sqlite:///test.db pytest tests/storage/conflicts/test_sqlalchemy_conflict_store.py

# PostgreSQL
DATABASE_URL=postgresql://localhost/test_db pytest tests/storage/conflicts/test_sqlalchemy_conflict_store.py
```

All 17 tests cover:
- Basic CRUD operations
- User isolation
- Conflict resolution workflows
- Transaction rollback
- Metadata serialization
- Query limits and sorting

## See Also

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [ConflictStore Protocol](../src/casual_memory/storage/protocols.py)
