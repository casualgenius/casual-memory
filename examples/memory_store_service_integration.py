"""
Example: Integrating SQLAlchemy Conflict Store with memory-store-service

This example shows how to replace the existing PostgreSQL implementation
in memory-store-service with the unified SQLAlchemy conflict store.
"""

# ==============================================================================
# BEFORE: Using separate PostgreSQL implementation
# ==============================================================================

# memory-store-service/app/conflict/detector.py (OLD)
"""
from app.database.connection import get_db_session
from app.database.models import ConflictDB

class ConflictDetector:
    def __init__(self):
        self.session = get_db_session()

    def store_conflict(self, conflict):
        db_conflict = ConflictDB(
            id=conflict.id,
            user_id=conflict.user_id,
            # ... map all fields
        )
        self.session.add(db_conflict)
        self.session.commit()
"""

# ==============================================================================
# AFTER: Using unified SQLAlchemy implementation from casual-memory
# ==============================================================================

# memory-store-service/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "memory_store"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"

    # For SQLite development
    USE_SQLITE_DEV: bool = False
    SQLITE_PATH: str = "dev_conflicts.db"

    @property
    def database_url(self) -> str:
        """Get database URL based on environment."""
        if self.USE_SQLITE_DEV:
            return f"sqlite:///{self.SQLITE_PATH}"
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()

# ==============================================================================
# memory-store-service/app/database/connection.py
# ==============================================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create engine with connection pooling
engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==============================================================================
# memory-store-service/app/main.py
# ==============================================================================

from app.database.connection import engine
from fastapi import FastAPI

from casual_memory.models import ConflictResolution, MemoryConflict
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

app = FastAPI()

# Initialize conflict store with shared engine
conflict_store = SQLAlchemyConflictStore(engine)


# Create tables on startup (or use Alembic migrations)
@app.on_event("startup")
async def startup_event():
    # Option 1: Auto-create tables (simple projects)
    conflict_store.create_tables()

    # Option 2: Use Alembic migrations (production)
    # alembic upgrade head


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.post("/conflicts")
async def create_conflict(conflict: MemoryConflict):
    """Store a new conflict."""
    conflict_id = conflict_store.add_conflict(conflict)
    return {"conflict_id": conflict_id}


@app.get("/conflicts/pending/{user_id}")
async def get_pending_conflicts(user_id: str, limit: int = 10):
    """Get pending conflicts for a user."""
    conflicts = conflict_store.get_pending_conflicts(user_id, limit=limit)
    return {"conflicts": conflicts}


@app.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: str, resolution: ConflictResolution):
    """Resolve a conflict."""
    success = conflict_store.resolve_conflict(conflict_id, resolution)
    return {"success": success}


@app.get("/conflicts/{conflict_id}")
async def get_conflict(conflict_id: str):
    """Get a specific conflict."""
    conflict = conflict_store.get_conflict(conflict_id)
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return conflict


# ==============================================================================
# Usage in existing services
# ==============================================================================

# memory-store-service/app/conflict/detector.py (NEW)
from app.database.connection import engine

from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore


class ConflictDetector:
    def __init__(self):
        self.conflict_store = SQLAlchemyConflictStore(engine)

    def store_conflict(self, conflict: MemoryConflict) -> str:
        """Store a detected conflict."""
        return self.conflict_store.add_conflict(conflict)

    def get_pending_for_user(self, user_id: str) -> list[MemoryConflict]:
        """Get all pending conflicts for resolution."""
        return self.conflict_store.get_pending_conflicts(user_id)


# ==============================================================================
# Development vs Production Configuration
# ==============================================================================

# .env.development
"""
USE_SQLITE_DEV=true
SQLITE_PATH=dev_conflicts.db
"""

# .env.production
"""
USE_SQLITE_DEV=false
POSTGRES_HOST=postgres-service
POSTGRES_PORT=5432
POSTGRES_DB=memory_store
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<secret>
"""

# ==============================================================================
# Alembic Migration (Optional, for production)
# ==============================================================================

# alembic/versions/001_create_conflicts_table.py
"""
Create conflicts table

Revision ID: 001
"""
import sqlalchemy as sa
from alembic import op


def upgrade():
    op.create_table(
        "conflicts",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("user_id", sa.String, nullable=False, index=True),
        sa.Column("memory_a_id", sa.String, nullable=False),
        sa.Column("memory_b_id", sa.String, nullable=False),
        sa.Column("category", sa.String, nullable=False),
        sa.Column("is_singleton_category", sa.Boolean, nullable=False, default=False),
        sa.Column("similarity_score", sa.Float, nullable=False),
        sa.Column("status", sa.String, nullable=False, default="pending", index=True),
        sa.Column("avg_importance", sa.Float, nullable=False),
        sa.Column("clarification_hint", sa.Text, nullable=False),
        sa.Column("resolution_type", sa.String, nullable=True),
        sa.Column("winning_memory_id", sa.String, nullable=True),
        sa.Column("resolution_attempts", sa.Integer, nullable=False, default=0),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("resolved_at", sa.DateTime, nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=False, default="{}"),
    )

    op.create_index("idx_conflicts_user_status", "conflicts", ["user_id", "status"])


def downgrade():
    op.drop_table("conflicts")


# ==============================================================================
# Benefits of This Approach
# ==============================================================================

"""
1. **Single Implementation**
   - One conflict store for all environments
   - No separate PostgreSQL/SQLite implementations

2. **Flexible Development**
   - Use SQLite for local development
   - Switch to PostgreSQL for production
   - Same code, different database URL

3. **Production Ready**
   - Transaction support with automatic rollback
   - Connection pooling
   - Proper error handling

4. **Easy Migration**
   - Use Alembic for schema versioning
   - Apply migrations across all environments

5. **Type Safe**
   - SQLAlchemy ORM with Pydantic models
   - Compile-time type checking
   - IDE autocomplete

6. **Maintainable**
   - Less code duplication
   - Single source of truth
   - Easier to update and test
"""

# ==============================================================================
# Testing
# ==============================================================================

import pytest
from sqlalchemy import create_engine

from casual_memory.models import MemoryConflict
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore


@pytest.fixture
def test_store():
    """Create in-memory SQLite store for testing."""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyConflictStore(engine)
    store.create_tables()
    return store


def test_conflict_workflow(test_store):
    """Test full conflict lifecycle."""
    # Create conflict
    conflict = MemoryConflict(
        id="test_conflict",
        user_id="test_user",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="location",
        similarity_score=0.91,
        avg_importance=0.8,
        clarification_hint="Which location?",
    )

    # Store it
    conflict_id = test_store.add_conflict(conflict)
    assert conflict_id == "test_conflict"

    # Retrieve it
    retrieved = test_store.get_conflict(conflict_id)
    assert retrieved.user_id == "test_user"

    # Resolve it
    resolution = ConflictResolution(
        conflict_id=conflict_id, decision="keep_a", resolution_type="manual", resolved_by="user"
    )
    success = test_store.resolve_conflict(conflict_id, resolution)
    assert success is True

    # Verify resolution
    resolved = test_store.get_conflict(conflict_id)
    assert resolved.status == "resolved"
    assert resolved.winning_memory_id == "mem_a"
