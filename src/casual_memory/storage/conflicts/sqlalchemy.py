"""
SQLAlchemy-based conflict storage implementation.

Provides a unified storage backend that works with any SQLAlchemy-compatible
database (PostgreSQL, SQLite, MySQL, etc.). Supports configurable schema
for multi-tenant deployments.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional, Type

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Text,
)
from sqlalchemy.orm import Session, declarative_base

from casual_memory.models import ConflictResolution, MemoryConflict

logger = logging.getLogger(__name__)


def create_conflict_model(
    schema: Optional[str] = None,
) -> tuple[Any, Type["ConflictDBBase"]]:
    """
    Factory function to create ConflictDB model with optional schema.

    Args:
        schema: Optional database schema name (e.g., 'memory_store')

    Returns:
        Tuple of (Base, ConflictDB class)
    """
    metadata = MetaData(schema=schema) if schema else MetaData()
    base = declarative_base(metadata=metadata)

    class ConflictDB(base):  # type: ignore[misc, valid-type]
        """SQLAlchemy model for conflict storage."""

        __tablename__ = "conflicts"

        # Primary key
        id = Column(String, primary_key=True)

        # Conflict details
        user_id = Column(String, nullable=False, index=True)
        memory_a_id = Column(String, nullable=False)
        memory_b_id = Column(String, nullable=False)
        category = Column(String, nullable=False)
        is_singleton_category = Column(Boolean, nullable=False, default=False)
        similarity_score = Column(Float, nullable=False)

        # Status tracking
        status = Column(String, nullable=False, default="pending", index=True)
        avg_importance = Column(Float, nullable=False)
        clarification_hint = Column(Text, nullable=True)  # Nullable for backwards compat

        # Resolution details
        resolution_type = Column(String, nullable=True)
        winning_memory_id = Column(String, nullable=True)
        resolution_attempts = Column(Integer, nullable=False, default=0)

        # Timestamps
        created_at = Column(DateTime, nullable=False, default=datetime.now)
        resolved_at = Column(DateTime, nullable=True)

        # Metadata - named conflict_metadata for backwards compat
        conflict_metadata = Column(JSON, nullable=False, default={})

        # Indexes
        __table_args__ = (
            Index("idx_conflicts_user_status", "user_id", "status"),
            {"schema": schema} if schema else {},
        )

        def to_memory_conflict(self) -> MemoryConflict:
            """Convert database model to MemoryConflict."""
            return MemoryConflict(
                id=self.id,  # type: ignore[arg-type]
                user_id=self.user_id,  # type: ignore[arg-type]
                memory_a_id=self.memory_a_id,  # type: ignore[arg-type]
                memory_b_id=self.memory_b_id,  # type: ignore[arg-type]
                category=self.category,  # type: ignore[arg-type]
                is_singleton_category=self.is_singleton_category,  # type: ignore[arg-type]
                similarity_score=self.similarity_score,  # type: ignore[arg-type]
                status=self.status,  # type: ignore[arg-type]
                avg_importance=self.avg_importance,  # type: ignore[arg-type]
                clarification_hint=self.clarification_hint,  # type: ignore[arg-type]
                resolution_type=self.resolution_type,  # type: ignore[arg-type]
                winning_memory_id=self.winning_memory_id,  # type: ignore[arg-type]
                resolution_attempts=self.resolution_attempts,  # type: ignore[arg-type]
                created_at=self.created_at,  # type: ignore[arg-type]
                resolved_at=self.resolved_at,  # type: ignore[arg-type]
                metadata=self.conflict_metadata or {},
            )

        @classmethod
        def from_memory_conflict(cls, conflict: MemoryConflict) -> "ConflictDB":
            """Create database model from MemoryConflict."""
            return cls(
                id=conflict.id,
                user_id=conflict.user_id,
                memory_a_id=conflict.memory_a_id,
                memory_b_id=conflict.memory_b_id,
                category=conflict.category,
                is_singleton_category=conflict.is_singleton_category,
                similarity_score=conflict.similarity_score,
                status=conflict.status,
                avg_importance=conflict.avg_importance,
                clarification_hint=conflict.clarification_hint,
                resolution_type=conflict.resolution_type,
                winning_memory_id=conflict.winning_memory_id,
                resolution_attempts=conflict.resolution_attempts,
                created_at=conflict.created_at or datetime.now(),
                resolved_at=conflict.resolved_at,
                conflict_metadata=conflict.metadata,
            )

    return base, ConflictDB


# Type alias for the ConflictDB base class (for type hints)
class ConflictDBBase:
    """Type stub for ConflictDB instances."""

    id: str
    user_id: str
    memory_a_id: str
    memory_b_id: str
    category: str
    is_singleton_category: bool
    similarity_score: float
    status: str
    avg_importance: float
    clarification_hint: Optional[str]
    resolution_type: Optional[str]
    winning_memory_id: Optional[str]
    resolution_attempts: int
    created_at: datetime
    resolved_at: Optional[datetime]
    conflict_metadata: str

    def to_memory_conflict(self) -> MemoryConflict: ...
    @classmethod
    def from_memory_conflict(cls, conflict: MemoryConflict) -> "ConflictDBBase": ...


class SQLAlchemyConflictStore:
    """
    SQLAlchemy-based conflict storage.

    Works with any SQLAlchemy-compatible database including PostgreSQL,
    SQLite, MySQL, and more. Provides persistence, transactions, and
    multi-instance support.

    Example:
        # PostgreSQL with schema
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:pass@localhost/db")
        store = SQLAlchemyConflictStore(engine, schema="memory_store")
        store.create_tables()

        # SQLite (no schema support)
        engine = create_engine("sqlite:///conflicts.db")
        store = SQLAlchemyConflictStore(engine)
        store.create_tables()
    """

    def __init__(self, engine: Engine, schema: Optional[str] = None):
        """
        Initialize the SQLAlchemy conflict store.

        Args:
            engine: SQLAlchemy engine for database connection
            schema: Optional database schema name (e.g., 'memory_store').
                   Use this for PostgreSQL deployments with schema isolation.
        """
        self.engine = engine
        self.schema = schema
        self._base, self._conflict_db = create_conflict_model(schema)
        logger.info(f"SQLAlchemyConflictStore initialized (engine={engine.url}, schema={schema})")

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic commit/rollback."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        self._base.metadata.create_all(self.engine)
        logger.info(f"Database tables created/verified (schema={self.schema})")

    def add_conflict(self, conflict: MemoryConflict) -> str:
        """Store a detected conflict."""
        with self._session() as session:
            db_conflict = self._conflict_db.from_memory_conflict(conflict)
            session.add(db_conflict)

            logger.info(
                f"Stored conflict {conflict.id} for user {conflict.user_id}: "
                f"{conflict.category} ({conflict.status})"
            )

            return conflict.id

    def get_conflict(self, conflict_id: str) -> Optional[MemoryConflict]:
        """Retrieve a conflict by ID."""
        with self._session() as session:
            db_conflict = (
                session.query(self._conflict_db).filter(self._conflict_db.id == conflict_id).first()
            )

            if not db_conflict:
                return None

            return db_conflict.to_memory_conflict()

    def get_pending_conflicts(
        self, user_id: str, limit: Optional[int] = None
    ) -> list[MemoryConflict]:
        """Get all pending conflicts for a user."""
        with self._session() as session:
            query = (
                session.query(self._conflict_db)
                .filter(
                    self._conflict_db.user_id == user_id,
                    self._conflict_db.status == "pending",
                )
                .order_by(self._conflict_db.avg_importance.desc())
            )

            if limit:
                query = query.limit(limit)

            db_conflicts = query.all()
            return [db_conflict.to_memory_conflict() for db_conflict in db_conflicts]

    def resolve_conflict(self, conflict_id: str, resolution: ConflictResolution) -> bool:
        """Mark a conflict as resolved."""
        with self._session() as session:
            db_conflict = (
                session.query(self._conflict_db).filter(self._conflict_db.id == conflict_id).first()
            )

            if not db_conflict:
                logger.warning(f"Cannot resolve conflict {conflict_id}: not found")
                return False

            # Determine winning memory
            winning_memory_id = None
            if resolution.decision == "keep_a":
                winning_memory_id = db_conflict.memory_a_id
            elif resolution.decision == "keep_b":
                winning_memory_id = db_conflict.memory_b_id

            # Update conflict
            db_conflict.status = "resolved"  # type: ignore[assignment]
            db_conflict.resolved_at = datetime.now()  # type: ignore[assignment]
            db_conflict.resolution_type = resolution.resolution_type  # type: ignore[assignment]
            db_conflict.winning_memory_id = winning_memory_id  # type: ignore[assignment]

            # Update metadata
            metadata = dict(db_conflict.conflict_metadata or {})
            metadata.update(
                {
                    "resolution_decision": resolution.decision,
                    "resolution_notes": resolution.notes,
                    "resolved_by": resolution.resolved_by,
                    "resolved_at": datetime.now().isoformat(),
                }
            )
            db_conflict.conflict_metadata = metadata  # type: ignore[assignment]

            logger.info(
                f"Resolved conflict {conflict_id}: {resolution.decision} "
                f"(by {resolution.resolved_by})"
            )

            return True

    def get_conflict_count(self, user_id: str, status: Optional[str] = None) -> int:
        """Count conflicts for a user."""
        with self._session() as session:
            query = session.query(self._conflict_db).filter(self._conflict_db.user_id == user_id)

            if status:
                query = query.filter(self._conflict_db.status == status)

            return query.count()

    def escalate_conflict(self, conflict_id: str) -> bool:
        """Escalate a conflict that couldn't be auto-resolved."""
        with self._session() as session:
            db_conflict = (
                session.query(self._conflict_db).filter(self._conflict_db.id == conflict_id).first()
            )

            if not db_conflict:
                logger.warning(f"Cannot escalate conflict {conflict_id}: not found")
                return False

            db_conflict.status = "escalated"  # type: ignore[assignment]
            db_conflict.resolution_attempts += 1  # type: ignore[assignment]

            logger.info(
                f"Escalated conflict {conflict_id} " f"(attempts={db_conflict.resolution_attempts})"
            )

            return True

    def clear_user_conflicts(self, user_id: str, status: Optional[str] = None) -> int:
        """Clear conflicts for a user."""
        with self._session() as session:
            query = session.query(self._conflict_db).filter(self._conflict_db.user_id == user_id)

            if status:
                query = query.filter(self._conflict_db.status == status)

            count = query.delete()

            logger.info(
                f"Cleared {count} conflicts for user {user_id} " f"(status={status or 'all'})"
            )

            return count
