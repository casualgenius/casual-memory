"""
SQLAlchemy-based conflict storage implementation.

Provides a unified storage backend that works with any SQLAlchemy-compatible
database (PostgreSQL, SQLite, MySQL, etc.). Uses SQLModel for type-safe
ORM operations with Pydantic validation.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Engine, Float, Index, Integer, String, Text
from sqlalchemy.orm import Session, declarative_base

from casual_memory.models import ConflictResolution, MemoryConflict

logger = logging.getLogger(__name__)

# Create SQLAlchemy Base
Base = declarative_base()


class ConflictDB(Base):
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
    clarification_hint = Column(Text, nullable=False)

    # Resolution details
    resolution_type = Column(String, nullable=True)
    winning_memory_id = Column(String, nullable=True)
    resolution_attempts = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    resolved_at = Column(DateTime, nullable=True)

    # Metadata (JSON serialized)
    metadata_json = Column(Text, nullable=False, default="{}")

    # Indexes
    __table_args__ = (Index("idx_conflicts_user_status", "user_id", "status"),)

    def to_memory_conflict(self) -> MemoryConflict:
        """Convert database model to MemoryConflict."""
        metadata = json.loads(self.metadata_json) if self.metadata_json else {}

        return MemoryConflict(
            id=self.id,
            user_id=self.user_id,
            memory_a_id=self.memory_a_id,
            memory_b_id=self.memory_b_id,
            category=self.category,
            is_singleton_category=self.is_singleton_category,
            similarity_score=self.similarity_score,
            status=self.status,
            avg_importance=self.avg_importance,
            clarification_hint=self.clarification_hint,
            resolution_type=self.resolution_type,
            winning_memory_id=self.winning_memory_id,
            resolution_attempts=self.resolution_attempts,
            created_at=self.created_at,
            resolved_at=self.resolved_at,
            metadata=metadata,
        )

    @staticmethod
    def from_memory_conflict(conflict: MemoryConflict) -> "ConflictDB":
        """Create database model from MemoryConflict."""
        return ConflictDB(
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
            metadata_json=json.dumps(conflict.metadata),
        )


class SQLAlchemyConflictStore:
    """
    SQLAlchemy-based conflict storage.

    Works with any SQLAlchemy-compatible database including PostgreSQL,
    SQLite, MySQL, and more. Provides persistence, transactions, and
    multi-instance support.

    Example:
        # PostgreSQL
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:pass@localhost/db")
        store = SQLAlchemyConflictStore(engine)
        store.create_tables()

        # SQLite
        engine = create_engine("sqlite:///conflicts.db")
        store = SQLAlchemyConflictStore(engine)
        store.create_tables()
    """

    def __init__(self, engine: Engine):
        """
        Initialize the SQLAlchemy conflict store.

        Args:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine
        logger.info(f"SQLAlchemyConflictStore initialized (engine={engine.url})")

    @contextmanager
    def _session(self):
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

    def create_tables(self):
        """Create database tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def add_conflict(self, conflict: MemoryConflict) -> str:
        """Store a detected conflict."""
        with self._session() as session:
            db_conflict = ConflictDB.from_memory_conflict(conflict)
            session.add(db_conflict)

            logger.info(
                f"Stored conflict {conflict.id} for user {conflict.user_id}: "
                f"{conflict.category} ({conflict.status})"
            )

            return conflict.id

    def get_conflict(self, conflict_id: str) -> Optional[MemoryConflict]:
        """Retrieve a conflict by ID."""
        with self._session() as session:
            db_conflict = session.query(ConflictDB).filter(ConflictDB.id == conflict_id).first()

            if not db_conflict:
                return None

            return db_conflict.to_memory_conflict()

    def get_pending_conflicts(
        self, user_id: str, limit: Optional[int] = None
    ) -> List[MemoryConflict]:
        """Get all pending conflicts for a user."""
        with self._session() as session:
            query = (
                session.query(ConflictDB)
                .filter(ConflictDB.user_id == user_id, ConflictDB.status == "pending")
                .order_by(ConflictDB.avg_importance.desc())
            )

            if limit:
                query = query.limit(limit)

            db_conflicts = query.all()
            return [db_conflict.to_memory_conflict() for db_conflict in db_conflicts]

    def resolve_conflict(self, conflict_id: str, resolution: ConflictResolution) -> bool:
        """Mark a conflict as resolved."""
        with self._session() as session:
            db_conflict = session.query(ConflictDB).filter(ConflictDB.id == conflict_id).first()

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
            db_conflict.status = "resolved"
            db_conflict.resolved_at = datetime.now()
            db_conflict.resolution_type = resolution.resolution_type
            db_conflict.winning_memory_id = winning_memory_id

            # Update metadata
            metadata = json.loads(db_conflict.metadata_json) if db_conflict.metadata_json else {}
            metadata.update(
                {
                    "resolution_decision": resolution.decision,
                    "resolution_notes": resolution.notes,
                    "resolved_by": resolution.resolved_by,
                    "resolved_at": datetime.now().isoformat(),
                }
            )
            db_conflict.metadata_json = json.dumps(metadata)

            logger.info(
                f"Resolved conflict {conflict_id}: {resolution.decision} "
                f"(by {resolution.resolved_by})"
            )

            return True

    def get_conflict_count(self, user_id: str, status: Optional[str] = None) -> int:
        """Count conflicts for a user."""
        with self._session() as session:
            query = session.query(ConflictDB).filter(ConflictDB.user_id == user_id)

            if status:
                query = query.filter(ConflictDB.status == status)

            return query.count()

    def escalate_conflict(self, conflict_id: str) -> bool:
        """Escalate a conflict that couldn't be auto-resolved."""
        with self._session() as session:
            db_conflict = session.query(ConflictDB).filter(ConflictDB.id == conflict_id).first()

            if not db_conflict:
                logger.warning(f"Cannot escalate conflict {conflict_id}: not found")
                return False

            db_conflict.status = "escalated"
            db_conflict.resolution_attempts += 1

            logger.info(
                f"Escalated conflict {conflict_id} " f"(attempts={db_conflict.resolution_attempts})"
            )

            return True

    def clear_user_conflicts(self, user_id: str, status: Optional[str] = None) -> int:
        """Clear conflicts for a user."""
        with self._session() as session:
            query = session.query(ConflictDB).filter(ConflictDB.user_id == user_id)

            if status:
                query = query.filter(ConflictDB.status == status)

            count = query.delete()

            logger.info(
                f"Cleared {count} conflicts for user {user_id} " f"(status={status or 'all'})"
            )

            return count
