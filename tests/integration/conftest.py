"""Fixtures and helpers for integration tests."""

import socket

import pytest


def is_service_available(host: str, port: int) -> bool:
    """Check if a service is available at host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


@pytest.fixture
def skip_if_no_qdrant():
    """Skip test if Qdrant is not available."""
    if not is_service_available("localhost", 6333):
        pytest.skip("Qdrant not available at localhost:6333")


@pytest.fixture
def skip_if_no_postgres():
    """Skip test if PostgreSQL is not available."""
    if not is_service_available("localhost", 5432):
        pytest.skip("PostgreSQL not available at localhost:5432")


@pytest.fixture
def skip_if_no_redis():
    """Skip test if Redis is not available."""
    if not is_service_available("localhost", 6379):
        pytest.skip("Redis not available at localhost:6379")
