"""Fixtures and helpers for integration tests."""

import os
import socket

import pytest


def get_docker_host() -> str:
    """
    Get the host to use for Docker services.

    Handles Docker-in-Docker scenarios by checking:
    1. INTEGRATION_TEST_HOST environment variable
    2. localhost
    3. Docker host gateway (for running inside a container)
    """
    # Allow override via environment variable
    if host := os.environ.get("INTEGRATION_TEST_HOST"):
        return host

    # Try localhost first (normal case)
    if is_service_available("localhost", 6333):
        return "localhost"

    # Try Docker host gateway (Docker-in-Docker case)
    gateway = _get_docker_gateway()
    if gateway and is_service_available(gateway, 6333):
        return gateway

    # Default to localhost
    return "localhost"


def _get_docker_gateway() -> str | None:
    """Get the Docker host gateway IP from routing table."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:  # Skip header
                fields = line.strip().split()
                if fields[1] == "00000000":  # Default route
                    # Gateway is in hex, little-endian
                    hex_gateway = fields[2]
                    # Convert hex to IP (little-endian)
                    gateway_bytes = bytes.fromhex(hex_gateway)
                    return ".".join(str(b) for b in reversed(gateway_bytes))
    except Exception:
        pass
    return None


def is_service_available(host: str, port: int) -> bool:
    """Check if a service is available at host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


# Determine the host once at module load
DOCKER_HOST = get_docker_host()


@pytest.fixture
def integration_host() -> str:
    """Return the host to use for integration tests."""
    return DOCKER_HOST


@pytest.fixture
def skip_if_no_qdrant(integration_host):
    """Skip test if Qdrant is not available."""
    if not is_service_available(integration_host, 6333):
        pytest.skip(f"Qdrant not available at {integration_host}:6333")
    return integration_host


@pytest.fixture
def skip_if_no_postgres(integration_host):
    """Skip test if PostgreSQL is not available."""
    if not is_service_available(integration_host, 5432):
        pytest.skip(f"PostgreSQL not available at {integration_host}:5432")
    return integration_host


@pytest.fixture
def skip_if_no_redis(integration_host):
    """Skip test if Redis is not available."""
    if not is_service_available(integration_host, 6379):
        pytest.skip(f"Redis not available at {integration_host}:6379")
    return integration_host
