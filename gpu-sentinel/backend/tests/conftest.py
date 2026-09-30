import pytest
from fastapi.testclient import TestClient

from sentinel.app import create_app
from sentinel.config import Settings
from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.sources import SimulatorSource


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path}/test.db", llm_provider="template", analysis_interval_s=10,
                    jwt_secret="test-secret-test-secret-test-secret", demo_mode=True, persist_every_cycles=1)


@pytest.fixture
def sim():
    return ClusterSimulator(seed=42)


@pytest.fixture
def client(settings, sim):
    src = SimulatorSource(sim, dt=10, warmup_s=60, virtual_time=True)
    app = create_app(settings, autostart=False, source=src)
    with TestClient(app) as c:
        c.sim = sim
        c.ctx = app.state.ctx

        def cycles(n: int = 1, explain: bool = False):
            for _ in range(n):
                c.portal.call(lambda: c.ctx.engine.run_cycle(explain=explain))

        c.cycles = cycles
        yield c


def login(client, user="admin", password="sentinel-admin") -> dict:
    r = client.post("/api/v1/auth/login", json={"username": user, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def admin(client):
    return login(client)


@pytest.fixture
def operator(client):
    return login(client, "operator", "sentinel-operator")


@pytest.fixture
def viewer(client):
    return login(client, "viewer", "sentinel-viewer")
