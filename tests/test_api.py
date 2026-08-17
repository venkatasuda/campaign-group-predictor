"""Integration tests for the HTTP layer."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings
from src.constants import LEAKAGE_FEATURES
from src.predictors import MajorityClassPredictor


class TestOperationsEndpoints:
    def test_root_returns_a_banner(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["docs"] == "/docs"

    def test_health_reports_degraded_when_serving_the_baseline(self, client: TestClient) -> None:
        """The fixture points at a missing artifact, so the baseline fallback loads.

        This test previously asserted ``status == "ok"`` and passed - which meant the
        suite was encoding the bug rather than catching it. A predictor being *loaded* is
        not the same as the *trained model* being loaded, and reporting the difference as
        healthy is the worst available failure mode: probes pass, traffic is routed, and
        campaign budget is allocated by a rule that is right 46% of the time.
        """
        body = client.get("/health").json()

        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["serving_baseline"] is True

    def test_health_reports_ok_for_a_real_model(
        self, settings: Settings, trained_artifact: Path
    ) -> None:
        """The positive case, built around a genuinely loaded artifact.

        Note this cannot reuse the `client_with_model` fixture: that overrides the injected
        predictor, while `/health` reports on the *registry*. Testing the endpoint through
        a dependency override would have asserted "ok" while the registry held the
        fallback - the same confusion the endpoint itself used to make.
        """
        app = create_app(settings.model_copy(update={"model_path": str(trained_artifact)}))
        with TestClient(app) as client:
            body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["serving_baseline"] is False

    def test_model_info_lists_the_feature_contract(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert len(body["expected_features"]["group_1"]) == 20
        assert len(body["expected_features"]["comparison"]) == 27
        assert body["excluded_leakage_features"] == LEAKAGE_FEATURES

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestPredictEndpoint:
    def test_returns_a_valid_prediction(self, client: TestClient, valid_payload: dict) -> None:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        body = response.json()
        assert body["predicted_class"] in (0, 1, 2)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["recommended_action"]

    def test_uses_the_injected_predictor(
        self, client_with_model: TestClient, valid_payload: dict
    ) -> None:
        body = client_with_model.post("/predict", json=valid_payload).json()
        assert body["predicted_class"] == 2
        assert body["label"] == "group_2"
        assert body["confidence"] == 0.6

    def test_missing_feature_returns_422(self, client: TestClient, valid_payload: dict) -> None:
        del valid_payload["group_1"]["g1_1"]
        assert client.post("/predict", json=valid_payload).status_code == 422

    def test_post_campaign_variable_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["group_2"]["g2_21"] = 1.0
        assert client.post("/predict", json=valid_payload).status_code == 422

    def test_wrong_type_returns_422(self, client: TestClient, valid_payload: dict) -> None:
        valid_payload["group_1"]["g1_1"] = "text"
        assert client.post("/predict", json=valid_payload).status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        assert client.post("/predict", json={}).status_code == 422

    def test_null_value_is_accepted_and_imputed(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["comparison"]["c_1"] = None
        assert client.post("/predict", json=valid_payload).status_code == 200


class TestLivenessAndReadiness:
    """Two probes, because an orchestrator reacts to them differently.

    A failed liveness probe means *restart this container*. A failed readiness probe means
    *stop sending it traffic*. Conflating them makes both wrong: a liveness check that also
    verifies the model would restart a container whose only problem is a missing artifact,
    producing a crash loop that fixes nothing.
    """

    def test_live_reports_alive_even_without_a_model(self, client: TestClient) -> None:
        """The fixture points at a missing artifact, so the baseline is loaded.

        Liveness must still be 200: the process is running, and restarting it would not
        produce a model.
        """
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_ready_returns_503_when_serving_the_baseline(self, client: TestClient) -> None:
        """The status code is the whole point of this endpoint.

        `/health` reports `degraded` inside a 200 because it is read by humans. A load
        balancer reads only the code, so a degraded instance answering 200 keeps receiving
        traffic it should not have.
        """
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert "baseline" in response.json()["reason"]

    def test_ready_returns_200_with_a_real_model(self, tmp_path) -> None:
        from fastapi.testclient import TestClient as Client

        from src.api.dependencies import get_registry
        from src.api.main import create_app
        from src.config import Settings

        class _RealModelStub:
            """Stands in for a loaded artifact.

            Written as a plain stub rather than by mutating a MajorityClassPredictor:
            `metadata` on the real predictors is a property that rebuilds its dict on each
            access, so assigning into it has no effect and the object stays flagged as a
            baseline. An earlier version of this test did exactly that and asserted 200
            against an instance that was still, correctly, reporting 503.
            """

            metadata = {"is_baseline": False, "predictor": "sklearn_pipeline"}

        class _ReadyRegistry:
            is_loaded = True
            predictor = _RealModelStub()

        app = create_app(
            Settings(model_path=str(tmp_path / "missing.pkl"), allow_baseline_fallback=True)
        )
        app.dependency_overrides[get_registry] = lambda: _ReadyRegistry()
        with Client(app) as ready_client:
            response = ready_client.get("/ready")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["ready"] is True


class TestStartupCanary:
    """Loading an artifact and being able to predict with it are different properties."""

    def test_a_broken_real_model_prevents_startup(self, tmp_path) -> None:
        """A pickle can deserialise cleanly and still fail on the first request.

        Without the canary that failure lands on a real caller, after the container has
        passed its readiness probe and been given traffic. With it, the instance refuses to
        start and Cloud Run keeps the previous revision serving.
        """
        import pytest

        from src.api.main import _run_startup_canary
        from src.config import Settings

        class _BrokenPredictor:
            metadata = {"is_baseline": False, "predictor": "sklearn_pipeline"}

            def predict(self, frame):
                raise ValueError("feature names mismatch")

        with pytest.raises(RuntimeError, match="could not produce a prediction"):
            _run_startup_canary(_BrokenPredictor(), Settings())

    def test_a_broken_baseline_does_not_prevent_startup(self, tmp_path) -> None:
        """The asymmetry is deliberate.

        The baseline exists for environments with no artifact - CI, local development.
        Crashing the container when it fails would break the case the fallback was built to
        serve.
        """
        from src.api.main import _run_startup_canary
        from src.config import Settings

        class _BrokenBaseline:
            metadata = {"is_baseline": True, "predictor": "majority_class"}

            def predict(self, frame):
                raise ValueError("no")

        _run_startup_canary(_BrokenBaseline(), Settings())  # must not raise

    def test_a_working_model_passes(self) -> None:
        from src.api.main import _run_startup_canary
        from src.config import Settings

        _run_startup_canary(MajorityClassPredictor(majority_class=1), Settings())


class TestUnhandledExceptions:
    """The catch-all handler: opaque to the caller, complete in the log.

    The four domain handlers translate anticipated failures and their messages are safe
    because we wrote them. This one catches everything else - a numpy overflow, a shape
    mismatch - and those messages come from libraries. They can carry file paths, column
    names and array internals, and this service is unauthenticated.
    """

    @staticmethod
    def _exploding_client(client: TestClient) -> TestClient:
        """A client whose predictor raises something nobody anticipated."""
        from src.api.dependencies import get_predictor

        def _boom():
            raise RuntimeError("ndarray shape mismatch at /secret/path/model.pkl column g1_7")

        app = client.app
        app.dependency_overrides[get_predictor] = _boom
        # raise_server_exceptions=False makes TestClient return the 500 response rather than
        # re-raising, which is what a real HTTP client would see.
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_500_without_leaking_the_exception_message(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        exploding = self._exploding_client(client)
        response = exploding.post("/predict", json=valid_payload)

        assert response.status_code == 500
        body = response.json()

        assert body["error"] == "InternalServerError"
        # The library's message must not reach an unauthenticated caller.
        assert "secret/path" not in body["detail"]
        assert "ndarray" not in body["detail"]
        assert "g1_7" not in body["detail"]

        client.app.dependency_overrides.clear()

    def test_the_response_carries_a_request_id_for_diagnosis(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Opaque is not the same as useless.

        A caller who reports "request abc123 failed" can be answered exactly, because the
        same ID is bound to the log record carrying the traceback.
        """
        exploding = self._exploding_client(client)
        response = exploding.post(
            "/predict", json=valid_payload, headers={"X-Request-ID": "abc123"}
        )

        assert response.status_code == 500
        assert "abc123" in response.json()["detail"]
        assert response.headers["X-Request-ID"] == "abc123"

        client.app.dependency_overrides.clear()


class TestDecisionLayer:
    """The layer is opt-in, so these tests use `client_with_decisions`.

    Enabling it explicitly is the point. The layer turns probabilities into a recommended
    action using cost values nobody has supplied, so it ships **off**; a test that got the
    decision block from the default fixture would be asserting behaviour the deployed
    service does not have.
    """

    def test_no_decision_block_by_default(
        self, client_with_model: TestClient, valid_payload: dict
    ) -> None:
        """The shipped default: a prediction, and no recommended action.

        This is the important one. The reported 54.83% accuracy describes the classifier's
        argmax; with the decision layer on, the API recommends a *different* action on some
        campaigns. Defaulting to off keeps the number and the behaviour describing the same
        thing.
        """
        body = client_with_model.post("/predict", json=valid_payload).json()
        assert body["decision"] is None
        assert body["predicted_class"] in (0, 1, 2)

    def test_decision_block_is_attached_when_enabled(
        self, client_with_decisions: TestClient, valid_payload: dict
    ) -> None:
        body = client_with_decisions.post("/predict", json=valid_payload).json()
        assert body["decision"] is not None
        assert body["decision"]["action"] in (0, 1, 2)
        assert body["decision"]["rationale"]

    def test_decision_reports_expected_costs_for_every_action(
        self, client_with_decisions: TestClient, valid_payload: dict
    ) -> None:
        decision = client_with_decisions.post("/predict", json=valid_payload).json()["decision"]
        assert set(decision["expected_costs"]) == {
            "do_not_run",
            "target_group_1",
            "target_group_2",
        }

    def test_decision_records_the_argmax_for_comparison(
        self, client_with_decisions: TestClient, valid_payload: dict
    ) -> None:
        decision = client_with_decisions.post("/predict", json=valid_payload).json()["decision"]
        assert decision["argmax_class"] == 2
        assert isinstance(decision["differs_from_argmax"], bool)

    def test_model_info_reports_no_policy_by_default(self, client: TestClient) -> None:
        assert client.get("/model/info").json()["decision_policy"] is None

    def test_model_info_exposes_the_cost_matrix_when_enabled(
        self, client_with_decisions: TestClient
    ) -> None:
        policy = client_with_decisions.get("/model/info").json()["decision_policy"]
        assert policy is not None
        assert "cost_matrix" in policy
        # The costs are a ratio scale supplied by this project, not money. Labelling them
        # with a currency code would present invented weights as measured amounts.
        assert policy["cost_matrix"]["currency"] == "relative units"

    def test_decision_layer_can_be_disabled(self, settings, valid_payload: dict) -> None:
        from fastapi.testclient import TestClient as Client

        from src.api.dependencies import get_decision_policy
        from src.api.main import create_app

        app = create_app(settings)
        app.dependency_overrides[get_decision_policy] = lambda: None
        with Client(app) as disabled:
            body = disabled.post("/predict", json=valid_payload).json()
        app.dependency_overrides.clear()

        assert body["decision"] is None
        assert body["predicted_class"] in (0, 1, 2)

    def test_batch_predictions_carry_decisions(
        self, client_with_decisions: TestClient, valid_payload: dict
    ) -> None:
        body = client_with_decisions.post(
            "/predict/batch", json={"comparisons": [valid_payload] * 3}
        ).json()
        assert all(item["decision"] is not None for item in body["predictions"])

    def test_decision_reports_the_exploration_flag(
        self, client_with_decisions: TestClient, valid_payload: dict
    ) -> None:
        decision = client_with_decisions.post("/predict", json=valid_payload).json()["decision"]
        assert decision["exploration"] is False  # exploration is disabled in test settings


class TestConfigurationIsSingleSource:
    """The app's Settings must drive its dependencies, not a separate global cache."""

    def test_decision_policy_follows_the_app_settings(self, settings, valid_payload: dict) -> None:
        from fastapi.testclient import TestClient as Client

        from src.api.main import create_app

        custom = settings.model_copy(
            update={
                "campaign_spend": 500.0,
                "profit_if_correct": 2000.0,
                # Explicit: the layer is off by default, so a test of its configuration has
                # to switch it on.
                "enable_decision_layer": True,
            }
        )
        app = create_app(custom)
        with Client(app) as client:
            costs = client.get("/model/info").json()["decision_policy"]["cost_matrix"]["costs"]

        assert costs["true_group_1"]["target_group_1"] == -2000.0
        assert costs["true_group_1"]["target_group_2"] == 500.0

    def test_disabling_the_layer_via_settings_is_respected(
        self, settings, valid_payload: dict
    ) -> None:
        from fastapi.testclient import TestClient as Client

        from src.api.main import create_app

        app = create_app(settings.model_copy(update={"enable_decision_layer": False}))
        with Client(app) as client:
            body = client.post("/predict", json=valid_payload).json()
            info = client.get("/model/info").json()

        assert body["decision"] is None
        assert info["decision_policy"] is None

    def test_settings_are_exposed_on_app_state(self, settings) -> None:
        from src.api.main import create_app

        app = create_app(settings)
        assert app.state.settings is settings


class TestCorsConfiguration:
    def test_wildcard_disables_credentials(self, settings) -> None:
        from starlette.middleware.cors import CORSMiddleware

        from src.api.main import create_app

        app = create_app(settings.model_copy(update={"allowed_origins": "*"}))
        cors = [m for m in app.user_middleware if m.cls is CORSMiddleware][0]

        assert cors.kwargs["allow_origins"] == ["*"]
        assert cors.kwargs["allow_credentials"] is False

    def test_explicit_origins_enable_credentials(self, settings) -> None:
        from starlette.middleware.cors import CORSMiddleware

        from src.api.main import create_app

        app = create_app(
            settings.model_copy(
                update={"allowed_origins": "https://ui.example.com, https://admin.example.com"}
            )
        )
        cors = [m for m in app.user_middleware if m.cls is CORSMiddleware][0]

        assert cors.kwargs["allow_origins"] == [
            "https://ui.example.com",
            "https://admin.example.com",
        ]
        assert cors.kwargs["allow_credentials"] is True

    def test_methods_are_restricted(self, settings) -> None:
        from starlette.middleware.cors import CORSMiddleware

        from src.api.main import create_app

        app = create_app(settings)
        cors = [m for m in app.user_middleware if m.cls is CORSMiddleware][0]
        assert set(cors.kwargs["allow_methods"]) == {"GET", "POST"}


class TestBatchPredictEndpoint:
    def test_scores_every_comparison(self, client: TestClient, valid_payload: dict) -> None:
        response = client.post("/predict/batch", json={"comparisons": [valid_payload] * 5})
        assert response.status_code == 200

        body = response.json()
        assert body["count"] == 5
        assert len(body["predictions"]) == 5

    def test_empty_batch_returns_422(self, client: TestClient) -> None:
        assert client.post("/predict/batch", json={"comparisons": []}).status_code == 422

    def test_invalid_item_in_batch_returns_422(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        broken = {"group_1": {}, "group_2": {}, "comparison": {}}
        response = client.post("/predict/batch", json={"comparisons": [valid_payload, broken]})
        assert response.status_code == 422
