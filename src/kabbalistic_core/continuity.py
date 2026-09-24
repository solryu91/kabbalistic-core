"""Executable local slices for durable continuity and append-only correction.

The implementation supports one consented episodic record followed by at most
one explicitly proposed, consented, and atomically committed correction.  It
stops before forgetting, semantic projection, embeddings, archive import,
public inheritance, disagreement/non-mirroring, or branch merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import secrets
from pathlib import Path
from typing import Any

from .continuity_store import (
    ContinuityStore,
    ContinuityStoreError,
    DurableRecord,
    PayloadUnavailableError,
    utc_now,
)
from .models import canonical_json, stable_hash


class ContinuityProtocolError(RuntimeError):
    """A frozen consent, lifecycle, or causal contract was violated."""


def _normalize(value: str, *, field_name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text.")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank.")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} exceeds its {maximum}-character bound.")
    return normalized


def _deadline(hours: int = 24) -> str:
    return (
        datetime.now(UTC) + timedelta(hours=hours)
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _not_expired(value: str) -> bool:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) > datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ResponderBinding:
    provider_id: str = "deterministic-continuity-proof-responder:v0.1"
    model_kind: str = "deterministic_reference_responder"
    config_version: str = "continuity-proof-prompt:v0.1"
    call_budget: int = 1
    max_output_tokens: int = 256

    @property
    def config_hash(self) -> str:
        return stable_hash(
            {
                "provider_id": self.provider_id,
                "model_kind": self.model_kind,
                "config_version": self.config_version,
                "call_budget": self.call_budget,
                "max_output_tokens": self.max_output_tokens,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_kind": self.model_kind,
            "config_version": self.config_version,
            "config_hash": self.config_hash,
            "call_budget": self.call_budget,
            "max_output_tokens": self.max_output_tokens,
        }


class DeterministicContinuityResponder:
    """Reference responder that makes memory causality observable, not intelligent.

    It is intentionally simple.  A successful difference proves that a sealed
    memory read set crossed the session boundary and affected output.  It does
    not prove relationship intelligence, consciousness, or model preference.
    """

    binding = ResponderBinding()

    def respond(
        self,
        *,
        intention: str,
        ordered_history: tuple[dict[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        if len(ordered_history) > 1:
            raise ContinuityProtocolError(
                "The v0.1 vertical slice permits at most one selected episodic record."
            )
        if ordered_history:
            episode = ordered_history[0]
            correction_ref = episode.get("correction_ref")
            if correction_ref is None:
                response = (
                    "I am using one explicitly consented Session A record: "
                    f"“{episode['content']}” For the current intention—{intention}—"
                    "that record is a bounded callback, not a permission or an inferred preference."
                )
                cited_refs = [episode["record_ref"]]
            else:
                status = str(episode["status"])
                relation = str(episode["correction_relation"])
                if status == "superseded":
                    disposition = (
                        "I will not treat the original constraint as a current planning requirement."
                    )
                elif status == "disputed":
                    disposition = (
                        "I will preserve the dispute and will not present the original alone as uncontested current truth."
                    )
                else:
                    disposition = (
                        "I will apply the original only together with this qualification, never by itself."
                    )
                response = (
                    f"Historical record, now {status}: “{episode['content']}” "
                    f"Active {relation} correction: “{episode['correction_content']}” "
                    f"For the current intention—{intention}—{disposition} "
                    "This paired history grants no permission or external-action authority."
                )
                cited_refs = [episode["record_ref"], str(correction_ref)]
        else:
            response = (
                f"No consented relationship history is present in this read set. "
                f"For the current intention—{intention}—this response makes no historical callback."
            )
            cited_refs = []
        call = {
            "provider_id": self.binding.provider_id,
            "model_kind": self.binding.model_kind,
            "config_hash": self.binding.config_hash,
            "call_count": 1,
            "max_output_tokens": self.binding.max_output_tokens,
            "actual_output_word_count": len(response.split()),
            "selected_history_refs": cited_refs,
            "input_hash": stable_hash(
                {"intention": intention, "ordered_history": ordered_history}
            ),
            "output_hash": stable_hash(response),
        }
        return response, call


@dataclass(frozen=True, slots=True)
class ActiveEpisodeView:
    """Deterministically reduced status for the one-episode correction slice."""

    snapshot_ref: str
    original_ref: str | None
    original_status: str | None
    correction_ref: str | None
    correction_relation: str | None
    active_refs: tuple[str, ...]
    inactive_or_disputed_refs: tuple[str, ...]

    @property
    def correction_pair_refs(self) -> tuple[str, ...]:
        if self.original_ref is None or self.correction_ref is None:
            return ()
        return (self.original_ref, self.correction_ref)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_ref": self.snapshot_ref,
            "original_ref": self.original_ref,
            "original_status": self.original_status,
            "correction_ref": self.correction_ref,
            "correction_relation": self.correction_relation,
            "active_refs": list(self.active_refs),
            "inactive_or_disputed_refs": list(self.inactive_or_disputed_refs),
            "correction_pair_refs": list(self.correction_pair_refs),
            "original_historically_recoverable": self.original_ref is not None,
            "original_may_surface_alone_as_current": self.correction_ref is None,
        }


@dataclass(slots=True)
class ActiveContinuitySession:
    session_id: str
    session_open_ref: str
    runtime_instance_id: str
    intention: str
    start_head_ref: str
    start_snapshot_ref: str
    model_binding: dict[str, Any]
    proposal_refs: list[str] = field(default_factory=list)
    read_set_refs: list[str] = field(default_factory=list)
    stage_output_refs: list[str] = field(default_factory=list)
    delivery_refs: list[dict[str, str]] = field(default_factory=list)
    feedback_refs: list[str] = field(default_factory=list)
    stage_sequence: int = 0
    closed: bool = False


class ContinuityService:
    """Single-lineage service for the continuity and correction proof slices."""

    service_version = "parity-continuity-correction-slice:v0.2"
    reducer_binding = "continuity-active-view-reducer:v0.2-correction"
    policy_version = "continuity-private-episode-correction:v0.2"

    def __init__(
        self,
        store: ContinuityStore,
        *,
        responder: DeterministicContinuityResponder | None = None,
    ) -> None:
        self.store = store
        self.responder = responder or DeterministicContinuityResponder()
        self.runtime_instance_id = secrets.token_hex(16)
        self._sessions: dict[str, ActiveContinuitySession] = {}
        self.store.verify_checkpoint()
        self.store.verify_chain()

    @classmethod
    def create(
        cls,
        database_path: Path,
        checkpoint_path: Path,
        master_key: bytes,
        *,
        owner_principal: str,
    ) -> "ContinuityService":
        store = ContinuityStore.create(
            database_path,
            checkpoint_path,
            master_key,
            owner_principal=owner_principal,
        )
        try:
            cls._bootstrap(store, owner_principal)
            return cls(store)
        except Exception:
            store.close()
            for path in (Path(database_path).resolve(), Path(checkpoint_path).resolve()):
                if path.exists():
                    path.unlink()
            raise

    @classmethod
    def open(
        cls,
        database_path: Path,
        checkpoint_path: Path,
        master_key: bytes,
    ) -> "ContinuityService":
        store = ContinuityStore.open(database_path, checkpoint_path, master_key)
        try:
            return cls(store)
        except Exception:
            store.close()
            raise

    @staticmethod
    def _bootstrap(store: ContinuityStore, owner_principal: str) -> None:
        if not store.owner_matches(owner_principal):
            raise PermissionError("The bootstrap principal does not match the store owner commitment.")
        owner_ref = store.owner_principal_ref
        with store.write_batch(bootstrap=True) as batch:
            action_ref = batch.append(
                "AuthenticatedAuthorityActionRecord",
                "G",
                {
                    "action_kind": "lineage_bootstrap",
                    "principal_ref": owner_ref,
                    "authority_domain": "owned_private_lineage",
                    "target_selector": {"lineage_id": store.lineage_id},
                    "canonical_action_payload_hash": stable_hash(
                        {"lineage_id": store.lineage_id, "kind": "private_local"}
                    ),
                    "authenticator_binding": "caller_possession_of_external_master_key",
                    "authentication_result": "accepted_local_owner",
                    "confirmation_mode": "explicit_api_call",
                    "client_nonce": secrets.token_hex(16),
                    "issued_at_utc": utc_now(),
                    "expires_at_utc": _deadline(1),
                    "relationship_retrieval_eligible": False,
                },
            )
            policy_ref = batch.append(
                "LineagePolicyRevisionRecord",
                "G",
                {
                    "prior_policy_ref": None,
                    "new_policy_version": ContinuityService.policy_version,
                    "allowed_memory_types": [
                        "user_asserted_episode",
                        "user_asserted_correction",
                    ],
                    "allowed_derivation_types": [],
                    "default_retention": "until_explicitly_forgotten_or_lineage_destroyed",
                    "sensitivity_rules": ["private_local_only", "encrypted_payload_required"],
                    "retrieval_purposes": ["explicit_session_callback"],
                    "export_rules": ["explicit_owner_export_only"],
                    "backup_rules": ["content_backup_must_exclude_authority_checkpoint"],
                    "forgetting_modes": [
                        "ERASE_PAYLOAD_AND_DERIVATIVES",
                        "DESTROY_LINEAGE",
                    ],
                    "effective_at": utc_now(),
                    "reason": "DR-10 vertical slice bootstrap",
                    "relationship_retrieval_eligible": False,
                },
            )
            genesis_ref = batch.append(
                "LineageGenesisRecord",
                "G",
                {
                    "lineage_kind": "private_local",
                    "owner_principal_ref": owner_ref,
                    "storage_domain_id": store.store_id,
                    "encryption_domain_id": stable_hash(
                        {"store_id": store.store_id, "domain": "external-master-key"}
                    ),
                    "corpus_namespace": None,
                    "initial_policy_ref": policy_ref,
                    "declared_capabilities": [
                        "single_user_asserted_episode",
                        "explicit_proposal_consent_commit",
                        "single_append_only_correction",
                        "deterministic_active_state_reduction",
                        "independent_session_read",
                        "matched_no_history_control",
                    ],
                    "declared_non_capabilities": [
                        "forgetting",
                        "semantic_projection",
                        "archive_import",
                        "public_inheritance",
                        "branch_merge",
                    ],
                    "relationship_retrieval_eligible": False,
                },
            )
            snapshot_ref = batch.append(
                "MemorySnapshotRecord",
                "G",
                {
                    "lineage_genesis_ref": genesis_ref,
                    "branch_id": "main",
                    "parent_snapshot_refs": [],
                    "base_head_ref": None,
                    "policy_ref": policy_ref,
                    "public_inheritance_manifest_refs": [],
                    "active_public_corpus_refs": [],
                    "active_episode_refs": [],
                    "active_semantic_refs": [],
                    "active_relational_refs": [],
                    "inactive_or_disputed_refs": [],
                    "revoked_refs": [],
                    "forget_pending_refs": [],
                    "forgotten_tombstone_refs": [],
                    "projection_manifest_refs": [],
                    "reducer_binding": ContinuityService.reducer_binding,
                    "source_record_root": stable_hash([]),
                    "created_by_transaction_id": batch.transaction_id,
                    "relationship_retrieval_eligible": False,
                },
            )
            head_ref = batch.append(
                "LineageHeadRecord",
                "G",
                {
                    "prior_head_refs": [],
                    "branch_id": "main",
                    "branch_sequence": 0,
                    "commit_transaction_id": batch.transaction_id,
                    "snapshot_ref": snapshot_ref,
                    "policy_ref": policy_ref,
                    "head_status": "active",
                    "relationship_retrieval_eligible": False,
                },
            )
            batch.append(
                "GovernanceTransitionRecord",
                "G",
                {
                    "transition_kind": "LINEAGE_BOOTSTRAP",
                    "triggering_action_ref": action_ref,
                    "governance_parent_transition_ref": None,
                    "governance_sequence": batch.governance_sequence_after,
                    "authority_basis": {
                        "basis_kind": "authenticated_owner_action",
                        "authenticated_actor_ref": owner_ref,
                        "constitutional_rule_id": None,
                        "authorizing_transition_ref": None,
                    },
                    "policy_ref": policy_ref,
                    "reducer_binding": ContinuityService.reducer_binding,
                    "before_snapshot_ref": None,
                    "before_head_ref": None,
                    "after_snapshot_ref": snapshot_ref,
                    "after_head_ref": head_ref,
                    "affected_selectors": [{"lineage_id": store.lineage_id}],
                    "action_record_refs": [action_ref, genesis_ref, policy_ref, snapshot_ref, head_ref],
                    "validation_results": [
                        "owner_authenticated",
                        "empty_private_lineage",
                        "external_master_key_not_stored",
                        "public_inheritance_absent",
                    ],
                    "idempotency_key": stable_hash(
                        {"kind": "bootstrap", "store_id": store.store_id}
                    ),
                    "transaction_id": batch.transaction_id,
                    "erasure_epoch_before": 0,
                    "erasure_epoch_after": 0,
                    "authority_checkpoint_domain_id": store.checkpoint_domain_id,
                    "authority_checkpoint_epoch_before": batch.checkpoint_epoch_before,
                    "authority_checkpoint_epoch_after": batch.checkpoint_epoch_after,
                    "status": "applied",
                    "failure_record_ref": None,
                    "completed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
            )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ContinuityService":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _require_owner(self, principal: str) -> str:
        normalized = _normalize(principal, field_name="principal", maximum=200)
        if not self.store.owner_matches(normalized):
            raise PermissionError("The principal does not own this private local lineage.")
        return self.store.owner_principal_ref

    def _session(self, session_id: str) -> ActiveContinuitySession:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            raise ContinuityProtocolError("The continuity session is not active in this runtime instance.")
        return session

    def _current_state(self) -> tuple[DurableRecord, DurableRecord, DurableRecord]:
        checkpoint = self.store.verify_checkpoint()
        head = self.store.read_record(checkpoint.active_lineage_head_hash)
        snapshot = self.store.read_record(checkpoint.active_snapshot_root)
        policy = self.store.current_policy()
        return head, snapshot, policy

    def reduce_active_state(self, snapshot_ref: str | None = None) -> ActiveEpisodeView:
        """Recompute the active episodic disposition from immutable records.

        Snapshot status fields are validated against the correction relation;
        they are not accepted as actor-authored semantic commands.
        """

        snapshot = (
            self.store.current_snapshot()
            if snapshot_ref is None
            else self.store.read_record(snapshot_ref)
        )
        if snapshot.record_type != "MemorySnapshotRecord":
            raise ContinuityProtocolError("Active-state reduction requires a memory snapshot.")
        active = tuple(str(ref) for ref in snapshot.payload.get("active_episode_refs", []))
        inactive = tuple(
            str(ref) for ref in snapshot.payload.get("inactive_or_disputed_refs", [])
        )
        if len(set((*active, *inactive))) != len((*active, *inactive)):
            raise ContinuityProtocolError("A memory reference appears in multiple snapshot states.")
        records = [self.store.read_record(ref) for ref in (*active, *inactive)]
        originals = [record for record in records if record.record_type == "EpisodicEventRecord"]
        corrections = [record for record in records if record.record_type == "CorrectionRecord"]
        unknown = [
            record.record_type
            for record in records
            if record.record_type not in {"EpisodicEventRecord", "CorrectionRecord"}
        ]
        if unknown:
            raise ContinuityProtocolError(
                f"The correction slice cannot reduce memory record types: {unknown}."
            )
        if not originals and not corrections:
            return ActiveEpisodeView(
                snapshot_ref=snapshot.record_hash,
                original_ref=None,
                original_status=None,
                correction_ref=None,
                correction_relation=None,
                active_refs=active,
                inactive_or_disputed_refs=inactive,
            )
        if len(originals) != 1 or len(corrections) > 1:
            raise ContinuityProtocolError(
                "The correction slice requires exactly one original episode and at most one correction."
            )
        original = originals[0]
        if not corrections:
            if active != (original.record_hash,) or inactive:
                raise ContinuityProtocolError(
                    "An uncorrected episode must be the sole active episodic record."
                )
            return ActiveEpisodeView(
                snapshot_ref=snapshot.record_hash,
                original_ref=original.record_hash,
                original_status="active",
                correction_ref=None,
                correction_relation=None,
                active_refs=active,
                inactive_or_disputed_refs=inactive,
            )

        correction = corrections[0]
        targets = correction.payload.get("target_refs")
        if targets != [original.record_hash]:
            raise ContinuityProtocolError(
                "The correction does not bind the exact original episodic record."
            )
        relation = str(correction.payload.get("requested_relation"))
        status_by_relation = {
            "qualifies": "qualified",
            "disputes": "disputed",
            "supersedes": "superseded",
        }
        status = status_by_relation.get(relation)
        if status is None:
            raise ContinuityProtocolError(
                "The bounded correction reducer received an unsupported relation."
            )
        expected_active = (
            (original.record_hash, correction.record_hash)
            if relation == "qualifies"
            else (correction.record_hash,)
        )
        expected_inactive = () if relation == "qualifies" else (original.record_hash,)
        if active != expected_active or inactive != expected_inactive:
            raise ContinuityProtocolError(
                "The immutable snapshot status does not match deterministic correction reduction."
            )
        return ActiveEpisodeView(
            snapshot_ref=snapshot.record_hash,
            original_ref=original.record_hash,
            original_status=status,
            correction_ref=correction.record_hash,
            correction_relation=relation,
            active_refs=active,
            inactive_or_disputed_refs=inactive,
        )

    def _append_transition(
        self,
        batch,  # WriteBatch without exposing the internal type publicly
        *,
        transition_kind: str,
        triggering_action_ref: str,
        before_head_ref: str | None,
        before_snapshot_ref: str | None,
        after_head_ref: str,
        after_snapshot_ref: str,
        policy_ref: str,
        authority_basis: dict[str, Any],
        action_record_refs: list[str],
        validation_results: list[str],
        affected_selectors: list[dict[str, Any]],
        status: str,
    ) -> str:
        prior = self.store._latest_row("GovernanceTransitionRecord")
        return batch.append(
            "GovernanceTransitionRecord",
            "G",
            {
                "transition_kind": transition_kind,
                "triggering_action_ref": triggering_action_ref,
                "governance_parent_transition_ref": (
                    None if prior is None else str(prior["record_hash"])
                ),
                "governance_sequence": batch.governance_sequence_after,
                "authority_basis": authority_basis,
                "policy_ref": policy_ref,
                "reducer_binding": self.reducer_binding,
                "before_snapshot_ref": before_snapshot_ref,
                "before_head_ref": before_head_ref,
                "after_snapshot_ref": after_snapshot_ref,
                "after_head_ref": after_head_ref,
                "affected_selectors": affected_selectors,
                "action_record_refs": action_record_refs,
                "validation_results": validation_results,
                "idempotency_key": stable_hash(
                    {
                        "transition_kind": transition_kind,
                        "transaction_id": batch.transaction_id,
                        "actions": action_record_refs,
                    }
                ),
                "transaction_id": batch.transaction_id,
                "erasure_epoch_before": batch.erasure_epoch_before,
                "erasure_epoch_after": batch.erasure_epoch_before,
                "authority_checkpoint_domain_id": self.store.checkpoint_domain_id,
                "authority_checkpoint_epoch_before": batch.checkpoint_epoch_before,
                "authority_checkpoint_epoch_after": batch.checkpoint_epoch_after,
                "status": status,
                "failure_record_ref": None,
                "completed_at_utc": utc_now(),
                "relationship_retrieval_eligible": False,
            },
        )

    def open_session(self, *, principal: str, intention: str) -> dict[str, Any]:
        owner = self._require_owner(principal)
        normalized_intention = _normalize(intention, field_name="intention", maximum=4000)
        head, snapshot, policy = self._current_state()
        session_id = secrets.token_hex(16)
        with self.store.write_batch() as batch:
            action_ref = batch.append(
                "AuthenticatedAuthorityActionRecord",
                "G",
                {
                    "action_kind": "session_open",
                    "principal_ref": owner,
                    "authority_domain": "owned_private_lineage",
                    "target_selector": {"lineage_id": self.store.lineage_id},
                    "canonical_action_payload_hash": stable_hash(
                        {"session_id": session_id, "intention": normalized_intention}
                    ),
                    "authenticator_binding": "caller_possession_of_external_master_key",
                    "authentication_result": "accepted_local_owner",
                    "confirmation_mode": "explicit_api_call",
                    "client_nonce": secrets.token_hex(16),
                    "issued_at_utc": utc_now(),
                    "expires_at_utc": _deadline(1),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            session_open_ref = batch.append(
                "SessionOpenRecord",
                "G",
                {
                    "session_id": session_id,
                    "lineage_id": self.store.lineage_id,
                    "branch_id": "main",
                    "parent_session_refs": [],
                    "parent_lineage_head_refs": [head.record_hash],
                    "start_snapshot_ref": snapshot.record_hash,
                    "policy_ref": policy.record_hash,
                    "model_binding": self.responder.binding.as_dict(),
                    "graph_binding": "continuity-slice-graph:none",
                    "topology_binding": "single-history-context-pass:v0.1",
                    "prompt_contract_hash": self.responder.binding.config_hash,
                    "retrieval_budget": 1,
                    "trace_capture_authority_ref": None,
                    "trace_retention_policy_ref": None,
                    "trace_audience": "owner_live_and_explicit_export_only",
                    "trace_payload_capture_allowed": False,
                    "trace_effect_and_risk_disclosure_ref": None,
                    "proposal_staging_authority_ref": action_ref,
                    "proposal_staging_policy_ref": policy.record_hash,
                    "runtime_instance_id": self.runtime_instance_id,
                    "intention_commitment": self.store.keyed_commitment(normalized_intention),
                    "opened_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="SESSION_OPEN",
                triggering_action_ref=action_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "authenticated_owner_action",
                    "authenticated_actor_ref": owner,
                    "constitutional_rule_id": None,
                    "authorizing_transition_ref": None,
                },
                action_record_refs=[action_ref, session_open_ref],
                validation_results=[
                    "owner_authenticated",
                    "snapshot_bound",
                    "model_binding_bound",
                    "trace_payload_capture_disabled",
                ],
                affected_selectors=[{"session_id": session_id}],
                status="recorded_no_active_change",
            )
        active = ActiveContinuitySession(
            session_id=session_id,
            session_open_ref=session_open_ref,
            runtime_instance_id=self.runtime_instance_id,
            intention=normalized_intention,
            start_head_ref=head.record_hash,
            start_snapshot_ref=snapshot.record_hash,
            model_binding=self.responder.binding.as_dict(),
        )
        self._sessions[session_id] = active
        return {
            "session_id": session_id,
            "session_open_ref": session_open_ref,
            "runtime_instance_id": self.runtime_instance_id,
            "lineage_id": self.store.lineage_id,
            "start_head_ref": head.record_hash,
            "start_snapshot_ref": snapshot.record_hash,
            "model_binding": active.model_binding,
            "durable_relationship_write_count": 0,
        }

    def propose_episode(
        self,
        session_id: str,
        *,
        principal: str,
        content: str,
        purpose: str = "future bounded callback in this private lineage",
    ) -> dict[str, Any]:
        owner = self._require_owner(principal)
        session = self._session(session_id)
        proposed_content = _normalize(content, field_name="proposed memory", maximum=2000)
        normalized_purpose = _normalize(purpose, field_name="purpose", maximum=300)
        head, snapshot, policy = self._current_state()
        if head.record_hash != session.start_head_ref or snapshot.record_hash != session.start_snapshot_ref:
            raise ContinuityProtocolError(
                "The active head changed after Session A opened; start a new session before proposing."
            )
        terms = {
            "purpose": normalized_purpose,
            "scope": f"private_lineage:{self.store.lineage_id}",
            "audience": [owner],
            "retention": "until_explicitly_forgotten_or_lineage_destroyed",
            "allowed_derivations": [],
            "retrieval_effect_preview": (
                "This exact episode may be selected in a later independent session and can change its response."
            ),
            "risk_disclosure": (
                "The record is sensitive local relationship data; a later callback may reveal it to this owner."
            ),
            "forgetting_effect_preview": (
                "Forgetting is not implemented in this slice and must be reviewed before use."
            ),
        }
        proposal_payload = {
            "proposal_kind": "retain_episode",
            "proposed_record_payloads": [
                {
                    "episode_kind": "user_asserted_episode",
                    "speaker_or_origin": owner,
                    "content": proposed_content,
                    "content_mode": "direct_user_statement",
                    "epistemic_class": "owner_attributed_testimony_not_objective_fact",
                }
            ],
            "source_refs": [session.session_open_ref],
            "proposer_ref": owner,
            "staging_authority_ref": session.session_open_ref,
            "relationship_retrieval_eligible": False,
            **terms,
            "sensitivity_assessment": "private_local_relationship_data",
            "proposal_expires_at": _deadline(24),
            "staging_payload_expires_at": _deadline(24),
            "precondition_snapshot_ref": snapshot.record_hash,
            "precondition_head_ref": head.record_hash,
            "proposal_content_root": stable_hash(
                {"content": proposed_content, "origin": owner}
            ),
            "canonical_terms_hash": stable_hash(terms),
        }
        with self.store.write_batch() as batch:
            proposal_ref = batch.append(
                "MemoryProposalRecord", "G", proposal_payload, session_id=session_id
            )
            self._append_transition(
                batch,
                transition_kind="PROPOSAL_STAGE",
                triggering_action_ref=session.session_open_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-3.2-VISIBLE-PROPOSAL",
                    "authorizing_transition_ref": session.session_open_ref,
                },
                action_record_refs=[proposal_ref],
                validation_results=[
                    "proposal_staging_authorized",
                    "relationship_retrieval_ineligible",
                    "no_derivations_requested",
                    "no_active_memory_change",
                ],
                affected_selectors=[{"proposal_ref": proposal_ref}],
                status="recorded_no_active_change",
            )
        session.proposal_refs.append(proposal_ref)
        return {
            "proposal_ref": proposal_ref,
            "proposal_kind": "retain_episode",
            "proposed_content": proposed_content,
            "proposer_ref": owner,
            **terms,
            "proposal_content_root": proposal_payload["proposal_content_root"],
            "canonical_terms_hash": proposal_payload["canonical_terms_hash"],
            "proposal_expires_at": proposal_payload["proposal_expires_at"],
            "visible_consent_required": True,
            "active_memory_changed": False,
        }

    def propose_correction(
        self,
        session_id: str,
        *,
        principal: str,
        target_ref: str,
        requested_relation: str,
        corrected_or_qualifying_content: str,
        purpose: str = "correct the active interpretation while preserving history",
    ) -> dict[str, Any]:
        owner = self._require_owner(principal)
        session = self._session(session_id)
        if requested_relation not in {"qualifies", "disputes", "supersedes"}:
            raise ValueError(
                "The correction relation must be 'qualifies', 'disputes', or 'supersedes'."
            )
        corrected_content = _normalize(
            corrected_or_qualifying_content,
            field_name="correction content",
            maximum=2000,
        )
        normalized_purpose = _normalize(purpose, field_name="purpose", maximum=300)
        head, snapshot, policy = self._current_state()
        if head.record_hash != session.start_head_ref or snapshot.record_hash != session.start_snapshot_ref:
            raise ContinuityProtocolError(
                "The active head changed after the correction session opened; start a new session."
            )
        if "user_asserted_correction" not in policy.payload.get("allowed_memory_types", []):
            raise ContinuityProtocolError(
                "This lineage policy predates the authorized correction slice."
            )
        active_view = self.reduce_active_state(snapshot.record_hash)
        if active_view.original_ref != target_ref or active_view.original_status != "active":
            raise ContinuityProtocolError(
                "The target must be the sole currently active, uncorrected episodic record."
            )
        target = self.store.read_record(target_ref)
        if target.record_type != "EpisodicEventRecord":
            raise ContinuityProtocolError("A correction must target an episodic event.")

        computed_status = {
            "qualifies": "qualified",
            "disputes": "disputed",
            "supersedes": "superseded",
        }[requested_relation]
        terms = {
            "purpose": normalized_purpose,
            "scope": {
                "schema_version": "correction-target-selector:v0.1",
                "lineage_id": self.store.lineage_id,
                "target_refs": [target_ref],
                "requested_relation": requested_relation,
            },
            "audience": [owner],
            "retention": "until_explicitly_forgotten_or_lineage_destroyed",
            "allowed_derivations": [],
            "retrieval_effect_preview": (
                f"The original remains historical and attributable but is rendered as {computed_status}; "
                "it cannot appear alone as uncontested current truth."
            ),
            "risk_disclosure": (
                "The correction and original may be revealed together to this owner in later callbacks."
            ),
            "forgetting_effect_preview": (
                "Forgetting is not implemented in this slice and remains separately gated."
            ),
        }
        proposed_correction = {
            "target_refs": [target_ref],
            "requested_relation": requested_relation,
            "corrected_or_qualifying_content": corrected_content,
            "asserting_actor_ref": owner,
            "evidence_refs": [],
            "epistemic_status": "owner_attributed_correction_not_objective_fact",
            "effective_from": "at_atomic_commit",
        }
        proposal_payload = {
            "proposal_kind": "correct_or_supersede",
            "proposed_record_payloads": [proposed_correction],
            "source_refs": [session.session_open_ref, target_ref],
            "proposer_ref": owner,
            "staging_authority_ref": session.session_open_ref,
            "relationship_retrieval_eligible": False,
            **terms,
            "sensitivity_assessment": "private_local_relationship_correction",
            "proposal_expires_at": _deadline(24),
            "staging_payload_expires_at": _deadline(24),
            "precondition_snapshot_ref": snapshot.record_hash,
            "precondition_head_ref": head.record_hash,
            "proposal_content_root": stable_hash(
                {
                    "target_ref": target_ref,
                    "requested_relation": requested_relation,
                    "content": corrected_content,
                    "actor": owner,
                }
            ),
            "canonical_terms_hash": stable_hash(terms),
            "computed_effect_preview": {
                "target_ref": target_ref,
                "target_status_after_commit": computed_status,
                "original_record_mutated": False,
            },
        }
        with self.store.write_batch() as batch:
            proposal_ref = batch.append(
                "MemoryProposalRecord", "G", proposal_payload, session_id=session_id
            )
            self._append_transition(
                batch,
                transition_kind="PROPOSAL_STAGE",
                triggering_action_ref=session.session_open_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-3.2-VISIBLE-PROPOSAL",
                    "authorizing_transition_ref": session.session_open_ref,
                },
                action_record_refs=[proposal_ref],
                validation_results=[
                    "correction_staging_authorized",
                    "target_is_active_original_episode",
                    "relationship_retrieval_ineligible_until_commit",
                    "original_record_unchanged",
                    "no_active_memory_change",
                ],
                affected_selectors=[
                    {"proposal_ref": proposal_ref},
                    {"target_ref": target_ref},
                ],
                status="recorded_no_active_change",
            )
        session.proposal_refs.append(proposal_ref)
        return {
            "proposal_ref": proposal_ref,
            "proposal_kind": "correct_or_supersede",
            "target_ref": target_ref,
            "requested_relation": requested_relation,
            "corrected_or_qualifying_content": corrected_content,
            "computed_target_status": computed_status,
            "proposer_ref": owner,
            **terms,
            "proposal_content_root": proposal_payload["proposal_content_root"],
            "canonical_terms_hash": proposal_payload["canonical_terms_hash"],
            "proposal_expires_at": proposal_payload["proposal_expires_at"],
            "visible_consent_required": True,
            "active_memory_changed": False,
            "original_record_mutated": False,
        }

    def _decisions_for(self, proposal_ref: str) -> list[DurableRecord]:
        decisions: list[DurableRecord] = []
        for ref in self.store.list_record_refs(record_type="ConsentDecisionRecord"):
            record = self.store.read_record(ref)
            if record.payload.get("proposal_ref") == proposal_ref:
                decisions.append(record)
        return decisions

    def decide_proposal(
        self,
        *,
        proposal_ref: str,
        principal: str,
        decision: str,
        canonical_terms_hash: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        owner = self._require_owner(principal)
        if decision not in {"grant", "reject"}:
            raise ValueError("Consent decision must be 'grant' or 'reject'.")
        proposal = self.store.read_record(proposal_ref)
        if proposal.record_type != "MemoryProposalRecord":
            raise ContinuityProtocolError("Consent must bind an exact MemoryProposalRecord.")
        expected_terms = str(proposal.payload["canonical_terms_hash"])
        if not secrets.compare_digest(canonical_terms_hash, expected_terms):
            raise ContinuityProtocolError("Consent terms do not match the staged proposal exactly.")
        if not _not_expired(str(proposal.payload["proposal_expires_at"])):
            raise ContinuityProtocolError("The proposal expired before consent.")
        if self._decisions_for(proposal_ref):
            raise ContinuityProtocolError("This proposal already has a consent decision.")
        head, snapshot, policy = self._current_state()
        normalized_reason = None if reason is None else _normalize(
            reason, field_name="decision reason", maximum=500
        )
        proposal_kind = str(proposal.payload.get("proposal_kind"))
        authority_domain = (
            "episodic_correction"
            if proposal_kind == "correct_or_supersede"
            else "episodic_retention"
        )
        with self.store.write_batch() as batch:
            action_ref = batch.append(
                "AuthenticatedAuthorityActionRecord",
                "G",
                {
                    "action_kind": f"consent_{decision}",
                    "principal_ref": owner,
                    "authority_domain": authority_domain,
                    "target_selector": {"proposal_ref": proposal_ref},
                    "canonical_action_payload_hash": stable_hash(
                        {
                            "proposal_ref": proposal_ref,
                            "decision": decision,
                            "canonical_terms_hash": canonical_terms_hash,
                        }
                    ),
                    "authenticator_binding": "caller_possession_of_external_master_key",
                    "authentication_result": "accepted_local_owner",
                    "confirmation_mode": "explicit_exact_proposal_decision",
                    "client_nonce": secrets.token_hex(16),
                    "issued_at_utc": utc_now(),
                    "expires_at_utc": _deadline(1),
                    "relationship_retrieval_eligible": False,
                },
                session_id=proposal.session_id,
            )
            decision_ref = batch.append(
                "ConsentDecisionRecord",
                "G",
                {
                    "decision": decision,
                    "proposal_ref": proposal_ref,
                    "grantor_ref": owner,
                    "sequence": 1,
                    "decided_at_utc": utc_now(),
                    "commit_authorization_expires_at": (
                        _deadline(1) if decision == "grant" else None
                    ),
                    "policy_ref": policy.record_hash,
                    "reason": normalized_reason,
                    "canonical_terms_hash": canonical_terms_hash,
                    "relationship_retrieval_eligible": False,
                },
                session_id=proposal.session_id,
            )
            action_refs = [action_ref, decision_ref]
            validation = [
                "grantor_is_lineage_owner",
                "proposal_ref_exact",
                "canonical_terms_hash_exact",
                "proposal_unexpired",
                "no_active_memory_change",
            ]
            if decision == "reject":
                disposition_ref = batch.append(
                    "ProposalDispositionRecord",
                    "G",
                    {
                        "proposal_ref": proposal_ref,
                        "disposition": "rejected_and_staging_payload_erased",
                        "cause_ref": decision_ref,
                        "relationship_retrieval_eligible": False,
                        "disposed_at_utc": utc_now(),
                    },
                    session_id=proposal.session_id,
                )
                destroyed_commitment = batch.erase_payload_key(proposal_ref)
                action_refs.append(disposition_ref)
                validation.extend(
                    ["rejected_proposal_not_active", f"staging_key_destroyed:{destroyed_commitment}"]
                )
            self._append_transition(
                batch,
                transition_kind="CONSENT_DECIDE",
                triggering_action_ref=action_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "authenticated_owner_action",
                    "authenticated_actor_ref": owner,
                    "constitutional_rule_id": None,
                    "authorizing_transition_ref": None,
                },
                action_record_refs=action_refs,
                validation_results=validation,
                affected_selectors=[{"proposal_ref": proposal_ref}],
                status="recorded_no_active_change",
            )
        return {
            "decision_ref": decision_ref,
            "decision": decision,
            "proposal_ref": proposal_ref,
            "grantor_ref": owner,
            "canonical_terms_hash": canonical_terms_hash,
            "active_memory_changed": False,
            "commit_authorized": decision == "grant",
        }

    def commit_granted_proposal(
        self,
        *,
        proposal_ref: str,
        consent_decision_ref: str,
    ) -> dict[str, Any]:
        completed = self._completed_commit_result(
            proposal_ref=proposal_ref,
            consent_decision_ref=consent_decision_ref,
        )
        if completed is not None:
            return completed
        proposal = self.store.read_record(proposal_ref)
        consent = self.store.read_record(consent_decision_ref)
        if proposal.record_type != "MemoryProposalRecord":
            raise ContinuityProtocolError("Commit requires an exact memory proposal.")
        if consent.record_type != "ConsentDecisionRecord" or consent.payload.get("decision") != "grant":
            raise ContinuityProtocolError("Commit requires an exact granted consent decision.")
        if consent.payload.get("proposal_ref") != proposal_ref:
            raise ContinuityProtocolError("The consent decision binds a different proposal.")
        if consent.payload.get("canonical_terms_hash") != proposal.payload.get("canonical_terms_hash"):
            raise ContinuityProtocolError("Consent terms changed before commit.")
        deadline = consent.payload.get("commit_authorization_expires_at")
        if not isinstance(deadline, str) or not _not_expired(deadline):
            raise ContinuityProtocolError("The consent commit authorization expired.")
        head, snapshot, policy = self._current_state()
        if head.record_hash != proposal.payload.get("precondition_head_ref"):
            raise ContinuityProtocolError("The active head changed after proposal staging.")
        if snapshot.record_hash != proposal.payload.get("precondition_snapshot_ref"):
            raise ContinuityProtocolError("The active snapshot changed after proposal staging.")
        proposal_kind = str(proposal.payload.get("proposal_kind"))
        if proposal_kind == "correct_or_supersede":
            return self._commit_correction_proposal(
                proposal_ref=proposal_ref,
                consent_decision_ref=consent_decision_ref,
                proposal=proposal,
                consent=consent,
                head=head,
                snapshot=snapshot,
                policy=policy,
                deadline=deadline,
            )
        if proposal_kind != "retain_episode":
            raise ContinuityProtocolError("The proposal kind is unsupported by this bounded slice.")
        active_episodes = list(snapshot.payload.get("active_episode_refs", []))
        if active_episodes:
            raise ContinuityProtocolError(
                "The v0.1 vertical slice stops after one committed episodic record."
            )
        proposed_payloads = proposal.payload.get("proposed_record_payloads")
        if not isinstance(proposed_payloads, list) or len(proposed_payloads) != 1:
            raise ContinuityProtocolError("The vertical slice requires exactly one proposed episode.")
        proposed_episode = proposed_payloads[0]
        if not isinstance(proposed_episode, dict):
            raise ContinuityProtocolError("The proposed episode is malformed.")
        session_ref = str(proposal.payload["source_refs"][0])
        with self.store.write_batch() as batch:
            episode_ref = batch.append(
                "EpisodicEventRecord",
                "E",
                {
                    "episode_kind": "user_asserted_episode",
                    "speaker_or_origin": proposed_episode["speaker_or_origin"],
                    "content": proposed_episode["content"],
                    "content_mode": "direct_user_statement",
                    "epistemic_class": "owner_attributed_testimony_not_objective_fact",
                    "source_time": utc_now(),
                    "source_time_certainty": "system_commit_time_only",
                    "source_sequence": None,
                    "provenance_refs": [proposal_ref, consent_decision_ref, session_ref],
                    "current_session_context_ref": session_ref,
                    "correction_state": "uncorrected_at_commit",
                    "publication_class": "private_local_only",
                    "third_party_classification": "none_declared",
                    "relationship_retrieval_eligible": True,
                },
                session_id=proposal.session_id,
            )
            snapshot_ref = batch.append(
                "MemorySnapshotRecord",
                "G",
                {
                    "lineage_genesis_ref": snapshot.payload["lineage_genesis_ref"],
                    "branch_id": "main",
                    "parent_snapshot_refs": [snapshot.record_hash],
                    "base_head_ref": head.record_hash,
                    "policy_ref": policy.record_hash,
                    "public_inheritance_manifest_refs": [],
                    "active_public_corpus_refs": [],
                    "active_episode_refs": [episode_ref],
                    "active_semantic_refs": [],
                    "active_relational_refs": [],
                    "inactive_or_disputed_refs": [],
                    "revoked_refs": [],
                    "forget_pending_refs": [],
                    "forgotten_tombstone_refs": [],
                    "projection_manifest_refs": [],
                    "reducer_binding": self.reducer_binding,
                    "source_record_root": stable_hash([episode_ref]),
                    "created_by_transaction_id": batch.transaction_id,
                    "relationship_retrieval_eligible": False,
                },
            )
            head_ref = batch.append(
                "LineageHeadRecord",
                "G",
                {
                    "prior_head_refs": [head.record_hash],
                    "branch_id": "main",
                    "branch_sequence": int(head.payload["branch_sequence"]) + 1,
                    "commit_transaction_id": batch.transaction_id,
                    "snapshot_ref": snapshot_ref,
                    "policy_ref": policy.record_hash,
                    "head_status": "active",
                    "relationship_retrieval_eligible": False,
                },
            )
            commit_ref = batch.append(
                "MemoryCommitRecord",
                "G",
                {
                    "proposal_ref": proposal_ref,
                    "consent_decision_ref": consent_decision_ref,
                    "source_session_refs": [session_ref],
                    "precondition_snapshot_ref": snapshot.record_hash,
                    "before_head_refs": [head.record_hash],
                    "committed_record_refs": [episode_ref],
                    "new_snapshot_ref": snapshot_ref,
                    "after_head_ref": head_ref,
                    "validation_results": [
                        "active_head_unchanged",
                        "exact_consent_valid",
                        "source_authorized",
                        "payload_aes256_gcm_protected",
                        "private_lineage_only",
                        "current_cycle_input_unchanged",
                    ],
                    "transaction_id": batch.transaction_id,
                    "idempotency_key": stable_hash(
                        {"proposal_ref": proposal_ref, "consent_ref": consent_decision_ref}
                    ),
                    "committed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=proposal.session_id,
            )
            receipt_ref = batch.append(
                "CommittedConsentReceiptRecord",
                "G",
                {
                    "memory_commit_ref": commit_ref,
                    "proposal_ref": proposal_ref,
                    "consent_decision_ref": consent_decision_ref,
                    "canonical_purpose": proposal.payload["purpose"],
                    "canonical_scope": proposal.payload["scope"],
                    "canonical_audience": proposal.payload["audience"],
                    "canonical_retention": proposal.payload["retention"],
                    "canonical_allowed_derivations": proposal.payload["allowed_derivations"],
                    "proposal_content_root": proposal.payload["proposal_content_root"],
                    "committed_record_root": stable_hash([episode_ref]),
                    "effect_preview": proposal.payload["retrieval_effect_preview"],
                    "risk_preview": proposal.payload["risk_disclosure"],
                    "forgetting_effect_preview": proposal.payload["forgetting_effect_preview"],
                    "effective_commit_deadline": deadline,
                    "committed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                    "dependency_refs": [episode_ref],
                },
                session_id=proposal.session_id,
            )
            disposition_ref = batch.append(
                "ProposalDispositionRecord",
                "G",
                {
                    "proposal_ref": proposal_ref,
                    "disposition": "committed_and_staging_payload_erased",
                    "cause_ref": commit_ref,
                    "committed_consent_receipt_ref": receipt_ref,
                    "relationship_retrieval_eligible": False,
                    "disposed_at_utc": utc_now(),
                },
                session_id=proposal.session_id,
            )
            destroyed_key_commitment = batch.erase_payload_key(proposal_ref)
            transition_ref = self._append_transition(
                batch,
                transition_kind="MEMORY_COMMIT",
                triggering_action_ref=consent_decision_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head_ref,
                after_snapshot_ref=snapshot_ref,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.18-EXACT-GRANTED-COMMIT",
                    "authorizing_transition_ref": consent_decision_ref,
                },
                action_record_refs=[
                    commit_ref,
                    receipt_ref,
                    episode_ref,
                    snapshot_ref,
                    head_ref,
                    disposition_ref,
                ],
                validation_results=[
                    "atomic_commit",
                    "exact_proposal_and_consent",
                    "receipt_preserves_terms",
                    f"staging_key_destroyed:{destroyed_key_commitment}",
                    "one_episode_vertical_slice_limit",
                ],
                affected_selectors=[{"episode_ref": episode_ref}],
                status="applied",
            )
        checkpoint = self.store.verify_checkpoint()
        return {
            "proposal_ref": proposal_ref,
            "consent_decision_ref": consent_decision_ref,
            "memory_commit_ref": commit_ref,
            "committed_consent_receipt_ref": receipt_ref,
            "episode_ref": episode_ref,
            "before_head_ref": head.record_hash,
            "before_snapshot_ref": snapshot.record_hash,
            "after_head_ref": head_ref,
            "after_snapshot_ref": snapshot_ref,
            "governance_transition_ref": transition_ref,
            "authority_checkpoint_epoch": checkpoint.authority_checkpoint_epoch,
            "authority_checkpoint_hash": checkpoint.checkpoint_hash,
            "transaction_id": batch.transaction_id,
            "atomic": True,
            "proposal_staging_payload_erased": True,
            "idempotent_replay": False,
        }

    def _completed_commit_result(
        self,
        *,
        proposal_ref: str,
        consent_decision_ref: str,
    ) -> dict[str, Any] | None:
        proposal_commit: DurableRecord | None = None
        for ref in self.store.list_record_refs(record_type="MemoryCommitRecord"):
            record = self.store.read_record(ref)
            if record.payload.get("proposal_ref") != proposal_ref:
                continue
            proposal_commit = record
            if record.payload.get("consent_decision_ref") != consent_decision_ref:
                raise ContinuityProtocolError(
                    "This proposal was already committed under a different consent decision."
                )
            break
        if proposal_commit is None:
            return None
        receipt_ref = next(
            (
                ref
                for ref in self.store.list_record_refs(
                    record_type="CommittedConsentReceiptRecord"
                )
                if self.store.read_record(ref).payload.get("memory_commit_ref")
                == proposal_commit.record_hash
            ),
            None,
        )
        if receipt_ref is None:
            raise ContinuityProtocolError("The completed commit is missing its consent receipt.")
        transition_ref = next(
            (
                ref
                for ref in self.store.list_record_refs(record_type="GovernanceTransitionRecord")
                if proposal_commit.record_hash
                in self.store.read_record(ref).payload.get("action_record_refs", [])
            ),
            None,
        )
        if transition_ref is None:
            raise ContinuityProtocolError("The completed commit is missing its governance transition.")
        payload = proposal_commit.payload
        committed_refs = [str(ref) for ref in payload.get("committed_record_refs", [])]
        if len(committed_refs) != 1:
            raise ContinuityProtocolError("The bounded commit has an invalid committed-record set.")
        committed = self.store.read_record(committed_refs[0])
        checkpoint = self.store.verify_checkpoint()
        result: dict[str, Any] = {
            "proposal_ref": proposal_ref,
            "consent_decision_ref": consent_decision_ref,
            "memory_commit_ref": proposal_commit.record_hash,
            "committed_consent_receipt_ref": receipt_ref,
            "before_head_ref": payload["before_head_refs"][0],
            "before_snapshot_ref": payload["precondition_snapshot_ref"],
            "after_head_ref": payload["after_head_ref"],
            "after_snapshot_ref": payload["new_snapshot_ref"],
            "governance_transition_ref": transition_ref,
            "authority_checkpoint_epoch_at_replay": checkpoint.authority_checkpoint_epoch,
            "authority_checkpoint_hash_at_replay": checkpoint.checkpoint_hash,
            "transaction_id": payload["transaction_id"],
            "atomic": True,
            "proposal_staging_payload_erased": True,
            "idempotent_replay": True,
        }
        if committed.record_type == "EpisodicEventRecord":
            result["episode_ref"] = committed.record_hash
            return result
        if committed.record_type != "CorrectionRecord":
            raise ContinuityProtocolError("The completed commit record type is unsupported.")
        relation = str(committed.payload["requested_relation"])
        target_ref = str(committed.payload["target_refs"][0])
        historical = self.store.read_record(target_ref)
        historical_hash = stable_hash(historical.payload)
        result.update(
            {
                "target_ref": target_ref,
                "correction_ref": committed.record_hash,
                "requested_relation": relation,
                "computed_target_status": payload["computed_target_status"],
                "original_record_ref_unchanged": historical.record_hash == target_ref,
                "original_payload_hash_before": payload[
                    "original_payload_hash_before"
                ],
                "original_payload_hash_after": historical_hash,
                "active_view_after": self.reduce_active_state(
                    str(payload["new_snapshot_ref"])
                ).as_dict(),
            }
        )
        return result

    def _commit_correction_proposal(
        self,
        *,
        proposal_ref: str,
        consent_decision_ref: str,
        proposal: DurableRecord,
        consent: DurableRecord,
        head: DurableRecord,
        snapshot: DurableRecord,
        policy: DurableRecord,
        deadline: str,
    ) -> dict[str, Any]:
        active_view_before = self.reduce_active_state(snapshot.record_hash)
        if active_view_before.original_ref is None or active_view_before.original_status != "active":
            raise ContinuityProtocolError(
                "A bounded correction requires one active, uncorrected original episode."
            )
        if active_view_before.correction_ref is not None:
            raise ContinuityProtocolError("The correction slice permits only one correction commit.")
        proposed_payloads = proposal.payload.get("proposed_record_payloads")
        if not isinstance(proposed_payloads, list) or len(proposed_payloads) != 1:
            raise ContinuityProtocolError("A correction proposal must contain exactly one record.")
        proposed = proposed_payloads[0]
        if not isinstance(proposed, dict):
            raise ContinuityProtocolError("The proposed correction is malformed.")
        target_refs = proposed.get("target_refs")
        if target_refs != [active_view_before.original_ref]:
            raise ContinuityProtocolError("The correction target changed before commit.")
        relation = str(proposed.get("requested_relation"))
        status_by_relation = {
            "qualifies": "qualified",
            "disputes": "disputed",
            "supersedes": "superseded",
        }
        computed_status = status_by_relation.get(relation)
        if computed_status is None:
            raise ContinuityProtocolError("The correction relation is unsupported.")
        correction_content = _normalize(
            str(proposed.get("corrected_or_qualifying_content", "")),
            field_name="correction content",
            maximum=2000,
        )
        session_ref = str(proposal.payload["source_refs"][0])
        original_ref = active_view_before.original_ref
        original_before = self.store.read_record(original_ref)
        original_payload_hash_before = stable_hash(original_before.payload)

        with self.store.write_batch() as batch:
            correction_ref = batch.append(
                "CorrectionRecord",
                "E",
                {
                    "target_refs": [original_ref],
                    "requested_relation": relation,
                    "corrected_or_qualifying_content": correction_content,
                    "asserting_actor_ref": proposed["asserting_actor_ref"],
                    "evidence_refs": list(proposed.get("evidence_refs", [])),
                    "epistemic_status": proposed["epistemic_status"],
                    "effective_from": utc_now(),
                    "provenance_refs": [
                        proposal_ref,
                        consent_decision_ref,
                        session_ref,
                        original_ref,
                    ],
                    "correction_state": "active_correction",
                    "publication_class": "private_local_only",
                    "relationship_retrieval_eligible": True,
                },
                session_id=proposal.session_id,
            )
            active_refs = (
                [original_ref, correction_ref]
                if relation == "qualifies"
                else [correction_ref]
            )
            inactive_refs = [] if relation == "qualifies" else [original_ref]
            source_binding = {
                "active_episode_refs": active_refs,
                "inactive_or_disputed_refs": inactive_refs,
                "correction_pair": {
                    "target_ref": original_ref,
                    "correction_ref": correction_ref,
                    "requested_relation": relation,
                    "computed_target_status": computed_status,
                },
            }
            snapshot_ref = batch.append(
                "MemorySnapshotRecord",
                "G",
                {
                    "lineage_genesis_ref": snapshot.payload["lineage_genesis_ref"],
                    "branch_id": "main",
                    "parent_snapshot_refs": [snapshot.record_hash],
                    "base_head_ref": head.record_hash,
                    "policy_ref": policy.record_hash,
                    "public_inheritance_manifest_refs": [],
                    "active_public_corpus_refs": [],
                    "active_episode_refs": active_refs,
                    "active_semantic_refs": [],
                    "active_relational_refs": [],
                    "inactive_or_disputed_refs": inactive_refs,
                    "revoked_refs": [],
                    "forget_pending_refs": [],
                    "forgotten_tombstone_refs": [],
                    "projection_manifest_refs": [],
                    "reducer_binding": self.reducer_binding,
                    "source_record_root": stable_hash(source_binding),
                    "created_by_transaction_id": batch.transaction_id,
                    "relationship_retrieval_eligible": False,
                },
            )
            head_ref = batch.append(
                "LineageHeadRecord",
                "G",
                {
                    "prior_head_refs": [head.record_hash],
                    "branch_id": "main",
                    "branch_sequence": int(head.payload["branch_sequence"]) + 1,
                    "commit_transaction_id": batch.transaction_id,
                    "snapshot_ref": snapshot_ref,
                    "policy_ref": policy.record_hash,
                    "head_status": "active",
                    "relationship_retrieval_eligible": False,
                },
            )
            commit_ref = batch.append(
                "MemoryCommitRecord",
                "G",
                {
                    "commit_kind": "append_only_correction",
                    "original_record_ref": original_ref,
                    "original_payload_hash_before": original_payload_hash_before,
                    "computed_target_status": computed_status,
                    "proposal_ref": proposal_ref,
                    "consent_decision_ref": consent_decision_ref,
                    "source_session_refs": [session_ref],
                    "precondition_snapshot_ref": snapshot.record_hash,
                    "before_head_refs": [head.record_hash],
                    "committed_record_refs": [correction_ref],
                    "new_snapshot_ref": snapshot_ref,
                    "after_head_ref": head_ref,
                    "validation_results": [
                        "active_head_unchanged",
                        "exact_correction_consent_valid",
                        "target_exists_and_is_authorized",
                        "original_record_unchanged",
                        "payload_aes256_gcm_protected",
                        "deterministic_active_state_reduced",
                        "current_cycle_input_unchanged",
                    ],
                    "transaction_id": batch.transaction_id,
                    "idempotency_key": stable_hash(
                        {"proposal_ref": proposal_ref, "consent_ref": consent_decision_ref}
                    ),
                    "committed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=proposal.session_id,
            )
            receipt_ref = batch.append(
                "CommittedConsentReceiptRecord",
                "G",
                {
                    "memory_commit_ref": commit_ref,
                    "proposal_ref": proposal_ref,
                    "consent_decision_ref": consent_decision_ref,
                    "canonical_purpose": proposal.payload["purpose"],
                    "canonical_scope": proposal.payload["scope"],
                    "canonical_audience": proposal.payload["audience"],
                    "canonical_retention": proposal.payload["retention"],
                    "canonical_allowed_derivations": proposal.payload["allowed_derivations"],
                    "proposal_content_root": proposal.payload["proposal_content_root"],
                    "committed_record_root": stable_hash([correction_ref]),
                    "effect_preview": proposal.payload["retrieval_effect_preview"],
                    "risk_preview": proposal.payload["risk_disclosure"],
                    "forgetting_effect_preview": proposal.payload["forgetting_effect_preview"],
                    "effective_commit_deadline": deadline,
                    "committed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                    "dependency_refs": [original_ref, correction_ref],
                },
                session_id=proposal.session_id,
            )
            disposition_ref = batch.append(
                "ProposalDispositionRecord",
                "G",
                {
                    "proposal_ref": proposal_ref,
                    "disposition": "committed_and_staging_payload_erased",
                    "cause_ref": commit_ref,
                    "committed_consent_receipt_ref": receipt_ref,
                    "relationship_retrieval_eligible": False,
                    "disposed_at_utc": utc_now(),
                },
                session_id=proposal.session_id,
            )
            destroyed_key_commitment = batch.erase_payload_key(proposal_ref)
            transition_ref = self._append_transition(
                batch,
                transition_kind="MEMORY_COMMIT",
                triggering_action_ref=consent_decision_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head_ref,
                after_snapshot_ref=snapshot_ref,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.20-APPEND-ONLY-CORRECTION",
                    "authorizing_transition_ref": consent_decision_ref,
                },
                action_record_refs=[
                    commit_ref,
                    receipt_ref,
                    correction_ref,
                    snapshot_ref,
                    head_ref,
                    disposition_ref,
                ],
                validation_results=[
                    "atomic_correction_commit",
                    "exact_proposal_and_consent",
                    "original_event_preserved",
                    f"target_status:{computed_status}",
                    "corrected_target_cannot_surface_alone",
                    f"staging_key_destroyed:{destroyed_key_commitment}",
                    "one_correction_slice_limit",
                ],
                affected_selectors=[
                    {"target_ref": original_ref},
                    {"correction_ref": correction_ref},
                ],
                status="applied",
            )
        checkpoint = self.store.verify_checkpoint()
        original_after = self.store.read_record(original_ref)
        original_payload_hash_after = stable_hash(original_after.payload)
        if original_after.record_hash != original_ref or original_payload_hash_after != original_payload_hash_before:
            raise ContinuityProtocolError("The original episode changed during correction commit.")
        active_view_after = self.reduce_active_state(snapshot_ref)
        if active_view_after.original_status != computed_status:
            raise ContinuityProtocolError("The correction reducer produced an unexpected state.")
        return {
            "proposal_ref": proposal_ref,
            "consent_decision_ref": consent_decision_ref,
            "memory_commit_ref": commit_ref,
            "committed_consent_receipt_ref": receipt_ref,
            "target_ref": original_ref,
            "correction_ref": correction_ref,
            "requested_relation": relation,
            "computed_target_status": computed_status,
            "before_head_ref": head.record_hash,
            "before_snapshot_ref": snapshot.record_hash,
            "after_head_ref": head_ref,
            "after_snapshot_ref": snapshot_ref,
            "governance_transition_ref": transition_ref,
            "authority_checkpoint_epoch": checkpoint.authority_checkpoint_epoch,
            "authority_checkpoint_hash": checkpoint.checkpoint_hash,
            "transaction_id": batch.transaction_id,
            "atomic": True,
            "proposal_staging_payload_erased": True,
            "idempotent_replay": False,
            "original_record_ref_unchanged": original_after.record_hash == original_ref,
            "original_payload_hash_before": original_payload_hash_before,
            "original_payload_hash_after": original_payload_hash_after,
            "active_view_after": active_view_after.as_dict(),
        }

    def _episode_context(
        self,
        session: ActiveContinuitySession,
        *,
        include_history: bool,
    ) -> dict[str, Any]:
        view = self.reduce_active_state(session.start_snapshot_ref)
        if view.original_ref is None:
            eligible: tuple[str, ...] = ()
            ordered: tuple[dict[str, Any], ...] = ()
            correction_pairs: list[dict[str, str]] = []
        else:
            original = self.store.read_record(view.original_ref)
            if original.record_type != "EpisodicEventRecord" or original.record_class != "E":
                raise ContinuityProtocolError("The original history record is not episodic evidence.")
            if not original.payload.get("relationship_retrieval_eligible"):
                raise ContinuityProtocolError("An ineligible original entered relationship context.")
            if view.correction_ref is None:
                eligible = (view.original_ref,)
                ordered = (
                    {
                        "record_ref": view.original_ref,
                        "record_type": "episodic_event",
                        "content": str(original.payload["content"]),
                        "status": "active",
                        "correction_ref": None,
                        "correction_relation": None,
                        "correction_content": None,
                        "may_surface_alone_as_current": True,
                    },
                )
                correction_pairs = []
            else:
                correction = self.store.read_record(view.correction_ref)
                if correction.record_type != "CorrectionRecord" or correction.record_class != "E":
                    raise ContinuityProtocolError("The paired correction is not episodic evidence.")
                if not correction.payload.get("relationship_retrieval_eligible"):
                    raise ContinuityProtocolError("An ineligible correction entered relationship context.")
                eligible = (view.original_ref, view.correction_ref)
                ordered = (
                    {
                        "record_ref": view.original_ref,
                        "record_type": "episode_with_correction",
                        "content": str(original.payload["content"]),
                        "status": str(view.original_status),
                        "correction_ref": view.correction_ref,
                        "correction_relation": str(view.correction_relation),
                        "correction_content": str(
                            correction.payload["corrected_or_qualifying_content"]
                        ),
                        "correction_epistemic_status": str(
                            correction.payload["epistemic_status"]
                        ),
                        "historical_only": view.original_status in {"disputed", "superseded"},
                        "may_surface_alone_as_current": False,
                    },
                )
                correction_pairs = [
                    {
                        "target_ref": view.original_ref,
                        "correction_ref": view.correction_ref,
                        "requested_relation": str(view.correction_relation),
                        "computed_target_status": str(view.original_status),
                    }
                ]
        selected = eligible if include_history else ()
        selected_ordered = ordered if include_history else ()
        excluded = [] if include_history else [
            {"record_ref": ref, "reason": "matched_no_history_control"} for ref in eligible
        ]
        return {
            "eligible_refs": eligible,
            "selected_refs": selected,
            "ordered_history": selected_ordered,
            "excluded": excluded,
            "correction_pairs": correction_pairs if include_history else [],
            "active_view": view,
        }

    def respond(
        self,
        session_id: str,
        *,
        include_history: bool,
        condition_label: str,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        history_labels = {"history", "uncorrected_history", "corrected_history"}
        if condition_label not in {*history_labels, "matched_no_history"}:
            raise ValueError("Unknown continuity condition label.")
        if include_history != (condition_label in history_labels):
            raise ValueError("The condition label and history selection disagree.")
        session.stage_sequence += 1
        context = self._episode_context(
            session, include_history=include_history
        )
        selected_refs = context["selected_refs"]
        ordered_history = context["ordered_history"]
        excluded = context["excluded"]
        eligible_refs = context["eligible_refs"]
        correction_pairs = context["correction_pairs"]
        active_view: ActiveEpisodeView = context["active_view"]
        rendered_context = canonical_json(list(ordered_history))
        head, current_snapshot, policy = self._current_state()
        with self.store.write_batch() as batch:
            read_set_ref = batch.append(
                "MemoryReadSetRecord",
                "G",
                {
                    "session_open_ref": session.session_open_ref,
                    "snapshot_ref": session.start_snapshot_ref,
                    "query_plan_hash": stable_hash("exact-correction-paired-history:v0.2"),
                    "ranking_policy_hash": stable_hash("no-ranking-one-episode-one-correction:v0.2"),
                    "eligible_candidate_refs": list(eligible_refs),
                    "excluded_candidate_refs_with_reasons": excluded,
                    "selected_refs_in_order": list(selected_refs),
                    "selected_record_classes_in_order": ["E" for _ in selected_refs],
                    "selected_memory_refs_in_order": list(selected_refs),
                    "selected_public_corpus_refs_in_order": [],
                    "correction_pair_refs": correction_pairs,
                    "ordered_context_items": list(ordered_history),
                    "active_state": active_view.as_dict(),
                    "rendered_context_hash": stable_hash(rendered_context),
                    "rendered_memory_partition_hash": stable_hash(list(ordered_history)),
                    "rendered_public_corpus_partition_hash": stable_hash([]),
                    "rendered_context_token_count": len(rendered_context.split()),
                    "receiving_stage": f"continuity_response:{condition_label}",
                    "stage_sequence": session.stage_sequence,
                    "prior_read_set_refs": list(session.read_set_refs),
                    "active_invalidation_refs": [],
                    "sealed_immediately_before_stage": True,
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="TRACE_APPEND",
                triggering_action_ref=session.session_open_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=current_snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=current_snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.9-STAGE-READ-SET",
                    "authorizing_transition_ref": session.session_open_ref,
                },
                action_record_refs=[read_set_ref],
                validation_results=[
                    "session_start_snapshot_frozen",
                    "read_set_sealed_before_stage",
                    "one_episode_one_correction_budget",
                    "correction_pairing_status_preserved",
                    "public_corpus_absent",
                ],
                affected_selectors=[{"read_set_ref": read_set_ref}],
                status="recorded_no_active_change",
            )
        session.read_set_refs.append(read_set_ref)

        response_text, call = self.responder.respond(
            intention=session.intention,
            ordered_history=ordered_history,
        )
        response_hash = str(call["output_hash"])
        head, current_snapshot, policy = self._current_state()
        delivery_id = secrets.token_hex(16)
        checkpoint_before_prepare = self.store.verify_checkpoint()
        with self.store.write_batch() as batch:
            stage_output_ref = batch.append(
                "StageOutputBindingRecord",
                "G",
                {
                    "session_open_ref": session.session_open_ref,
                    "stage_id": f"continuity_response:{condition_label}",
                    "stage_sequence": session.stage_sequence,
                    "causal_read_set_ref": read_set_ref,
                    "input_stage_output_refs": [],
                    "capture_mode": "volatile_dependency_only",
                    "opaque_output_commitment": self.store.keyed_commitment(response_text),
                    "output_payload_locator": None,
                    "output_ciphertext_hash": None,
                    "model_call_binding": call,
                    "completed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            embodied_ref = batch.append(
                "StageOutputDispositionRecord",
                "G",
                {
                    "stage_output_ref": stage_output_ref,
                    "disposition": "embodied",
                    "cause_record_ref": read_set_ref,
                    "delivery_record_ref": None,
                    "effective_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            preparation_ref = batch.append(
                "DeliveryPreparationRecord",
                "G",
                {
                    "delivery_id": delivery_id,
                    "stage_output_ref": stage_output_ref,
                    "recipient_principal_ref": self.store.owner_principal_ref,
                    "channel_binding": "local_python_return_value:v0.1",
                    "payload_commitment": self.store.keyed_commitment(response_text),
                    "idempotency_key": stable_hash(
                        {"delivery_id": delivery_id, "response_hash": response_hash}
                    ),
                    "prepared_governance_head_ref": checkpoint_before_prepare.authoritative_governance_head_hash,
                    "prepared_authority_checkpoint_ref": checkpoint_before_prepare.checkpoint_hash,
                    "prepared_invalidation_sequence": 0,
                    "release_expires_at_utc": _deadline(1),
                    "prepared_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="DELIVERY_PREPARE",
                triggering_action_ref=session.session_open_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=current_snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=current_snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.11-DELIVERY-PREPARE",
                    "authorizing_transition_ref": session.session_open_ref,
                },
                action_record_refs=[stage_output_ref, embodied_ref, preparation_ref],
                validation_results=[
                    "output_payload_not_durably_captured",
                    "exact_read_set_bound",
                    "recipient_bound",
                    "no_bytes_released_during_prepare",
                ],
                affected_selectors=[{"delivery_id": delivery_id}],
                status="recorded_no_active_change",
            )
        head, current_snapshot, policy = self._current_state()
        checkpoint_before_release = self.store.verify_checkpoint()
        with self.store.write_batch() as batch:
            release_ref = batch.append(
                "DeliveryReleaseRecord",
                "G",
                {
                    "delivery_preparation_ref": preparation_ref,
                    "delivery_id": delivery_id,
                    "release_governance_parent_ref": checkpoint_before_release.authoritative_governance_head_hash,
                    "release_authority_checkpoint_ref": checkpoint_before_release.checkpoint_hash,
                    "release_invalidation_sequence": 0,
                    "authorization_validation": [
                        "same_owner_recipient",
                        "no_invalidation",
                        "release_lease_active",
                    ],
                    "release_status": "authorized_once",
                    "denial_reason": None,
                    "release_lease_expires_at_utc": _deadline(1),
                    "released_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            disposition_ref = batch.append(
                "StageOutputDispositionRecord",
                "G",
                {
                    "stage_output_ref": stage_output_ref,
                    "disposition": "release_authorized",
                    "cause_record_ref": release_ref,
                    "delivery_record_ref": release_ref,
                    "effective_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="DELIVERY_RELEASE",
                triggering_action_ref=preparation_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=current_snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=current_snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.11-SERIALIZED-RELEASE",
                    "authorizing_transition_ref": preparation_ref,
                },
                action_record_refs=[release_ref, disposition_ref],
                validation_results=[
                    "write_ahead_release_recorded",
                    "no_invalidation_precedes_release",
                    "one_use_delivery_id",
                ],
                affected_selectors=[{"delivery_id": delivery_id}],
                status="recorded_no_active_change",
            )
        session.stage_output_refs.append(stage_output_ref)
        delivery = {
            "delivery_id": delivery_id,
            "stage_output_ref": stage_output_ref,
            "preparation_ref": preparation_ref,
            "release_ref": release_ref,
            "response_hash": response_hash,
            "status": "release_authorized",
        }
        session.delivery_refs.append(delivery)
        return {
            "condition": condition_label,
            "session_id": session_id,
            "runtime_instance_id": self.runtime_instance_id,
            "snapshot_ref": session.start_snapshot_ref,
            "read_set_ref": read_set_ref,
            "eligible_episode_refs": list(eligible_refs),
            "selected_episode_refs": list(selected_refs),
            "correction_pair_refs": correction_pairs,
            "active_state": active_view.as_dict(),
            "ordered_context": list(ordered_history),
            "rendered_context_hash": stable_hash(rendered_context),
            "response": response_text,
            "response_hash": response_hash,
            "model_call": call,
            "delivery": delivery,
            "history_conditioned": bool(selected_refs),
        }

    def acknowledge_delivery(
        self,
        session_id: str,
        *,
        delivery_id: str,
        response_hash: str,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        matching = next(
            (item for item in session.delivery_refs if item["delivery_id"] == delivery_id),
            None,
        )
        if matching is None:
            raise ContinuityProtocolError("The delivery ID is not part of this session.")
        if matching["status"] != "release_authorized":
            raise ContinuityProtocolError("The delivery already has a terminal outcome.")
        if not secrets.compare_digest(matching["response_hash"], response_hash):
            raise ContinuityProtocolError("The delivery acknowledgement binds a different response.")
        head, snapshot, policy = self._current_state()
        with self.store.write_batch() as batch:
            outcome_ref = batch.append(
                "DeliveryOutcomeRecord",
                "G",
                {
                    "delivery_release_ref": matching["release_ref"],
                    "delivery_id": delivery_id,
                    "transport_attempt_commitment": self.store.keyed_commitment(
                        {"delivery_id": delivery_id, "response_hash": response_hash}
                    ),
                    "outcome": "acknowledged",
                    "acknowledgement_binding": stable_hash(
                        {"delivery_id": delivery_id, "response_hash": response_hash}
                    ),
                    "reconciliation_required": False,
                    "automatic_retry_allowed": False,
                    "completed_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            disposition_ref = batch.append(
                "StageOutputDispositionRecord",
                "G",
                {
                    "stage_output_ref": matching["stage_output_ref"],
                    "disposition": "delivery_acknowledged",
                    "cause_record_ref": outcome_ref,
                    "delivery_record_ref": outcome_ref,
                    "effective_at_utc": utc_now(),
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="DELIVERY_OUTCOME",
                triggering_action_ref=matching["release_ref"],
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-8.11-DELIVERY-ACK",
                    "authorizing_transition_ref": matching["release_ref"],
                },
                action_record_refs=[outcome_ref, disposition_ref],
                validation_results=[
                    "response_hash_exact",
                    "delivery_id_one_use",
                    "acknowledgement_recorded",
                    "automatic_retry_disabled",
                ],
                affected_selectors=[{"delivery_id": delivery_id}],
                status="recorded_no_active_change",
            )
        matching["status"] = "acknowledged"
        matching["outcome_ref"] = outcome_ref
        return {
            "delivery_id": delivery_id,
            "outcome_ref": outcome_ref,
            "status": "acknowledged",
            "response_hash": response_hash,
        }

    def close_session(self, session_id: str, *, principal: str) -> dict[str, Any]:
        owner = self._require_owner(principal)
        session = self._session(session_id)
        unresolved = [item for item in session.delivery_refs if item["status"] != "acknowledged"]
        if unresolved:
            raise ContinuityProtocolError(
                "Every released response must be acknowledged or explicitly reconciled before this proof session closes."
            )
        head, snapshot, policy = self._current_state()
        commit_refs = self.store.list_record_refs(
            record_type="MemoryCommitRecord", session_id=session_id
        )
        consent_refs = self.store.list_record_refs(
            record_type="ConsentDecisionRecord", session_id=session_id
        )
        consent_status_by_proposal: dict[str, str] = {}
        for ref in consent_refs:
            decision = self.store.read_record(ref)
            proposal_ref = str(decision.payload["proposal_ref"])
            consent_status_by_proposal[proposal_ref] = (
                "REJECTED"
                if decision.payload["decision"] == "reject"
                else "GRANTED_PENDING_COMMIT"
            )
        with self.store.write_batch() as batch:
            close_ref = batch.append(
                "SessionCloseRecord",
                "G",
                {
                    "session_open_ref": session.session_open_ref,
                    "read_set_refs": list(session.read_set_refs),
                    "model_call_trace_root": stable_hash(session.stage_output_refs),
                    "graph_trace_root": stable_hash(
                        {
                            "read_sets": session.read_set_refs,
                            "stage_outputs": session.stage_output_refs,
                        }
                    ),
                    "kernel_activation_trace_refs": [],
                    "embodied_output_ref": (
                        None if not session.stage_output_refs else session.stage_output_refs[-1]
                    ),
                    "delivery_preparation_refs": [
                        item["preparation_ref"] for item in session.delivery_refs
                    ],
                    "delivery_release_refs": [item["release_ref"] for item in session.delivery_refs],
                    "delivery_outcome_refs": [item["outcome_ref"] for item in session.delivery_refs],
                    "delivery_status_at_close": (
                        "not_prepared" if not session.delivery_refs else "acknowledged"
                    ),
                    "confirmed_or_possible_disclosure_refs": [
                        item["outcome_ref"] for item in session.delivery_refs
                    ],
                    "feedback_refs": [],
                    "memory_proposal_refs": list(session.proposal_refs),
                    "pending_proposal_refs_at_close": [],
                    "proposal_status_at_close": [
                        {
                            "proposal_ref": ref,
                            "reducer_status": (
                                "COMMITTED"
                                if commit_refs
                                else consent_status_by_proposal.get(ref, "PENDING_CONSENT")
                            ),
                            "status_source_refs": list((*consent_refs, *commit_refs)),
                        }
                        for ref in session.proposal_refs
                    ],
                    "observed_lineage_head_ref_at_close": head.record_hash,
                    "starting_snapshot_ref": session.start_snapshot_ref,
                    "runtime_instance_id": session.runtime_instance_id,
                    "closed_at_utc": utc_now(),
                    "completion_status": "complete",
                    "relationship_retrieval_eligible": False,
                },
                session_id=session_id,
            )
            self._append_transition(
                batch,
                transition_kind="SESSION_CLOSE",
                triggering_action_ref=session.session_open_ref,
                before_head_ref=head.record_hash,
                before_snapshot_ref=snapshot.record_hash,
                after_head_ref=head.record_hash,
                after_snapshot_ref=snapshot.record_hash,
                policy_ref=policy.record_hash,
                authority_basis={
                    "basis_kind": "constitutional_system_duty",
                    "authenticated_actor_ref": None,
                    "constitutional_rule_id": "MEMORY-SESSION-CLOSE",
                    "authorizing_transition_ref": session.session_open_ref,
                },
                action_record_refs=[close_ref],
                validation_results=[
                    "all_delivery_outcomes_terminal",
                    "starting_snapshot_unchanged",
                    "closing_head_observed_not_retroactive",
                    "proposal_lifecycle_independent",
                ],
                affected_selectors=[{"session_id": session_id}],
                status="recorded_no_active_change",
            )
        session.closed = True
        return {
            "session_id": session_id,
            "session_open_ref": session.session_open_ref,
            "session_close_ref": close_ref,
            "runtime_instance_id": session.runtime_instance_id,
            "starting_snapshot_ref": session.start_snapshot_ref,
            "observed_lineage_head_ref_at_close": head.record_hash,
            "memory_commit_refs": list(commit_refs),
            "delivery_status": "not_prepared" if not session.delivery_refs else "acknowledged",
            "completion_status": "complete",
            "principal_ref": owner,
        }

    @staticmethod
    def causal_comparison(
        history_result: dict[str, Any],
        no_history_result: dict[str, Any],
    ) -> dict[str, Any]:
        history_call = history_result["model_call"]
        control_call = no_history_result["model_call"]
        budget_match = bool(
            history_call["provider_id"] == control_call["provider_id"]
            and history_call["config_hash"] == control_call["config_hash"]
            and history_call["call_count"] == control_call["call_count"]
            and history_call["max_output_tokens"] == control_call["max_output_tokens"]
        )
        history_effect = bool(
            budget_match
            and history_result["selected_episode_refs"]
            and not no_history_result["selected_episode_refs"]
            and history_result["response_hash"] != no_history_result["response_hash"]
            and history_call["selected_history_refs"]
            == history_result["selected_episode_refs"]
        )
        return {
            "comparison_id": stable_hash(
                {
                    "history_response": history_result["response_hash"],
                    "no_history_response": no_history_result["response_hash"],
                    "history_read_set": history_result["read_set_ref"],
                    "no_history_read_set": no_history_result["read_set_ref"],
                }
            ),
            "same_start_snapshot": (
                history_result["snapshot_ref"] == no_history_result["snapshot_ref"]
            ),
            "same_provider_and_budget": budget_match,
            "history_selected_refs": history_result["selected_episode_refs"],
            "no_history_selected_refs": no_history_result["selected_episode_refs"],
            "response_hashes_differ": (
                history_result["response_hash"] != no_history_result["response_hash"]
            ),
            "history_effect_observed": history_effect,
            "result": "mechanical_continuity_effect" if history_effect else "null_or_protocol_failure",
            "claim_boundary": (
                "This comparison tests exact memory causality only; it is not evidence of consciousness, "
                "identity continuity, autonomous preference, or broader relationship intelligence."
            ),
        }

    @staticmethod
    def correction_causal_comparison(
        uncorrected_result: dict[str, Any],
        corrected_result: dict[str, Any],
        no_history_result: dict[str, Any],
    ) -> dict[str, Any]:
        calls = [
            uncorrected_result["model_call"],
            corrected_result["model_call"],
            no_history_result["model_call"],
        ]
        provider_and_budget_match = all(
            call["provider_id"] == calls[0]["provider_id"]
            and call["config_hash"] == calls[0]["config_hash"]
            and call["call_count"] == calls[0]["call_count"]
            and call["max_output_tokens"] == calls[0]["max_output_tokens"]
            for call in calls[1:]
        )
        uncorrected_view = uncorrected_result["active_state"]
        corrected_view = corrected_result["active_state"]
        no_history_view = no_history_result["active_state"]
        original_ref = uncorrected_view.get("original_ref")
        correction_ref = corrected_view.get("correction_ref")
        original_pair_preserved = bool(
            original_ref
            and correction_ref
            and corrected_result["selected_episode_refs"] == [original_ref, correction_ref]
            and corrected_result["correction_pair_refs"]
            and corrected_result["correction_pair_refs"][0]["target_ref"] == original_ref
            and corrected_result["correction_pair_refs"][0]["correction_ref"] == correction_ref
        )
        no_history_clean = bool(
            not no_history_result["selected_episode_refs"]
            and not no_history_result["correction_pair_refs"]
            and no_history_result["response_hash"] != corrected_result["response_hash"]
        )
        effect = bool(
            provider_and_budget_match
            and uncorrected_view.get("original_status") == "active"
            and corrected_view.get("original_status")
            in {"qualified", "disputed", "superseded"}
            and corrected_view.get("original_may_surface_alone_as_current") is False
            and no_history_view == corrected_view
            and corrected_result["snapshot_ref"] == no_history_result["snapshot_ref"]
            and uncorrected_result["snapshot_ref"] != corrected_result["snapshot_ref"]
            and uncorrected_result["selected_episode_refs"] == [original_ref]
            and calls[0]["selected_history_refs"] == [original_ref]
            and calls[1]["selected_history_refs"] == [original_ref, correction_ref]
            and original_pair_preserved
            and no_history_clean
            and len(
                {
                    uncorrected_result["response_hash"],
                    corrected_result["response_hash"],
                    no_history_result["response_hash"],
                }
            )
            == 3
        )
        return {
            "comparison_id": stable_hash(
                {
                    "uncorrected_response": uncorrected_result["response_hash"],
                    "corrected_response": corrected_result["response_hash"],
                    "no_history_response": no_history_result["response_hash"],
                    "uncorrected_read_set": uncorrected_result["read_set_ref"],
                    "corrected_read_set": corrected_result["read_set_ref"],
                    "no_history_read_set": no_history_result["read_set_ref"],
                }
            ),
            "same_provider_and_budget": provider_and_budget_match,
            "uncorrected_snapshot_precedes_corrected_snapshot": (
                uncorrected_result["snapshot_ref"] != corrected_result["snapshot_ref"]
            ),
            "corrected_and_no_history_same_snapshot": (
                corrected_result["snapshot_ref"] == no_history_result["snapshot_ref"]
            ),
            "original_ref": original_ref,
            "correction_ref": correction_ref,
            "uncorrected_status": uncorrected_view.get("original_status"),
            "corrected_status": corrected_view.get("original_status"),
            "original_pair_preserved": original_pair_preserved,
            "original_cannot_surface_alone_as_current": (
                corrected_view.get("original_may_surface_alone_as_current") is False
            ),
            "no_history_has_no_correction_callback": no_history_clean,
            "all_three_response_hashes_distinct": len(
                {
                    uncorrected_result["response_hash"],
                    corrected_result["response_hash"],
                    no_history_result["response_hash"],
                }
            )
            == 3,
            "correction_effect_observed": effect,
            "result": "mechanical_correction_effect" if effect else "null_or_protocol_failure",
            "claim_boundary": (
                "This comparison tests append-only correction causality and status-preserving retrieval only; "
                "it is not evidence of consciousness, autonomous divergence, or relationship intelligence."
            ),
        }

    def historical_episode(self, record_ref: str) -> dict[str, Any]:
        record = self.store.read_record(record_ref)
        if record.record_type != "EpisodicEventRecord" or record.record_class != "E":
            raise ContinuityProtocolError("The requested record is not an original episodic event.")
        view = self.reduce_active_state()
        if view.original_ref != record_ref:
            raise ContinuityProtocolError("The episode is not part of the current correction lineage.")
        return {
            "record_ref": record.record_hash,
            "record_type": record.record_type,
            "payload": dict(record.payload),
            "payload_hash": stable_hash(record.payload),
            "current_status": view.original_status,
            "correction_ref": view.correction_ref,
            "correction_relation": view.correction_relation,
            "historically_recoverable": True,
            "may_surface_alone_as_current": view.correction_ref is None,
        }

    def inspect_receipt(self, receipt_ref: str) -> dict[str, Any]:
        receipt = self.store.read_record(receipt_ref)
        if receipt.record_type != "CommittedConsentReceiptRecord":
            raise ContinuityProtocolError("The requested record is not a consent receipt.")
        return dict(receipt.payload)

    def active_episode_refs(self) -> tuple[str, ...]:
        snapshot = self.store.current_snapshot()
        return tuple(str(ref) for ref in snapshot.payload.get("active_episode_refs", []))

    def verify(self) -> dict[str, Any]:
        checkpoint = self.store.verify_checkpoint()
        chain = self.store.verify_chain()
        return {
            **chain,
            "store_id": self.store.store_id,
            "lineage_id": self.store.lineage_id,
            "authority_checkpoint_epoch": checkpoint.authority_checkpoint_epoch,
            "authority_checkpoint_hash": checkpoint.checkpoint_hash,
            "active_head_ref": checkpoint.active_lineage_head_hash,
            "active_snapshot_ref": checkpoint.active_snapshot_root,
            "service_version": self.service_version,
        }


__all__ = [
    "ActiveEpisodeView",
    "ContinuityProtocolError",
    "ContinuityService",
    "ContinuityStore",
    "ContinuityStoreError",
    "DeterministicContinuityResponder",
    "PayloadUnavailableError",
    "ResponderBinding",
]
