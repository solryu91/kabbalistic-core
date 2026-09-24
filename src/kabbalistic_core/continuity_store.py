"""Minimal encrypted, append-only store for the v0.1 continuity proof.

This module deliberately implements only the storage mechanics needed by the
released DR-10 vertical slice.  It is not a general memory database, archive
store, semantic index, or forgetting implementation.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
from threading import RLock
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .models import canonical_json, stable_hash


class ContinuityStoreError(RuntimeError):
    """Base error for fail-closed continuity storage."""


class CheckpointMismatchError(ContinuityStoreError):
    """The content store and independent authority checkpoint do not agree."""


class RecordIntegrityError(ContinuityStoreError):
    """A durable record failed hash, ancestry, or authenticated-decryption checks."""


class PayloadUnavailableError(ContinuityStoreError):
    """A payload key has been disposed and its encrypted payload is unreadable."""


class RecordNotFoundError(ContinuityStoreError):
    """A requested durable record is absent."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_master_key(master_key: bytes) -> bytes:
    if not isinstance(master_key, bytes) or len(master_key) != 32:
        raise ValueError("The continuity master key must be exactly 32 bytes.")
    return master_key


@dataclass(frozen=True, slots=True)
class DurableRecord:
    record_hash: str
    record_type: str
    record_class: str
    lineage_id: str
    session_id: str | None
    sequence: int
    previous_record_hash: str | None
    transaction_id: str
    created_at_utc: str
    ciphertext_hash: str
    payload_key_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuthorityCheckpoint:
    checkpoint_version: str
    store_id: str
    lineage_id: str
    authority_checkpoint_epoch: int
    authoritative_governance_sequence: int
    authoritative_governance_head_hash: str
    active_lineage_head_hash: str
    active_snapshot_root: str
    record_chain_head_hash: str
    revocation_epoch: int
    erasure_epoch: int
    destroyed_key_commitment_root: str
    prior_authority_checkpoint_hash: str | None
    pending_or_finalized: str
    expected_transaction_commitment: str
    updated_at_utc: str
    checkpoint_authenticator: str

    def body(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "store_id": self.store_id,
            "lineage_id": self.lineage_id,
            "authority_checkpoint_epoch": self.authority_checkpoint_epoch,
            "authoritative_governance_sequence": self.authoritative_governance_sequence,
            "authoritative_governance_head_hash": self.authoritative_governance_head_hash,
            "active_lineage_head_hash": self.active_lineage_head_hash,
            "active_snapshot_root": self.active_snapshot_root,
            "record_chain_head_hash": self.record_chain_head_hash,
            "revocation_epoch": self.revocation_epoch,
            "erasure_epoch": self.erasure_epoch,
            "destroyed_key_commitment_root": self.destroyed_key_commitment_root,
            "prior_authority_checkpoint_hash": self.prior_authority_checkpoint_hash,
            "pending_or_finalized": self.pending_or_finalized,
            "expected_transaction_commitment": self.expected_transaction_commitment,
            "updated_at_utc": self.updated_at_utc,
        }

    @property
    def checkpoint_hash(self) -> str:
        return stable_hash({**self.body(), "checkpoint_authenticator": self.checkpoint_authenticator})


class ContinuityStore:
    """Encrypted append-only record store with a separately persisted checkpoint.

    The SQLite file is the content domain.  ``checkpoint_path`` is the authority
    domain and must be excluded from ordinary content-only backups.  A caller-
    supplied 32-byte master key authenticates the checkpoint and wraps one fresh
    AES-256-GCM data key per payload.  The master key is never written here.
    """

    schema_version = "continuity-store:v0.1"
    checkpoint_version = "authority-checkpoint:v0.1"

    def __init__(
        self,
        database_path: Path,
        checkpoint_path: Path,
        master_key: bytes,
        *,
        require_initialized: bool = True,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if self.database_path == self.checkpoint_path:
            raise ValueError("The content store and authority checkpoint must be separate files.")
        if self.database_path.parent == self.checkpoint_path.parent:
            raise ValueError(
                "The content store and authority checkpoint must use separate directories "
                "so content-only backups cannot silently capture both trust domains."
            )
        self._master_key = _require_master_key(master_key)
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_connection()
        if require_initialized:
            try:
                self._assert_schema()
                self.verify_checkpoint()
            except Exception:
                self._connection.close()
                self._closed = True
                raise

    @classmethod
    def create(
        cls,
        database_path: Path,
        checkpoint_path: Path,
        master_key: bytes,
        *,
        owner_principal: str,
        lineage_kind: str = "private_local",
    ) -> "ContinuityStore":
        database = Path(database_path).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        if database.exists() or checkpoint.exists():
            raise FileExistsError("Continuity creation refuses to overwrite an existing store or checkpoint.")
        if not owner_principal.strip():
            raise ValueError("An owner principal is required.")
        if lineage_kind != "private_local":
            raise ValueError("The released vertical slice supports only one private local lineage.")
        database.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        store = cls(database, checkpoint, master_key, require_initialized=False)
        try:
            store._initialize_schema(owner_principal.strip(), lineage_kind)
            return store
        except Exception:
            store.close()
            for path in (database, checkpoint):
                if path.exists():
                    path.unlink()
            raise

    @classmethod
    def open(
        cls,
        database_path: Path,
        checkpoint_path: Path,
        master_key: bytes,
    ) -> "ContinuityStore":
        database = Path(database_path).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        if not database.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Both the continuity store and authority checkpoint are required.")
        return cls(database, checkpoint, master_key, require_initialized=True)

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA trusted_schema=OFF")

    def _initialize_schema(self, owner_principal: str, lineage_kind: str) -> None:
        store_id = secrets.token_hex(16)
        lineage_id = secrets.token_hex(16)
        checkpoint_domain_id = secrets.token_hex(16)
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE payload_keys (
                key_id TEXT PRIMARY KEY,
                wrap_nonce BLOB NOT NULL,
                wrapped_key BLOB NOT NULL,
                created_at_utc TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE destroyed_key_events (
                commitment TEXT PRIMARY KEY,
                record_hash TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                destroyed_at_utc TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE records (
                record_hash TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                record_class TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                session_id TEXT,
                sequence INTEGER NOT NULL UNIQUE,
                previous_record_hash TEXT,
                transaction_id TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                ciphertext_hash TEXT NOT NULL,
                payload_key_id TEXT NOT NULL,
                payload_nonce BLOB NOT NULL,
                payload_ciphertext BLOB NOT NULL
            );
            CREATE TRIGGER records_no_update
            BEFORE UPDATE ON records
            BEGIN SELECT RAISE(ABORT, 'append-only records cannot be updated'); END;
            CREATE TRIGGER records_no_delete
            BEFORE DELETE ON records
            BEGIN SELECT RAISE(ABORT, 'append-only records cannot be deleted'); END;
            CREATE TRIGGER keys_no_update
            BEFORE UPDATE ON payload_keys
            BEGIN SELECT RAISE(ABORT, 'wrapped payload keys cannot be updated'); END;
            CREATE TRIGGER destroyed_keys_no_update
            BEFORE UPDATE ON destroyed_key_events
            BEGIN SELECT RAISE(ABORT, 'destroyed-key events cannot be updated'); END;
            CREATE TRIGGER destroyed_keys_no_delete
            BEFORE DELETE ON destroyed_key_events
            BEGIN SELECT RAISE(ABORT, 'destroyed-key events cannot be deleted'); END;
            COMMIT;
            """
        )
        metadata = {
            "schema_version": self.schema_version,
            "store_id": store_id,
            "lineage_id": lineage_id,
            "lineage_kind": lineage_kind,
            "owner_principal_ref": "owner:" + secrets.token_hex(16),
            "owner_principal_commitment": hmac.new(
                self._master_key,
                b"seed-owner-principal-v0.1\x00" + owner_principal.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
            "checkpoint_domain_id": checkpoint_domain_id,
            "created_at_utc": utc_now(),
        }
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _assert_schema(self) -> None:
        try:
            version = self.metadata("schema_version")
        except sqlite3.Error as exc:
            raise ContinuityStoreError("The continuity store schema is unavailable.") from exc
        if version != self.schema_version:
            raise ContinuityStoreError(f"Unsupported continuity store version: {version!r}")

    def metadata(self, key: str) -> str:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ContinuityStoreError(f"Missing required store metadata: {key}")
        return str(row["value"])

    @property
    def store_id(self) -> str:
        return self.metadata("store_id")

    @property
    def lineage_id(self) -> str:
        return self.metadata("lineage_id")

    @property
    def owner_principal_ref(self) -> str:
        return self.metadata("owner_principal_ref")

    def owner_matches(self, principal: str) -> bool:
        candidate = hmac.new(
            self._master_key,
            b"seed-owner-principal-v0.1\x00" + principal.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(candidate, self.metadata("owner_principal_commitment"))

    @property
    def checkpoint_domain_id(self) -> str:
        return self.metadata("checkpoint_domain_id")

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "ContinuityStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _checkpoint_authenticator(self, body: dict[str, Any]) -> str:
        encoded = canonical_json(body).encode("utf-8")
        return hmac.new(
            self._master_key,
            b"seed-authority-checkpoint-v0.1\x00" + encoded,
            hashlib.sha256,
        ).hexdigest()

    def _load_checkpoint(self, *, allow_pending: bool = False) -> AuthorityCheckpoint:
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointMismatchError("The authority checkpoint is absent or malformed.") from exc
        if not isinstance(data, dict):
            raise CheckpointMismatchError("The authority checkpoint payload is malformed.")
        try:
            checkpoint = AuthorityCheckpoint(**data)
        except (TypeError, ValueError) as exc:
            raise CheckpointMismatchError("The authority checkpoint fields are invalid.") from exc
        expected = self._checkpoint_authenticator(checkpoint.body())
        if not hmac.compare_digest(checkpoint.checkpoint_authenticator, expected):
            raise CheckpointMismatchError("The authority checkpoint authenticator is invalid.")
        if checkpoint.store_id != self.store_id or checkpoint.lineage_id != self.lineage_id:
            raise CheckpointMismatchError("The checkpoint belongs to a different store or lineage.")
        if checkpoint.checkpoint_version != self.checkpoint_version:
            raise CheckpointMismatchError("The checkpoint version is unsupported.")
        if not allow_pending and checkpoint.pending_or_finalized != "finalized":
            raise CheckpointMismatchError("The authority checkpoint is pending; recovery must fail closed.")
        return checkpoint

    def _write_checkpoint(self, checkpoint: AuthorityCheckpoint) -> None:
        body = canonical_json(
            {**checkpoint.body(), "checkpoint_authenticator": checkpoint.checkpoint_authenticator}
        )
        temporary = self.checkpoint_path.with_name(
            f".{self.checkpoint_path.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.checkpoint_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _build_checkpoint(
        self,
        *,
        prior: AuthorityCheckpoint | None,
        pending_or_finalized: str,
        governance_sequence: int,
        governance_head_hash: str,
        active_lineage_head_hash: str,
        active_snapshot_root: str,
        record_chain_head_hash: str,
        expected_transaction_commitment: str,
        updated_at_utc: str,
    ) -> AuthorityCheckpoint:
        body = {
            "checkpoint_version": self.checkpoint_version,
            "store_id": self.store_id,
            "lineage_id": self.lineage_id,
            "authority_checkpoint_epoch": 1 if prior is None else prior.authority_checkpoint_epoch + 1,
            "authoritative_governance_sequence": governance_sequence,
            "authoritative_governance_head_hash": governance_head_hash,
            "active_lineage_head_hash": active_lineage_head_hash,
            "active_snapshot_root": active_snapshot_root,
            "record_chain_head_hash": record_chain_head_hash,
            "revocation_epoch": 0 if prior is None else prior.revocation_epoch,
            "erasure_epoch": 0 if prior is None else prior.erasure_epoch,
            "destroyed_key_commitment_root": self._destroyed_key_commitment_root(),
            "prior_authority_checkpoint_hash": None if prior is None else prior.checkpoint_hash,
            "pending_or_finalized": pending_or_finalized,
            "expected_transaction_commitment": expected_transaction_commitment,
            "updated_at_utc": updated_at_utc,
        }
        return AuthorityCheckpoint(
            **body,
            checkpoint_authenticator=self._checkpoint_authenticator(body),
        )

    def _destroyed_key_commitment_root(self) -> str:
        rows = self._connection.execute(
            "SELECT commitment FROM destroyed_key_events ORDER BY commitment"
        ).fetchall()
        return stable_hash([str(row["commitment"]) for row in rows])

    def _latest_row(self, record_type: str | None = None) -> sqlite3.Row | None:
        if record_type is None:
            return self._connection.execute(
                "SELECT * FROM records ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        return self._connection.execute(
            "SELECT * FROM records WHERE record_type = ? ORDER BY sequence DESC LIMIT 1",
            (record_type,),
        ).fetchone()

    def _header_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "record_type": row["record_type"],
            "record_class": row["record_class"],
            "lineage_id": row["lineage_id"],
            "session_id": row["session_id"],
            "sequence": int(row["sequence"]),
            "previous_record_hash": row["previous_record_hash"],
            "transaction_id": row["transaction_id"],
            "created_at_utc": row["created_at_utc"],
            "payload_key_id": row["payload_key_id"],
        }

    def _record_hash_from_row(self, row: sqlite3.Row) -> str:
        ciphertext = bytes(row["payload_ciphertext"])
        ciphertext_hash = hashlib.sha256(ciphertext).hexdigest()
        if not hmac.compare_digest(ciphertext_hash, str(row["ciphertext_hash"])):
            raise RecordIntegrityError("A durable payload ciphertext hash does not match.")
        return stable_hash({**self._header_from_row(row), "ciphertext_hash": ciphertext_hash})

    def read_record(self, record_hash: str) -> DurableRecord:
        self.verify_checkpoint()
        row = self._connection.execute(
            "SELECT * FROM records WHERE record_hash = ?", (record_hash,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Unknown durable record: {record_hash}")
        computed = self._record_hash_from_row(row)
        if not hmac.compare_digest(computed, str(row["record_hash"])):
            raise RecordIntegrityError("A durable record hash does not match its envelope.")
        key_row = self._connection.execute(
            "SELECT * FROM payload_keys WHERE key_id = ?", (row["payload_key_id"],)
        ).fetchone()
        if key_row is None:
            raise PayloadUnavailableError(
                "The record is retained for audit ancestry but its payload key is unavailable."
            )
        key_id = str(row["payload_key_id"])
        try:
            data_key = AESGCM(self._master_key).decrypt(
                bytes(key_row["wrap_nonce"]),
                bytes(key_row["wrapped_key"]),
                ("seed-wrap-v0.1:" + key_id).encode("utf-8"),
            )
            plaintext = AESGCM(data_key).decrypt(
                bytes(row["payload_nonce"]),
                bytes(row["payload_ciphertext"]),
                canonical_json(self._header_from_row(row)).encode("utf-8"),
            )
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise RecordIntegrityError("The durable payload failed authenticated decryption.") from exc
        if not isinstance(payload, dict):
            raise RecordIntegrityError("The durable payload is not a canonical object.")
        return DurableRecord(
            record_hash=str(row["record_hash"]),
            record_type=str(row["record_type"]),
            record_class=str(row["record_class"]),
            lineage_id=str(row["lineage_id"]),
            session_id=row["session_id"],
            sequence=int(row["sequence"]),
            previous_record_hash=row["previous_record_hash"],
            transaction_id=str(row["transaction_id"]),
            created_at_utc=str(row["created_at_utc"]),
            ciphertext_hash=str(row["ciphertext_hash"]),
            payload_key_id=key_id,
            payload=payload,
        )

    def list_record_refs(
        self,
        *,
        record_type: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, ...]:
        self.verify_checkpoint()
        clauses: list[str] = []
        values: list[Any] = []
        if record_type is not None:
            clauses.append("record_type = ?")
            values.append(record_type)
        if session_id is not None:
            clauses.append("session_id = ?")
            values.append(session_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._connection.execute(
            f"SELECT record_hash FROM records{where} ORDER BY sequence", values
        ).fetchall()
        return tuple(str(row["record_hash"]) for row in rows)

    def latest_record(self, record_type: str) -> DurableRecord:
        row = self._latest_row(record_type)
        if row is None:
            raise RecordNotFoundError(f"No {record_type} record exists.")
        return self.read_record(str(row["record_hash"]))

    def latest_head(self) -> DurableRecord:
        return self.latest_record("LineageHeadRecord")

    def current_snapshot(self) -> DurableRecord:
        head = self.latest_head()
        return self.read_record(str(head.payload["snapshot_ref"]))

    def current_policy(self) -> DurableRecord:
        return self.latest_record("LineagePolicyRevisionRecord")

    def verify_chain(self) -> dict[str, Any]:
        self.verify_checkpoint()
        rows = self._connection.execute("SELECT * FROM records ORDER BY sequence").fetchall()
        previous: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            if int(row["sequence"]) != expected_sequence:
                raise RecordIntegrityError("The append-only record sequence is not contiguous.")
            if row["previous_record_hash"] != previous:
                raise RecordIntegrityError("The append-only record ancestry is broken.")
            computed = self._record_hash_from_row(row)
            if not hmac.compare_digest(computed, str(row["record_hash"])):
                raise RecordIntegrityError("An append-only record envelope was altered.")
            previous = str(row["record_hash"])
        return {
            "record_count": len(rows),
            "record_chain_head_hash": previous,
            "chain_valid": True,
        }

    def verify_checkpoint(self) -> AuthorityCheckpoint:
        checkpoint = self._load_checkpoint()
        last = self._latest_row()
        governance = self._latest_row("GovernanceTransitionRecord")
        head = self._latest_row("LineageHeadRecord")
        if last is None or governance is None or head is None:
            raise CheckpointMismatchError("The initialized store is missing required state records.")
        head_record = self._read_record_without_checkpoint(str(head["record_hash"]))
        governance_record = self._read_record_without_checkpoint(str(governance["record_hash"]))
        expected = {
            "record_chain_head_hash": str(last["record_hash"]),
            "authoritative_governance_head_hash": str(governance["record_hash"]),
            "active_lineage_head_hash": str(head["record_hash"]),
            "active_snapshot_root": str(head_record.payload["snapshot_ref"]),
            "authoritative_governance_sequence": int(
                governance_record.payload["governance_sequence"]
            ),
            "destroyed_key_commitment_root": self._destroyed_key_commitment_root(),
        }
        for field, value in expected.items():
            if getattr(checkpoint, field) != value:
                raise CheckpointMismatchError(
                    f"The content store does not match checkpoint field {field}."
                )
        return checkpoint

    def _read_record_without_checkpoint(self, record_hash: str) -> DurableRecord:
        row = self._connection.execute(
            "SELECT * FROM records WHERE record_hash = ?", (record_hash,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(record_hash)
        computed = self._record_hash_from_row(row)
        if not hmac.compare_digest(computed, str(row["record_hash"])):
            raise RecordIntegrityError("A durable record envelope was altered.")
        key_row = self._connection.execute(
            "SELECT * FROM payload_keys WHERE key_id = ?", (row["payload_key_id"],)
        ).fetchone()
        if key_row is None:
            raise PayloadUnavailableError("The required payload key is unavailable.")
        key_id = str(row["payload_key_id"])
        try:
            data_key = AESGCM(self._master_key).decrypt(
                bytes(key_row["wrap_nonce"]),
                bytes(key_row["wrapped_key"]),
                ("seed-wrap-v0.1:" + key_id).encode("utf-8"),
            )
            plaintext = AESGCM(data_key).decrypt(
                bytes(row["payload_nonce"]),
                bytes(row["payload_ciphertext"]),
                canonical_json(self._header_from_row(row)).encode("utf-8"),
            )
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise RecordIntegrityError("The durable payload failed authenticated decryption.") from exc
        return DurableRecord(
            record_hash=str(row["record_hash"]),
            record_type=str(row["record_type"]),
            record_class=str(row["record_class"]),
            lineage_id=str(row["lineage_id"]),
            session_id=row["session_id"],
            sequence=int(row["sequence"]),
            previous_record_hash=row["previous_record_hash"],
            transaction_id=str(row["transaction_id"]),
            created_at_utc=str(row["created_at_utc"]),
            ciphertext_hash=str(row["ciphertext_hash"]),
            payload_key_id=key_id,
            payload=payload,
        )

    def keyed_commitment(self, value: Any) -> str:
        encoded = canonical_json(value).encode("utf-8")
        return hmac.new(
            self._master_key,
            b"seed-nonpublic-commitment-v0.1\x00" + encoded,
            hashlib.sha256,
        ).hexdigest()

    def write_batch(self, *, bootstrap: bool = False) -> "WriteBatch":
        if self._closed:
            raise ContinuityStoreError("The continuity store is closed.")
        return WriteBatch(self, bootstrap=bootstrap)


class WriteBatch(AbstractContextManager["WriteBatch"]):
    """One SQLite transaction plus checkpoint prepare/finalize boundary."""

    def __init__(self, store: ContinuityStore, *, bootstrap: bool) -> None:
        self.store = store
        self.bootstrap = bootstrap
        self.transaction_id = secrets.token_hex(16)
        self._prior_checkpoint: AuthorityCheckpoint | None = None
        self._record_hashes: list[str] = []
        self._record_types: list[str] = []
        self._entered = False
        self._checkpoint_prepared = False

    @property
    def checkpoint_epoch_before(self) -> int:
        return 0 if self._prior_checkpoint is None else self._prior_checkpoint.authority_checkpoint_epoch

    @property
    def checkpoint_epoch_after(self) -> int:
        return self.checkpoint_epoch_before + 1

    @property
    def governance_sequence_before(self) -> int:
        return 0 if self._prior_checkpoint is None else self._prior_checkpoint.authoritative_governance_sequence

    @property
    def governance_sequence_after(self) -> int:
        return self.governance_sequence_before + 1

    @property
    def erasure_epoch_before(self) -> int:
        return 0 if self._prior_checkpoint is None else self._prior_checkpoint.erasure_epoch

    @property
    def revocation_epoch_before(self) -> int:
        return 0 if self._prior_checkpoint is None else self._prior_checkpoint.revocation_epoch

    @property
    def prior_checkpoint_hash(self) -> str | None:
        return None if self._prior_checkpoint is None else self._prior_checkpoint.checkpoint_hash

    @property
    def prior_governance_head_hash(self) -> str | None:
        return (
            None
            if self._prior_checkpoint is None
            else self._prior_checkpoint.authoritative_governance_head_hash
        )

    def __enter__(self) -> "WriteBatch":
        self.store._lock.acquire()
        try:
            if self.bootstrap:
                if self.store.checkpoint_path.exists() or self.store._latest_row() is not None:
                    raise ContinuityStoreError("Bootstrap requires an empty uncheckpointed store.")
            else:
                self._prior_checkpoint = self.store.verify_checkpoint()
            self.store._connection.execute("BEGIN IMMEDIATE")
            self._entered = True
            return self
        except Exception:
            self.store._lock.release()
            raise

    def append(
        self,
        record_type: str,
        record_class: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> str:
        if not self._entered:
            raise ContinuityStoreError("The write batch is not active.")
        if record_class not in {"G", "E", "S", "R", "C", "X"}:
            raise ValueError("Unknown record class.")
        if not record_type.strip() or not isinstance(payload, dict):
            raise ValueError("A durable record requires a type and object payload.")
        last = self.store._latest_row()
        sequence = 1 if last is None else int(last["sequence"]) + 1
        previous = None if last is None else str(last["record_hash"])
        created_at = utc_now()
        key_id = secrets.token_hex(16)
        header = {
            "record_type": record_type,
            "record_class": record_class,
            "lineage_id": self.store.lineage_id,
            "session_id": session_id,
            "sequence": sequence,
            "previous_record_hash": previous,
            "transaction_id": self.transaction_id,
            "created_at_utc": created_at,
            "payload_key_id": key_id,
        }
        data_key = AESGCM.generate_key(bit_length=256)
        payload_nonce = os.urandom(12)
        payload_ciphertext = AESGCM(data_key).encrypt(
            payload_nonce,
            canonical_json(payload).encode("utf-8"),
            canonical_json(header).encode("utf-8"),
        )
        ciphertext_hash = hashlib.sha256(payload_ciphertext).hexdigest()
        record_hash = stable_hash({**header, "ciphertext_hash": ciphertext_hash})
        wrap_nonce = os.urandom(12)
        wrapped_key = AESGCM(self.store._master_key).encrypt(
            wrap_nonce,
            data_key,
            ("seed-wrap-v0.1:" + key_id).encode("utf-8"),
        )
        self.store._connection.execute(
            "INSERT INTO payload_keys(key_id, wrap_nonce, wrapped_key, created_at_utc) "
            "VALUES (?, ?, ?, ?)",
            (key_id, wrap_nonce, wrapped_key, created_at),
        )
        self.store._connection.execute(
            """
            INSERT INTO records(
                record_hash, record_type, record_class, lineage_id, session_id,
                sequence, previous_record_hash, transaction_id, created_at_utc,
                ciphertext_hash, payload_key_id, payload_nonce, payload_ciphertext
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_hash,
                record_type,
                record_class,
                self.store.lineage_id,
                session_id,
                sequence,
                previous,
                self.transaction_id,
                created_at,
                ciphertext_hash,
                key_id,
                payload_nonce,
                payload_ciphertext,
            ),
        )
        self._record_hashes.append(record_hash)
        self._record_types.append(record_type)
        return record_hash

    def erase_payload_key(self, record_hash: str) -> str:
        row = self.store._connection.execute(
            "SELECT payload_key_id FROM records WHERE record_hash = ?", (record_hash,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(record_hash)
        key_id = str(row["payload_key_id"])
        deleted = self.store._connection.execute(
            "DELETE FROM payload_keys WHERE key_id = ?", (key_id,)
        ).rowcount
        if deleted != 1:
            raise PayloadUnavailableError("The payload key was already unavailable.")
        commitment = stable_hash({"destroyed_key_id": key_id, "record_hash": record_hash})
        self.store._connection.execute(
            "INSERT INTO destroyed_key_events(commitment, record_hash, transaction_id, destroyed_at_utc) "
            "VALUES (?, ?, ?, ?)",
            (commitment, record_hash, self.transaction_id, utc_now()),
        )
        return commitment

    def _checkpoint_state(self) -> tuple[int, str, str, str, str]:
        governance = self.store._latest_row("GovernanceTransitionRecord")
        head = self.store._latest_row("LineageHeadRecord")
        last = self.store._latest_row()
        if governance is None or head is None or last is None:
            raise ContinuityStoreError("A batch must leave governance, head, and chain state.")
        governance_record = self.store._read_record_without_checkpoint(str(governance["record_hash"]))
        head_record = self.store._read_record_without_checkpoint(str(head["record_hash"]))
        return (
            int(governance_record.payload["governance_sequence"]),
            str(governance["record_hash"]),
            str(head["record_hash"]),
            str(head_record.payload["snapshot_ref"]),
            str(last["record_hash"]),
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:  # type: ignore[no-untyped-def]
        try:
            if exc_type is not None:
                self.store._connection.rollback()
                return False
            if not self._record_hashes:
                raise ContinuityStoreError("An empty durable transaction is not allowed.")
            if self._record_types[-1] != "GovernanceTransitionRecord":
                raise ContinuityStoreError(
                    "Every durable transaction must end in a GovernanceTransitionRecord."
                )
            (
                governance_sequence,
                governance_head,
                active_head,
                active_snapshot,
                record_chain_head,
            ) = self._checkpoint_state()
            if governance_sequence != self.governance_sequence_after:
                raise ContinuityStoreError("The governance sequence did not advance exactly once.")
            transaction_commitment = stable_hash(
                {
                    "transaction_id": self.transaction_id,
                    "record_hashes": self._record_hashes,
                    "governance_head": governance_head,
                    "active_head": active_head,
                    "active_snapshot": active_snapshot,
                }
            )
            timestamp = utc_now()
            pending = self.store._build_checkpoint(
                prior=self._prior_checkpoint,
                pending_or_finalized="pending",
                governance_sequence=governance_sequence,
                governance_head_hash=governance_head,
                active_lineage_head_hash=active_head,
                active_snapshot_root=active_snapshot,
                record_chain_head_hash=record_chain_head,
                expected_transaction_commitment=transaction_commitment,
                updated_at_utc=timestamp,
            )
            self.store._write_checkpoint(pending)
            self._checkpoint_prepared = True
            self.store._connection.commit()
            finalized = self.store._build_checkpoint(
                prior=self._prior_checkpoint,
                pending_or_finalized="finalized",
                governance_sequence=governance_sequence,
                governance_head_hash=governance_head,
                active_lineage_head_hash=active_head,
                active_snapshot_root=active_snapshot,
                record_chain_head_hash=record_chain_head,
                expected_transaction_commitment=transaction_commitment,
                updated_at_utc=timestamp,
            )
            self.store._write_checkpoint(finalized)
            return False
        except Exception:
            if self.store._connection.in_transaction:
                self.store._connection.rollback()
            # A prepared checkpoint is intentionally left pending after an
            # uncertain commit boundary.  Reopen therefore fails closed.
            raise
        finally:
            self._entered = False
            self.store._lock.release()
