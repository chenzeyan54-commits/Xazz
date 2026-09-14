//! xazz-server/src/store.rs — SQLite run-history persistence (issue C1)
//!
//! Each pipeline execution (`POST /execute`) is recorded in a `runs` table so
//! run history survives a server restart and is queryable via the HTTP API.
//!
//!   GET  /runs        → list runs (id, status, rows, code_hash, created_at)
//!   GET  /runs/:id    → single run record (including stored error)
//!
//! Storage lives in `xazz.db` in the server's working directory. The DB is
//! opened lazily and the table is created if missing.
//!
//! Multi-tenant (issue C2): every run is tagged with a `tenant`; when a request
//! is authenticated with `X-Xazz-Tenant`, the store only sees/returns that
//! tenant's rows. The `""` tenant is the default (single-tenant / local).

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

/// SQLite database file (relative to the server's working directory).
pub const DB_FILE: &str = "xazz.db";

/// Current Unix epoch seconds.
fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Creates the run-history and per-tenant DP-budget tables if missing (idempotent).
fn ensure_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at INTEGER NOT NULL,
            tenant TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS dp_budget (
            tenant TEXT PRIMARY KEY,
            spent_epsilon REAL NOT NULL DEFAULT 0,
            spent_delta REAL NOT NULL DEFAULT 0,
            window_started_at INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS tenant_policies (
            tenant TEXT PRIMARY KEY,
            policy_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL DEFAULT 0
        );",
    )
    .map_err(|e| format!("failed to create store schema: {e}"))?;
    // Migration for DBs created before the tenant column (issue C2):
    // adding an already-present column is a no-op error we swallow.
    let _ = conn.execute_batch("ALTER TABLE runs ADD COLUMN tenant TEXT NOT NULL DEFAULT ''");
    // Migration for DBs created before budget windows (issue C2):
    let _ = conn.execute_batch(
        "ALTER TABLE dp_budget ADD COLUMN window_started_at INTEGER NOT NULL DEFAULT 0",
    );
    Ok(())
}

/// One persisted run record.
#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    pub id: i64,
    pub code_hash: String,
    pub status: String,
    pub rows: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Tenant tag (empty string = default/single-tenant) — issue C2
    #[serde(skip_serializing_if = "String::is_empty")]
    pub tenant: String,
    /// Unix epoch seconds
    pub created_at: i64,
}

/// Holds the lazily-opened SQLite connection.
pub struct Store {
    conn: Mutex<Option<Connection>>,
}

impl Store {
    pub fn new() -> Self {
        Store {
            conn: Mutex::new(None),
        }
    }

    /// Builds a store pinned to an explicit DB path (schema created eagerly).
    /// Used by tests to isolate runs from the real `xazz.db`.
    #[allow(dead_code)]
    pub fn open_at(path: &std::path::Path) -> Self {
        let conn = Connection::open(path).expect("open store db");
        ensure_schema(&conn).expect("create store schema");
        Store {
            conn: Mutex::new(Some(conn)),
        }
    }

    /// Opens the DB and creates the schema if needed. Called on first write.
    fn open(&self) -> Result<std::sync::MutexGuard<'_, Option<Connection>>, String> {
        let mut guard = self
            .conn
            .lock()
            .map_err(|_| "store lock poisoned".to_string())?;
        if guard.is_none() {
            let conn = Connection::open(PathBuf::from(DB_FILE))
                .map_err(|e| format!("failed to open {}: {e}", DB_FILE))?;
            ensure_schema(&conn)?;
            *guard = Some(conn);
        }
        Ok(guard)
    }

    /// Records a run and returns its id.
    pub fn record_run(
        &self,
        code_hash: &str,
        status: &str,
        rows: i64,
        error: Option<&str>,
        tenant: &str,
    ) -> Result<i64, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let created = now_epoch();
        conn.execute(
            "INSERT INTO runs (code_hash, status, rows, error, created_at, tenant) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![code_hash, status, rows, error, created, tenant],
        )
        .map_err(|e| format!("failed to insert run: {e}"))?;
        Ok(conn.last_insert_rowid())
    }

    /// Lists runs newest-first, filtered to a tenant.
    pub fn list_runs(&self, limit: usize, tenant: &str) -> Result<Vec<RunRecord>, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let mut stmt = conn
            .prepare("SELECT id, code_hash, status, rows, error, created_at, tenant FROM runs WHERE tenant = ?1 ORDER BY id DESC LIMIT ?2")
            .map_err(|e| format!("failed to prepare list: {e}"))?;
        let rows = stmt
            .query_map(params![tenant, limit as i64], |row| {
                Ok(RunRecord {
                    id: row.get(0)?,
                    code_hash: row.get(1)?,
                    status: row.get(2)?,
                    rows: row.get(3)?,
                    error: row.get(4)?,
                    created_at: row.get(5)?,
                    tenant: row.get(6)?,
                })
            })
            .map_err(|e| format!("failed to query runs: {e}"))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| format!("failed to read run row: {e}"))?);
        }
        Ok(out)
    }

    /// Fetches a single run by id, scoped to a tenant.
    pub fn get_run(&self, id: i64, tenant: &str) -> Result<Option<RunRecord>, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let mut stmt = conn
            .prepare("SELECT id, code_hash, status, rows, error, created_at, tenant FROM runs WHERE id = ?1 AND tenant = ?2")
            .map_err(|e| format!("failed to prepare get: {e}"))?;
        let mut rows = stmt
            .query_map(params![id, tenant], |row| {
                Ok(RunRecord {
                    id: row.get(0)?,
                    code_hash: row.get(1)?,
                    status: row.get(2)?,
                    rows: row.get(3)?,
                    error: row.get(4)?,
                    created_at: row.get(5)?,
                    tenant: row.get(6)?,
                })
            })
            .map_err(|e| format!("failed to query run: {e}"))?;
        match rows.next() {
            Some(r) => r.map_err(|e| format!("failed to read run: {e}")).map(Some),
            None => Ok(None),
        }
    }

    /// Returns the tenant's cumulative DP spend as `(epsilon, delta)` — issue C2.
    ///
    /// Unknown tenants report `(0.0, 0.0)` (no budget consumed yet). The ledger is
    /// keyed by tenant, so one tenant's spend never affects another's.
    ///
    /// When `window_secs > 0` the budget is a *sliding window*: once the window
    /// anchored at `window_started_at` has elapsed, the spend is reset to zero
    /// (the roll is persisted so later accruals start a fresh window).
    /// `window_secs == 0` keeps the legacy cumulative behavior.
    pub fn dp_spent(&self, tenant: &str, window_secs: u64) -> Result<(f64, f64), String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        roll_dp_window(conn, tenant, window_secs, now_epoch())?;
        conn.query_row(
            "SELECT spent_epsilon, spent_delta FROM dp_budget WHERE tenant = ?1",
            params![tenant],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, f64>(1)?)),
        )
        .optional()
        .map(|opt| opt.unwrap_or((0.0, 0.0)))
        .map_err(|e| format!("failed to read dp budget: {e}"))
    }

    /// Returns the tenant's current window anchor (Unix epoch seconds), if a row exists.
    pub fn dp_window_started_at(&self, tenant: &str) -> Result<Option<i64>, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        conn.query_row(
            "SELECT window_started_at FROM dp_budget WHERE tenant = ?1",
            params![tenant],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|e| format!("failed to read dp window: {e}"))
    }

    /// Adds a run's DP spend to the tenant's cumulative ledger — issue C2.
    ///
    /// The increment is a single atomic UPSERT, so two concurrent runs of the same
    /// tenant cannot lose a spend. Non-positive spends are a no-op. Non-finite
    /// values are rejected so a malformed marker cannot poison the ledger.
    ///
    /// `window_secs` behaves exactly as in [`Store::dp_spent`]: an elapsed window is
    /// rolled (zeroed) before the increment is applied.
    pub fn add_dp_spend(
        &self,
        tenant: &str,
        epsilon: f64,
        delta: f64,
        window_secs: u64,
    ) -> Result<(), String> {
        if !epsilon.is_finite() || !delta.is_finite() {
            return Err("dp spend must be finite".to_string());
        }
        let epsilon = epsilon.max(0.0);
        let delta = delta.max(0.0);
        if epsilon == 0.0 && delta == 0.0 {
            return Ok(());
        }
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let now = now_epoch();
        roll_dp_window(conn, tenant, window_secs, now)?;
        conn.execute(
            "INSERT INTO dp_budget (tenant, spent_epsilon, spent_delta, window_started_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(tenant) DO UPDATE SET
                 spent_epsilon = spent_epsilon + excluded.spent_epsilon,
                 spent_delta   = spent_delta   + excluded.spent_delta,
                 updated_at    = excluded.updated_at",
            params![tenant, epsilon, delta, now],
        )
        .map_err(|e| format!("failed to update dp budget: {e}"))?;
        Ok(())
    }

    /// Resets a tenant's accumulated DP spend to zero and restarts its window — issue C2.
    ///
    /// Tenant-scoped: other tenants' ledgers are untouched. A zeroed row is kept
    /// (rather than deleted) and re-anchored to now so a fresh window starts immediately.
    pub fn reset_dp_budget(&self, tenant: &str) -> Result<(), String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let now = now_epoch();
        conn.execute(
            "INSERT INTO dp_budget (tenant, spent_epsilon, spent_delta, window_started_at, updated_at)
             VALUES (?1, 0, 0, ?2, ?2)
             ON CONFLICT(tenant) DO UPDATE SET
                 spent_epsilon     = 0,
                 spent_delta       = 0,
                 window_started_at = ?2,
                 updated_at        = ?2",
            params![tenant, now],
        )
        .map_err(|e| format!("failed to reset dp budget: {e}"))?;
        Ok(())
    }

    /// Returns the tenant's stored policy pack JSON, if any — issue C2.
    ///
    /// The pack is keyed by tenant, so one tenant's policy pack is never applied to
    /// another. The caller is responsible for parsing/validating it (fail-closed).
    pub fn get_tenant_policy(&self, tenant: &str) -> Result<Option<String>, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        conn.query_row(
            "SELECT policy_json FROM tenant_policies WHERE tenant = ?1",
            params![tenant],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| format!("failed to read tenant policy: {e}"))
    }

    /// Stores (or replaces) a tenant's policy pack JSON — issue C2.
    ///
    /// Validation happens before the write (the caller parses with
    /// `Policy::from_json_str`), so a stored pack is always well-formed.
    pub fn set_tenant_policy(&self, tenant: &str, policy_json: &str) -> Result<(), String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        conn.execute(
            "INSERT INTO tenant_policies (tenant, policy_json, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(tenant) DO UPDATE SET
                 policy_json = excluded.policy_json,
                 updated_at  = excluded.updated_at",
            params![tenant, policy_json, now_epoch()],
        )
        .map_err(|e| format!("failed to store tenant policy: {e}"))?;
        Ok(())
    }

    /// Deletes a tenant's stored policy pack. Returns `true` if one existed — issue C2.
    pub fn delete_tenant_policy(&self, tenant: &str) -> Result<bool, String> {
        let guard = self.open()?;
        let conn = guard.as_ref().expect("open guarantees Some");
        let deleted = conn
            .execute(
                "DELETE FROM tenant_policies WHERE tenant = ?1",
                params![tenant],
            )
            .map_err(|e| format!("failed to delete tenant policy: {e}"))?;
        Ok(deleted > 0)
    }
}

impl Default for Store {
    fn default() -> Self {
        Store::new()
    }
}

/// Rolls a tenant's budget window when it has elapsed (no-op when disabled).
///
/// A row with `window_started_at = 0` is treated as "window not yet anchored" and
/// is re-anchored to `now`. The roll zeroes both ε and δ.
fn roll_dp_window(
    conn: &Connection,
    tenant: &str,
    window_secs: u64,
    now: i64,
) -> Result<(), String> {
    if window_secs == 0 {
        return Ok(());
    }
    conn.execute(
        "UPDATE dp_budget
         SET spent_epsilon = 0,
             spent_delta = 0,
             window_started_at = ?1,
             updated_at = ?1
         WHERE tenant = ?2
           AND (window_started_at = 0 OR ?1 - window_started_at >= ?3)",
        params![now, tenant, window_secs as i64],
    )
    .map_err(|e| format!("failed to roll dp window: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_lists_runs() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);

        let id = store
            .record_run("hash1", "success", 42, None, "tenant-a")
            .expect("insert");
        let id2 = store
            .record_run("hash2", "failed", 0, Some("boom"), "tenant-a")
            .expect("insert");
        // A different tenant's run must not be visible to tenant-a.
        let id_other = store
            .record_run("hash3", "success", 1, None, "tenant-b")
            .expect("insert");

        let list = store.list_runs(10, "tenant-a").expect("list");
        assert_eq!(list.len(), 2, "{list:?}");
        assert!(list.iter().all(|r| r.tenant == "tenant-a"));
        let first = &list[0];
        assert!(first.id == id2 || first.id == id);
        // Newest first.
        assert_eq!(first.id.max(id2), first.id);

        let got = store.get_run(id, "tenant-a").expect("get").expect("exists");
        assert_eq!(got.status, "success");
        assert_eq!(got.rows, 42);
        assert!(got.error.is_none());

        let got2 = store
            .get_run(id2, "tenant-a")
            .expect("get")
            .expect("exists");
        assert_eq!(got2.status, "failed");
        assert_eq!(got2.error.as_deref(), Some("boom"));

        // Cross-tenant access is denied: tenant-a cannot read tenant-b's run.
        assert!(
            store
                .get_run(id_other, "tenant-a")
                .expect("no err")
                .is_none()
        );
        // tenant-b sees only its own run.
        let b_list = store.list_runs(10, "tenant-b").expect("list");
        assert_eq!(b_list.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_run_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_miss_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);
        assert!(store.get_run(999_999, "").expect("no err").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// DP budget accrues cumulatively and is isolated per tenant (issue C2).
    #[test]
    fn dp_budget_accrues_and_is_tenant_scoped() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_dp_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);

        // Unknown tenant starts at zero.
        assert_eq!(store.dp_spent("a", 0).expect("read"), (0.0, 0.0));

        store.add_dp_spend("a", 1.5, 0.0, 0).expect("spend 1");
        store.add_dp_spend("a", 0.5, 1e-5, 0).expect("spend 2");
        let (eps, delta) = store.dp_spent("a", 0).expect("read");
        assert!((eps - 2.0).abs() < 1e-12, "eps={eps}");
        assert!((delta - 1e-5).abs() < 1e-15, "delta={delta}");

        // Tenant isolation: b is untouched by a's spend.
        assert_eq!(store.dp_spent("b", 0).expect("read"), (0.0, 0.0));

        // A zero-spend update is a no-op; non-finite values are rejected.
        store.add_dp_spend("a", 0.0, 0.0, 0).expect("no-op");
        assert!((store.dp_spent("a", 0).expect("read").0 - 2.0).abs() < 1e-12);
        assert!(store.add_dp_spend("a", f64::NAN, 0.0, 0).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tenant can reset its accumulated spend; the reset is tenant-scoped (issue C2).
    #[test]
    fn dp_budget_reset_is_tenant_scoped() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_reset_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);

        store.add_dp_spend("a", 3.0, 1e-5, 0).expect("spend a");
        store.add_dp_spend("b", 1.0, 0.0, 0).expect("spend b");

        store.reset_dp_budget("a").expect("reset a");
        assert_eq!(store.dp_spent("a", 0).expect("read a"), (0.0, 0.0));
        // b is unaffected.
        assert!((store.dp_spent("b", 0).expect("read b").0 - 1.0).abs() < 1e-12);

        // Reset also re-anchors the window.
        assert!(
            store
                .dp_window_started_at("a")
                .expect("anchor")
                .unwrap_or(0)
                > 0
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An elapsed budget window rolls the spend to zero; window 0 keeps cumulative (issue C2).
    #[test]
    fn dp_budget_window_rolls_over_when_elapsed() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_window_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);

        store.add_dp_spend("a", 2.0, 0.0, 3600).expect("spend");
        assert!((store.dp_spent("a", 3600).expect("read").0 - 2.0).abs() < 1e-12);

        // Age the window anchor by two hours so the one-hour window has elapsed.
        {
            let conn = Connection::open(&db).expect("open raw");
            conn.execute(
                "UPDATE dp_budget SET window_started_at = window_started_at - 7200 WHERE tenant = 'a'",
                [],
            )
            .expect("age window");
        }
        assert_eq!(
            store.dp_spent("a", 3600).expect("read after roll"),
            (0.0, 0.0)
        );

        // Window disabled (0) is cumulative across calls.
        store.add_dp_spend("a", 1.0, 0.0, 0).expect("spend again");
        assert!((store.dp_spent("a", 0).expect("read cumulative").0 - 1.0).abs() < 1e-12);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Tenant policy packs are stored and isolated per tenant (issue C2).
    #[test]
    fn tenant_policies_are_namespaced_per_tenant() {
        let dir = std::env::temp_dir().join(format!(
            "xazz_store_pol_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db = dir.join("xazz.db");
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open_at(&db);

        // Unknown tenants have no stored pack.
        assert!(store.get_tenant_policy("a").expect("read").is_none());

        store
            .set_tenant_policy("a", r#"{"id":"a-pack"}"#)
            .expect("write a");
        store
            .set_tenant_policy("b", r#"{"id":"b-pack"}"#)
            .expect("write b");

        // Each tenant reads only its own pack.
        assert_eq!(
            store.get_tenant_policy("a").expect("read a").as_deref(),
            Some(r#"{"id":"a-pack"}"#)
        );
        assert_eq!(
            store.get_tenant_policy("b").expect("read b").as_deref(),
            Some(r#"{"id":"b-pack"}"#)
        );

        // Replacing a pack only affects that tenant.
        store
            .set_tenant_policy("a", r#"{"id":"a-v2"}"#)
            .expect("rewrite a");
        assert_eq!(
            store.get_tenant_policy("a").expect("reread a").as_deref(),
            Some(r#"{"id":"a-v2"}"#)
        );
        assert_eq!(
            store.get_tenant_policy("b").expect("reread b").as_deref(),
            Some(r#"{"id":"b-pack"}"#)
        );

        // Deleting a pack is tenant-scoped.
        assert!(store.delete_tenant_policy("a").expect("delete a"));
        assert!(store.get_tenant_policy("a").expect("read a").is_none());
        assert!(store.get_tenant_policy("b").expect("read b").is_some());
        assert!(!store.delete_tenant_policy("a").expect("delete again"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
