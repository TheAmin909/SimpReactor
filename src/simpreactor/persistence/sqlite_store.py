"""SQLite persistence helpers for SimpReactor."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_utc TEXT,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS species (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  name TEXT,
  formula TEXT,
  phase TEXT,
  UNIQUE(project_id, name)
);
CREATE TABLE IF NOT EXISTS reaction (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  name TEXT,
  reversible INTEGER DEFAULT 0,
  stoich JSON
);
CREATE TABLE IF NOT EXISTS kinetic_model (
  id INTEGER PRIMARY KEY,
  reaction_id INTEGER,
  type TEXT,
  params JSON,
  covariance JSON,
  units JSON,
  source TEXT
);
CREATE TABLE IF NOT EXISTS experiment (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  kind TEXT,
  conditions JSON,
  data BLOB
);
CREATE TABLE IF NOT EXISTS run (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  model JSON,
  solver JSON,
  manifest JSON,
  started TEXT,
  duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS profile (
  run_id INTEGER,
  x REAL,
  var TEXT,
  value REAL,
  unit TEXT,
  PRIMARY KEY (run_id, x, var)
);
CREATE TABLE IF NOT EXISTS summary (
  run_id INTEGER PRIMARY KEY,
  conversion JSON,
  selectivity JSON,
  hotspots JSON
);
"""


def connect(project_file: str | Path) -> sqlite3.Connection:
    """Open (and create) a .crdproj SQLite project."""
    path = Path(project_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Ensure the SPEC-1 schema exists in the project file."""
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def create_project(
    connection: sqlite3.Connection,
    name: str,
    notes: str | None = None,
    created_utc: str | None = None,
) -> int:
    """Create a project entry and return its ID."""
    created_utc = created_utc or _utc_now()
    cursor = connection.execute(
        "INSERT INTO project (name, created_utc, notes) VALUES (?, ?, ?)",
        (name, created_utc, notes),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_run(
    connection: sqlite3.Connection,
    project_id: int,
    model: Mapping[str, object],
    solver: Mapping[str, object],
    manifest: Mapping[str, object],
    started_utc: str | None = None,
    duration_ms: int | None = None,
) -> int:
    """Persist a run record and return its ID."""
    started_utc = started_utc or _utc_now()
    cursor = connection.execute(
        "INSERT INTO run (project_id, model, solver, manifest, started, duration_ms)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            project_id,
            _json_dumps(model),
            _json_dumps(solver),
            _json_dumps(manifest),
            started_utc,
            duration_ms,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_profile(
    connection: sqlite3.Connection,
    run_id: int,
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    units: Mapping[str, str | None] | None = None,
) -> None:
    """Save normalized profile data for a run."""
    units = units or {}
    rows_list: list[tuple[object, ...]] = []
    for index, x_value in enumerate(x_values):
        for variable, values in series.items():
            rows_list.append(
                (run_id, float(x_value), variable, float(values[index]), units.get(variable)),
            )
    connection.executemany(
        "INSERT INTO profile (run_id, x, var, value, unit) VALUES (?, ?, ?, ?, ?)",
        rows_list,
    )
    connection.commit()


def _json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
