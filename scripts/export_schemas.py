"""Export the canonical schemas as published JSON Schema contracts.

    python scripts/export_schemas.py

The registry in :mod:`src.core.schema` is the single source of truth. These
files are generated from it rather than maintained by hand, so a contract can
never drift from the code that enforces it - which is the usual failure mode of
hand-written data contracts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import project_root
from src.core.schema import SCHEMAS


def main() -> int:
    target = project_root() / "schemas"
    target.mkdir(parents=True, exist_ok=True)

    written = []
    for name, schema in SCHEMAS.items():
        path = target / f"{name}_schema.json"
        path.write_text(json.dumps(schema.to_contract(), indent=2), encoding="utf-8")
        written.append((name, path.name, len(schema.fields), len(schema.required_columns)))

    index = {
        "contracts": [
            {
                "dataset": name,
                "file": f"{name}_schema.json",
                "version": schema.schema_version,
                "owner": schema.owner,
                "classification": schema.classification,
                "business_key": list(schema.business_key),
                "fields": len(schema.fields),
                "required_fields": len(schema.required_columns),
                "foreign_keys": [
                    {
                        "column": fk.column,
                        "references": f"{fk.references_dataset}.{fk.references_column}",
                        "blocking": fk.blocking,
                    }
                    for fk in schema.foreign_keys
                ],
            }
            for name, schema in SCHEMAS.items()
        ],
        "note": (
            "Generated from src/core/schema.py by scripts/export_schemas.py. "
            "Edit the registry, not these files."
        ),
    }
    (target / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"Wrote {len(written)} contracts to {target}")
    print(f"  {'dataset':<20}{'file':<30}{'fields':>8}{'required':>10}")
    for name, filename, fields, required in written:
        print(f"  {name:<20}{filename:<30}{fields:>8}{required:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
