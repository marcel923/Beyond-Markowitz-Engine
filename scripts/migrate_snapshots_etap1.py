"""
scripts/migrate_snapshots_etap1.py
====================================
One-time migration: the old "saved_portfolios.json" (a single JSON array
holding every snapshot ever saved) -> the new storage/portfolio_snapshots/
layout (one JSON file per snapshot), per the Etap 1 storage migration
(PROJECT_CONTEXT.md).

Per an explicit instruction from the project owner (2026-09-06 conversation):
only the snapshot named "Master Portfolio Forward" (the 2026-09-01 / 31-ticker
import) should carry over -- everything else in the old file was test data
("Test 123", "Test (random inputs)", "smiec test") and should NOT clutter
the new storage. The NAME_FILTER below encodes exactly that instruction; if
you want to migrate everything instead, pass --keep-all.

Original snapshot_id / created_at are PRESERVED exactly as they were in the
old file, not re-stamped to "now" -- forward-tracking's holding-day count
depends on the real historical creation date, so re-stamping would silently
corrupt that (a snapshot that's actually 5 days old would suddenly show as
0 days old).

SAFE BY DEFAULT: running this script with no arguments only PRINTS what it
would do (which snapshots match, where each new file would be written) and
writes NOTHING. Pass --apply to actually write the new files. The old
saved_portfolios.json is NEVER modified or deleted by this script, under
any flag -- it stays exactly as-is as a backup; delete it yourself once
you've confirmed the new storage works.

Usage
-----
    python3 scripts/migrate_snapshots_etap1.py                  # dry run (default)
    python3 scripts/migrate_snapshots_etap1.py --apply           # actually write
    python3 scripts/migrate_snapshots_etap1.py --keep-all --apply    # migrate everything, no name filter
    python3 scripts/migrate_snapshots_etap1.py --legacy-path other_file.json --apply
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_store import DEFAULT_STORAGE_DIR, _snapshot_file_path  # noqa: E402

NAME_FILTER = "master portfolio forward"  # case-insensitive substring match on snapshot_name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-path", default="saved_portfolios.json", help="Path to the old single-file snapshot array (default: saved_portfolios.json)")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR, help=f"Destination directory (default: {DEFAULT_STORAGE_DIR})")
    parser.add_argument("--keep-all", action="store_true", help="Migrate every snapshot instead of filtering by name")
    parser.add_argument("--apply", action="store_true", help="Actually write files. Without this flag, only prints what would happen.")
    args = parser.parse_args()

    if not os.path.exists(args.legacy_path):
        print(f"Nie znaleziono {args.legacy_path} -- nic do migracji.")
        return

    with open(args.legacy_path, "r", encoding="utf-8") as f:
        try:
            legacy_snapshots = json.load(f)
        except json.JSONDecodeError as e:
            print(f"BŁĄD: {args.legacy_path} nie jest poprawnym JSON-em: {e}")
            return

    if not isinstance(legacy_snapshots, list):
        print(f"BŁĄD: oczekiwano listy snapshotów w {args.legacy_path}, znaleziono {type(legacy_snapshots).__name__}.")
        return

    print(f"Znaleziono {len(legacy_snapshots)} snapshot(ów) w {args.legacy_path}:\n")

    kept, discarded = [], []
    for s in legacy_snapshots:
        name = s.get("snapshot_name", "")
        sid = s.get("snapshot_id", "?")
        matches = args.keep_all or NAME_FILTER in name.lower()
        (kept if matches else discarded).append(s)
        tag = "ZACHOWANY " if matches else "pominięty "
        print(f"  [{tag}] {sid}  —  \"{name}\"")

    print(f"\nDo migracji: {len(kept)} / {len(legacy_snapshots)} (pominięto {len(discarded)} jako dane testowe).")

    if not kept:
        print("Nic nie pasuje do filtra -- nic do zapisania. (Użyj --keep-all, jeśli to niezamierzone.)")
        return

    print(f"\nDocelowy katalog: {args.storage_dir}/")
    for s in kept:
        target_path = _snapshot_file_path(s["snapshot_id"], args.storage_dir)
        exists_note = " (JUŻ ISTNIEJE -- zostanie nadpisany)" if os.path.exists(target_path) else ""
        print(f"  {target_path}{exists_note}")

    if not args.apply:
        print("\nTo był dry-run -- nic nie zapisano. Uruchom ponownie z --apply, żeby faktycznie zapisać pliki.")
        return

    os.makedirs(args.storage_dir, exist_ok=True)
    for s in kept:
        target_path = _snapshot_file_path(s["snapshot_id"], args.storage_dir)
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target_path)

    print(f"\nGotowe -- zapisano {len(kept)} plik(ów) do {args.storage_dir}/.")
    print(f"Stary plik {args.legacy_path} NIE został zmieniony ani usunięty (zostaje jako kopia zapasowa).")


if __name__ == "__main__":
    main()
