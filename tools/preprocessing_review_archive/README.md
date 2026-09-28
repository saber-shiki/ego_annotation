# Preprocessing review script archive

Byte-for-byte copies of the external audit, evaluation and visualization scripts used through 2026-09-28. `archive_manifest.json` records original paths and SHA256. Source observations, CAD/GT evaluation sidecars, videos, arrays, logs and model assets remain outside Git at their existing immutable locations.

These historical scripts retain their original absolute paths and `__file__`-relative artifact assumptions: this is a source archive, **not a relocatable CLI package**. Run the original script in the recorded experiment directory, or restore the file to that directory when reproducing an old review. Do not run them in this archive expecting the data to be present. No dependency installation or data fetch is performed by archiving. `fetch.py` is a historical read-only S3 retrieval script, not a runtime prerequisite.

GT/CAD evaluation utilities are evaluation-only and must not enter prediction runtime snapshots. The prediction producer is `python -m preprocessing --profile visible-surface`.
