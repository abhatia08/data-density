# Releases

Git tags mark reproducible snapshots tied to the EDI manuscript.

**Navigation:** [README](../README.md) · [Pipeline](pipeline.md) · [Methods](methods.md) · [Artifacts](artifacts.md)

Release workflow: [`.github/workflows/release.yml`](../.github/workflows/release.yml)

---

## Contents

- [Tags](#tags)
- [What ships](#what-ships-in-a-release)
- [Cut a tag](#cut-a-tag)
- [Zenodo DOI](#zenodo-doi)
- [Branches](#branches)

---

## Tags

| Pattern | Meaning |
|---------|---------|
| `submission-YYYY-MM-DD` | Code at first submission |
| `vMAJOR.MINOR.PATCH` | Semver after publication |

Current tags: **`submission-2026-06-15`**, **`v1.0.0`**

Release notes: [`.github/release-notes/`](../.github/release-notes/)

---

## What ships in a release

| Included | Excluded |
|----------|----------|
| [`scripts/`](../scripts/), [`docs/`](.) | OMOP source data |
| [`.env.example`](../.env.example), [`LICENSE`](../LICENSE) | [`outputs/`](../outputs/) |
| | Trained bundles (`*.pkl`) |

Regenerate models and artifacts on your cluster — see [Pipeline](pipeline.md) and [Artifacts](artifacts.md).

---

## Cut a tag

### Submission snapshot

```bash
git tag -a submission-YYYY-MM-DD -m "EDI code at first submission"
git push origin submission-YYYY-MM-DD
```

### Semver (post-publication)

```bash
git tag -a v1.0.1 -m "EDI v1.0.1"
git push origin v1.0.1
```

Pushing `submission-*` or `v*` runs [`.github/workflows/release.yml`](../.github/workflows/release.yml).

---

## Zenodo DOI

**`v1.0.0`** ([GitHub release](https://github.com/abhatia08/data-density/releases/tag/v1.0.0)):

| Item | Link |
|------|------|
| Version DOI | [10.5281/zenodo.20706796](https://doi.org/10.5281/zenodo.20706796) |
| Record | [zenodo.org/records/20706796](https://zenodo.org/records/20706796) |

Metadata: [`.zenodo.json`](../.zenodo.json), [`CITATION.cff`](../CITATION.cff).

Future semver tags create new Zenodo versions in the same record family; update the DOI in [`README.md`](../README.md) and [`CITATION.cff`](../CITATION.cff) when you cut a release.

---

## Branches

| Branch | Role |
|--------|------|
| `main` | Default branch |

---

## See also

- [README → Quick start](../README.md#quick-start)
- [Artifacts → Commit policy](artifacts.md#commit-policy)
