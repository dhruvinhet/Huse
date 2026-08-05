"""Targeted errors for adapters provided by optional install profiles."""


class OptionalDependencyError(ImportError):
    """Explain which project install profile enables a requested adapter."""


def missing_extra(feature: str, profile: str, package: str) -> OptionalDependencyError:
    """Build one actionable missing-extra error without leaking import internals."""

    return OptionalDependencyError(
        f"{feature} requires the optional {package!r} dependency. Install the "
        f"{profile} profile with: python -m pip install -r {profile}"
    )
