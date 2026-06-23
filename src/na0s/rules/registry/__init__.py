"""rules/registry/ -- consolidated rule-set modules.

Per v1.0.0 Step 10 (refactor/promote-rules-modules): top-level scattered
rule extensions migrate here so all signature rule sets live under one package.

Modules:
  privacy_probe -- P (privacy / data leakage) detector + PRIVACY_RULES
"""

from . import privacy_probe  # noqa: F401
