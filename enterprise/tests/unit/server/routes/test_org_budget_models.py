from __future__ import annotations

import pytest
from pydantic import ValidationError
from server.routes import org_models
from server.routes.org_models import (
    OrgBudgetSettingsUpdate,
    OrgBudgetThresholdUpdate,
)


def test_budget_settings_rejects_invalid_reset_day():
    with pytest.raises(ValidationError, match='reset_day must be 1 or 15'):
        OrgBudgetSettingsUpdate(reset_day=2)


def test_budget_settings_rejects_duplicate_thresholds():
    with pytest.raises(ValidationError, match='threshold percentages must be unique'):
        OrgBudgetSettingsUpdate(
            thresholds=[
                OrgBudgetThresholdUpdate(percentage=80),
                OrgBudgetThresholdUpdate(percentage=80),
            ]
        )


def test_budget_settings_rejects_excess_thresholds(monkeypatch):
    monkeypatch.setattr('server.routes.org_models.MAX_BUDGET_THRESHOLDS', 1)
    with pytest.raises(
        ValidationError,
        match=f'thresholds cannot exceed {org_models.MAX_BUDGET_THRESHOLDS} entries',
    ):
        OrgBudgetSettingsUpdate(
            thresholds=[
                OrgBudgetThresholdUpdate(percentage=80),
                OrgBudgetThresholdUpdate(percentage=90),
            ]
        )
