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


def test_budget_settings_accepts_valid_slack_channel():
    settings = OrgBudgetSettingsUpdate(slack_channel='#usage-monitoring_1')
    assert settings.slack_channel == '#usage-monitoring_1'


def test_budget_settings_rejects_invalid_slack_channel():
    with pytest.raises(
        ValidationError, match='slack_channel must start with # and contain only'
    ):
        OrgBudgetSettingsUpdate(slack_channel='general')


def test_budget_settings_rejects_overlong_slack_channel():
    too_long = '#' + ('a' * org_models.SLACK_CHANNEL_MAX_LENGTH)
    with pytest.raises(
        ValidationError, match='slack_channel must be .* characters or fewer'
    ):
        OrgBudgetSettingsUpdate(slack_channel=too_long)


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
