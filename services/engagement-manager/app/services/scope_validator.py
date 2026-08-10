from __future__ import annotations

import fnmatch
import ipaddress


def target_matches(rule_target_type: str, rule_value: str, candidate: str) -> bool:
    if rule_target_type == "domain":
        return fnmatch.fnmatch(candidate.lower(), rule_value.lower())
    if rule_target_type == "url":
        return candidate.startswith(rule_value) or fnmatch.fnmatch(candidate, rule_value)
    if rule_target_type == "ip":
        return candidate == rule_value
    if rule_target_type == "cidr":
        try:
            return ipaddress.ip_address(candidate) in ipaddress.ip_network(
                rule_value, strict=False
            )
        except ValueError:
            return False
    return False


def is_target_allowed(
    rules: list[tuple[str, str, str]],
    *,
    target_type: str,
    target_value: str,
) -> bool:
    """
    Apply deny-first then allowlist semantics.

    rules: list of (rule_type, target_type, target_value)
    """
    matching = [
        r
        for r in rules
        if r[1] == target_type and target_matches(r[1], r[2], target_value)
    ]
    if any(r[0] == "deny" for r in matching):
        return False
    allows = [r for r in rules if r[0] == "allow"]
    if not allows:
        return False
    return any(
        r[1] == target_type and target_matches(r[1], r[2], target_value) for r in allows
    )
