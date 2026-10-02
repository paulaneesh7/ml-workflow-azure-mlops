"""Generate a synthetic change-request dataset for cycle-time prediction.

The CSV has 10,000 rows and 44 columns:

* 29 legitimate features known before implementation finishes
* 14 irrelevant or future-information columns that must not be used for training
* cycle_time_days, the prediction target

Cycle time is a noisy combination of many pre-resolution features. It is not a
transform of any single column. Closure, resolution, and post-implementation
fields are created afterward and leak the outcome.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

ROW_COUNT = 10_000
RANDOM_SEED = 42

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "change_requests.csv"
)

LEGITIMATE_FEATURES = [
    "created_at",
    "change_type",
    "change_category",
    "change_driver",
    "environment",
    "business_unit",
    "region",
    "requester_role",
    "priority",
    "risk_level",
    "complexity",
    "approval_path",
    "required_approver_count",
    "assignment_group",
    "assignment_backlog_size",
    "assignee_experience_months",
    "application_name",
    "application_criticality",
    "dependency_type",
    "dependency_count",
    "affected_system_count",
    "implementation_window",
    "requested_implementation_date",
    "rollback_plan_ready",
    "testing_scope",
    "customer_impact",
    "estimated_effort_hours",
    "planned_downtime_minutes",
    "description_length",
]

LEAKAGE_COLUMNS = [
    "cr_id",
    "requester_email",
    "final_approval_at",
    "actual_start_at",
    "actual_end_at",
    "resolved_at",
    "closed_at",
    "closure_code",
    "resolution_summary",
    "post_implementation_review",
    "actual_effort_hours",
    "rollback_performed",
    "sla_breached",
    "record_last_updated_at",
]

TARGET = "cycle_time_days"

COLUMNS = [
    "cr_id",
    "created_at",
    "change_type",
    "change_category",
    "change_driver",
    "environment",
    "business_unit",
    "region",
    "requester_role",
    "requester_email",
    "priority",
    "risk_level",
    "complexity",
    "approval_path",
    "required_approver_count",
    "assignment_group",
    "assignment_backlog_size",
    "assignee_experience_months",
    "application_name",
    "application_criticality",
    "dependency_type",
    "dependency_count",
    "affected_system_count",
    "implementation_window",
    "requested_implementation_date",
    "rollback_plan_ready",
    "testing_scope",
    "customer_impact",
    "estimated_effort_hours",
    "planned_downtime_minutes",
    "description_length",
    "final_approval_at",
    "actual_start_at",
    "actual_end_at",
    "resolved_at",
    "closed_at",
    "closure_code",
    "resolution_summary",
    "post_implementation_review",
    "actual_effort_hours",
    "rollback_performed",
    "sla_breached",
    "record_last_updated_at",
    "cycle_time_days",
]

CREATED_START = datetime(2024, 1, 1)
CREATED_END = datetime(2026, 9, 1)

HOUR_WEIGHTS = [
    0.2, 0.1, 0.1, 0.1, 0.1, 0.3,
    0.8, 1.5,
    4.0, 5.0, 5.5, 5.5, 5.0, 4.5, 4.0, 4.5, 5.0, 4.5,
    3.5, 2.0,
    1.2, 0.8, 0.5, 0.4,
]

CHANGE_TYPES = ["Standard", "Normal", "Emergency", "Expedited"]
CHANGE_CATEGORIES = [
    "Application",
    "Infrastructure",
    "Database",
    "Network",
    "Security",
    "Cloud",
]
CHANGE_DRIVERS = [
    "Enhancement",
    "Defect Fix",
    "Compliance",
    "Capacity",
    "Incident Prevention",
    "Configuration",
]
ENVIRONMENTS = ["Production", "Staging", "QA", "Development"]
BUSINESS_UNITS = [
    "Retail Banking",
    "Payments",
    "Lending",
    "Wealth",
    "Operations",
    "Risk and Compliance",
    "Technology",
]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REQUESTER_ROLES = [
    "Developer",
    "Business Analyst",
    "SRE",
    "Product Owner",
    "Security Engineer",
    "DBA",
]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
RISK_LEVELS = ["Low", "Medium", "High"]
COMPLEXITIES = ["Simple", "Moderate", "Complex", "Very Complex"]
APPROVAL_PATHS = ["Auto-Approved", "Manager", "CAB", "Emergency CAB"]
ASSIGNMENT_GROUPS = [
    "App Support",
    "Platform Engineering",
    "Database",
    "Network",
    "Security Ops",
    "Cloud Ops",
]
APPLICATIONS = [
    "Payments Hub",
    "Core Ledger",
    "Customer 360",
    "API Gateway",
    "Data Lake",
    "Identity Service",
    "Card Processor",
    "Mobile Banking",
]
APPLICATION_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
DEPENDENCY_TYPES = [
    "None",
    "Upstream Service",
    "Downstream Consumer",
    "Bidirectional",
    "External Vendor",
]
IMPLEMENTATION_WINDOWS = [
    "Business Hours",
    "After Hours",
    "Weekend",
    "Maintenance Window",
]
TESTING_SCOPES = ["Full Regression", "Smoke", "Unit Only", "Not Required"]
CUSTOMER_IMPACTS = ["None", "Internal Only", "Limited Customers", "Widespread"]
CLOSURE_CODES = [
    "Successful",
    "Successful with Issues",
    "Failed",
    "Backed Out",
    "Cancelled",
]

CATEGORY_GROUPS = {
    "Application": ["App Support", "Platform Engineering"],
    "Infrastructure": ["Platform Engineering", "Cloud Ops"],
    "Database": ["Database"],
    "Network": ["Network"],
    "Security": ["Security Ops"],
    "Cloud": ["Cloud Ops", "Platform Engineering"],
}

GIVEN_NAMES = [
    "alex", "jordan", "taylor", "morgan", "casey", "riley", "quinn",
    "avery", "parker", "drew", "sam", "jamie", "reese", "skyler", "cameron",
]
SURNAMES = [
    "lee", "patel", "nguyen", "garcia", "singh", "brown", "martin",
    "kim", "wagner", "rossi", "khan", "ibrahim", "cohen", "silva", "novak",
]

TYPE_EFFECT = {"Emergency": -0.75, "Expedited": -0.32, "Standard": 0.04, "Normal": 0.20}
PRIORITY_EFFECT = {"Critical": -0.48, "High": -0.16, "Medium": 0.08, "Low": 0.30}
COMPLEXITY_EFFECT = {"Simple": -0.38, "Moderate": 0.0, "Complex": 0.32, "Very Complex": 0.58}
RISK_EFFECT = {"Low": -0.14, "Medium": 0.04, "High": 0.26}
APPROVAL_EFFECT = {"Auto-Approved": -0.42, "Manager": -0.04, "Emergency CAB": 0.02, "CAB": 0.34}
ENV_EFFECT = {"Development": -0.22, "QA": -0.08, "Staging": 0.0, "Production": 0.16}
IMPACT_EFFECT = {"None": -0.10, "Internal Only": 0.0, "Limited Customers": 0.12, "Widespread": 0.24}
WINDOW_EFFECT = {
    "Business Hours": 0.04,
    "After Hours": 0.0,
    "Weekend": 0.12,
    "Maintenance Window": 0.08,
}
TIER_EFFECT = {"Tier 4": -0.08, "Tier 3": 0.0, "Tier 2": 0.06, "Tier 1": 0.14}
TEST_EFFECT = {
    "Not Required": -0.16,
    "Unit Only": -0.04,
    "Smoke": 0.04,
    "Full Regression": 0.18,
}
ROLLBACK_EFFECT = {"Yes": -0.04, "No": 0.12}
DRIVER_EFFECT = {
    "Enhancement": 0.08,
    "Defect Fix": -0.12,
    "Compliance": 0.14,
    "Capacity": 0.05,
    "Incident Prevention": -0.16,
    "Configuration": -0.08,
}
CATEGORY_EFFECT = {
    "Application": 0.0,
    "Infrastructure": 0.06,
    "Database": 0.10,
    "Network": 0.05,
    "Security": 0.12,
    "Cloud": 0.04,
}
DOW_EFFECT = {0: 0.04, 1: 0.0, 2: 0.0, 3: 0.02, 4: 0.16, 5: 0.10, 6: 0.12}
REGION_EFFECT = {"North America": 0.0, "EMEA": 0.04, "APAC": 0.07, "LATAM": 0.03}

SLA_DAYS = {"Critical": 2.0, "High": 5.0, "Medium": 12.0, "Low": 25.0}
APPROVAL_FRACTION = {
    "Auto-Approved": (0.01, 0.08),
    "Manager": (0.06, 0.28),
    "Emergency CAB": (0.04, 0.22),
    "CAB": (0.18, 0.55),
}
LEAD_TIME_DAYS = {
    "Emergency": (0, 2),
    "Expedited": (1, 6),
    "Standard": (4, 20),
    "Normal": (7, 35),
}


def pick(rng: random.Random, options: list[str], weights: list[float]) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def sample_created_at(rng: random.Random) -> datetime:
    span_days = (CREATED_END - CREATED_START).days
    for _ in range(30):
        day_offset = rng.randrange(span_days)
        created_day = CREATED_START + timedelta(days=day_offset)
        if created_day.weekday() >= 5 and rng.random() > 0.16:
            continue
        hour = pick(rng, [str(hour) for hour in range(24)], HOUR_WEIGHTS)
        return created_day.replace(
            hour=int(hour),
            minute=rng.randrange(60),
            second=rng.randrange(60),
        )
    return CREATED_START.replace(hour=10, minute=15, second=0)


def sample_priority(rng: random.Random, change_type: str, change_driver: str) -> str:
    if change_type == "Emergency":
        weights = [0.55, 0.34, 0.09, 0.02]
    elif change_type == "Expedited":
        weights = [0.18, 0.50, 0.26, 0.06]
    elif change_driver in {"Defect Fix", "Incident Prevention"}:
        weights = [0.10, 0.34, 0.42, 0.14]
    else:
        weights = [0.04, 0.18, 0.50, 0.28]
    return pick(rng, PRIORITIES, weights)


def sample_complexity(rng: random.Random, change_driver: str) -> str:
    if change_driver == "Configuration":
        weights = [0.46, 0.38, 0.13, 0.03]
    elif change_driver == "Compliance":
        weights = [0.08, 0.32, 0.40, 0.20]
    else:
        weights = [0.22, 0.44, 0.24, 0.10]
    return pick(rng, COMPLEXITIES, weights)


def sample_customer_impact(
    rng: random.Random, environment: str, change_type: str
) -> str:
    if environment == "Production":
        if change_type == "Emergency":
            weights = [0.05, 0.20, 0.40, 0.35]
        else:
            weights = [0.12, 0.34, 0.38, 0.16]
    elif environment == "Development":
        weights = [0.55, 0.35, 0.08, 0.02]
    else:
        weights = [0.30, 0.42, 0.22, 0.06]
    return pick(rng, CUSTOMER_IMPACTS, weights)


def sample_risk_level(
    rng: random.Random,
    complexity: str,
    environment: str,
    customer_impact: str,
    change_type: str,
) -> str:
    score = {
        "Simple": 0,
        "Moderate": 1,
        "Complex": 2,
        "Very Complex": 3,
    }[complexity]
    score += {"Development": 0, "QA": 0, "Staging": 1, "Production": 2}[environment]
    score += {
        "None": 0,
        "Internal Only": 1,
        "Limited Customers": 2,
        "Widespread": 3,
    }[customer_impact]
    score += 2 if change_type == "Emergency" else 0
    score += rng.randint(-1, 2)
    if score <= 2:
        return "Low"
    if score <= 5:
        return "Medium"
    return "High"


def sample_approval_path(
    rng: random.Random,
    change_type: str,
    risk_level: str,
    complexity: str,
    environment: str,
) -> str:
    if change_type == "Emergency":
        return pick(rng, ["Emergency CAB", "Manager", "CAB"], [0.72, 0.20, 0.08])
    if (
        risk_level == "Low"
        and complexity == "Simple"
        and environment != "Production"
    ):
        return pick(rng, APPROVAL_PATHS, [0.62, 0.28, 0.08, 0.02])
    if risk_level == "High" or (
        environment == "Production" and complexity in {"Complex", "Very Complex"}
    ):
        return pick(rng, APPROVAL_PATHS, [0.02, 0.18, 0.74, 0.06])
    return pick(rng, APPROVAL_PATHS, [0.12, 0.48, 0.36, 0.04])


def sample_required_approvers(rng: random.Random, approval_path: str) -> int:
    if approval_path == "Auto-Approved":
        return 0
    if approval_path == "Manager":
        return 1
    if approval_path == "Emergency CAB":
        return rng.choice([1, 2])
    return rng.choice([3, 3, 4, 4, 5])


def sample_assignment_group(rng: random.Random, change_category: str) -> str:
    if rng.random() < 0.84:
        return rng.choice(CATEGORY_GROUPS[change_category])
    return rng.choice(ASSIGNMENT_GROUPS)


def sample_application_tier(
    rng: random.Random, environment: str, customer_impact: str
) -> str:
    if environment == "Production" and customer_impact == "Widespread":
        weights = [0.46, 0.34, 0.15, 0.05]
    elif environment == "Development":
        weights = [0.05, 0.15, 0.35, 0.45]
    else:
        weights = [0.18, 0.32, 0.32, 0.18]
    return pick(rng, APPLICATION_TIERS, weights)


def sample_dependency(
    rng: random.Random, complexity: str
) -> tuple[str, int]:
    if complexity == "Simple":
        weights = [0.48, 0.22, 0.14, 0.08, 0.08]
    elif complexity == "Very Complex":
        weights = [0.08, 0.28, 0.18, 0.26, 0.20]
    else:
        weights = [0.26, 0.28, 0.18, 0.16, 0.12]
    dependency_type = pick(rng, DEPENDENCY_TYPES, weights)
    if dependency_type == "None":
        return dependency_type, 0
    upper = {"Simple": 3, "Moderate": 5, "Complex": 8, "Very Complex": 12}[complexity]
    return dependency_type, rng.randint(1, upper)


def sample_affected_systems(
    rng: random.Random, change_category: str, dependency_count: int
) -> int:
    base = 1 + dependency_count // 2
    extra = {
        "Application": (0, 3),
        "Infrastructure": (1, 8),
        "Database": (0, 4),
        "Network": (1, 6),
        "Security": (0, 5),
        "Cloud": (1, 7),
    }[change_category]
    return int(clip(base + rng.randint(extra[0], extra[1]), 1, 30))


def sample_implementation_window(
    rng: random.Random, change_type: str, environment: str
) -> str:
    if change_type == "Emergency":
        weights = [0.34, 0.46, 0.15, 0.05]
    elif environment == "Production":
        weights = [0.18, 0.22, 0.24, 0.36]
    else:
        weights = [0.55, 0.20, 0.12, 0.13]
    return pick(rng, IMPLEMENTATION_WINDOWS, weights)


def sample_testing_scope(
    rng: random.Random, complexity: str, environment: str
) -> str:
    if environment == "Production" and complexity in {"Complex", "Very Complex"}:
        weights = [0.62, 0.28, 0.08, 0.02]
    elif complexity == "Simple" and environment == "Development":
        weights = [0.05, 0.15, 0.35, 0.45]
    else:
        weights = [0.22, 0.40, 0.28, 0.10]
    return pick(rng, TESTING_SCOPES, weights)


def sample_rollback_plan(
    rng: random.Random, environment: str, risk_level: str
) -> str:
    if environment == "Production" and risk_level == "High":
        probability_yes = 0.92
    elif environment == "Development" and risk_level == "Low":
        probability_yes = 0.38
    else:
        probability_yes = 0.70
    return "Yes" if rng.random() < probability_yes else "No"


def sample_effort_hours(rng: random.Random, complexity: str) -> float:
    center = {"Simple": 4.0, "Moderate": 10.0, "Complex": 22.0, "Very Complex": 40.0}
    hours = center[complexity] * rng.lognormvariate(0.0, 0.38)
    return round(clip(hours, 0.5, 120.0), 1)


def sample_downtime_minutes(
    rng: random.Random,
    change_category: str,
    environment: str,
    customer_impact: str,
) -> int:
    if environment != "Production" or customer_impact == "None":
        if rng.random() < 0.72:
            return 0
    if change_category in {"Database", "Infrastructure", "Network"}:
        return int(rng.choice([0, 15, 30, 45, 60, 90, 120, 180]))
    if rng.random() < 0.45:
        return 0
    return int(rng.choice([5, 10, 15, 20, 30, 45]))


def sample_description_length(rng: random.Random, complexity: str) -> int:
    center = {"Simple": 5.2, "Moderate": 5.6, "Complex": 6.0, "Very Complex": 6.3}
    length = int(rng.lognormvariate(center[complexity], 0.35))
    return int(clip(length, 40, 2500))


def sample_backlog(rng: random.Random, assignment_group: str) -> int:
    ranges = {
        "App Support": (6, 45),
        "Platform Engineering": (4, 32),
        "Database": (2, 24),
        "Network": (1, 18),
        "Security Ops": (3, 28),
        "Cloud Ops": (5, 36),
    }
    low, high = ranges[assignment_group]
    return rng.randint(low, high)


def sample_experience_months(rng: random.Random) -> int:
    months = int(rng.lognormvariate(3.55, 0.55))
    return int(clip(months, 2, 220))


def sample_requested_date(
    rng: random.Random,
    created_at: datetime,
    change_type: str,
    priority: str,
) -> str:
    low, high = LEAD_TIME_DAYS[change_type]
    if priority == "Critical":
        high = max(low, high // 2)
    elif priority == "Low":
        low = min(high, low + 2)
    lead_days = rng.randint(low, high)
    requested = created_at + timedelta(days=lead_days)
    return requested.strftime("%Y-%m-%d")


def generate_cycle_time_days(rng: random.Random, features: dict) -> float:
    """Combine many pre-resolution signals with noise.

    No single feature determines the result. Two requests with the same effort,
    priority, or approval path still receive different cycle times.
    """
    signal = (
        1.12
        + TYPE_EFFECT[features["change_type"]]
        + PRIORITY_EFFECT[features["priority"]]
        + COMPLEXITY_EFFECT[features["complexity"]]
        + RISK_EFFECT[features["risk_level"]]
        + APPROVAL_EFFECT[features["approval_path"]]
        + ENV_EFFECT[features["environment"]]
        + IMPACT_EFFECT[features["customer_impact"]]
        + WINDOW_EFFECT[features["implementation_window"]]
        + TIER_EFFECT[features["application_criticality"]]
        + TEST_EFFECT[features["testing_scope"]]
        + ROLLBACK_EFFECT[features["rollback_plan_ready"]]
        + DRIVER_EFFECT[features["change_driver"]]
        + CATEGORY_EFFECT[features["change_category"]]
        + DOW_EFFECT[features["created_at"].weekday()]
        + REGION_EFFECT[features["region"]]
        + 0.009 * features["estimated_effort_hours"]
        + 0.05 * math.log1p(features["dependency_count"])
        + 0.025 * math.log1p(features["affected_system_count"])
        + 0.06 * features["required_approver_count"]
        + 0.005 * features["assignment_backlog_size"]
        - 0.0026 * features["assignee_experience_months"]
        + 0.00028 * features["description_length"]
        + 0.0011 * features["planned_downtime_minutes"]
    )
    cycle_time = math.exp(signal + rng.gauss(0, 0.38))
    return round(clip(cycle_time, 0.2, 90.0), 2)


def sample_closure_code(rng: random.Random) -> str:
    return pick(rng, CLOSURE_CODES, [78, 9, 4, 4, 5])


def build_resolution_fields(
    rng: random.Random,
    features: dict,
    cycle_time_days: float,
) -> dict:
    """Fields that are unknown until approval, implementation, or closure."""
    created_at = features["created_at"]
    cycle_delta = timedelta(days=cycle_time_days)
    jitter = timedelta(hours=rng.gauss(0, 4.5))
    resolved_at = created_at + cycle_delta + jitter
    if resolved_at < created_at + timedelta(hours=2):
        resolved_at = created_at + timedelta(hours=2)

    low, high = APPROVAL_FRACTION[features["approval_path"]]
    approval_fraction = rng.uniform(low, high)
    final_approval_at = created_at + timedelta(
        seconds=approval_fraction * cycle_delta.total_seconds()
    )
    if final_approval_at >= resolved_at - timedelta(minutes=20):
        final_approval_at = created_at + (resolved_at - created_at) * 0.25

    closure_code = sample_closure_code(rng)
    cancelled = closure_code == "Cancelled"

    if cancelled and rng.random() < 0.6:
        final_approval_at_value = ""
        actual_start_value = ""
        actual_end_value = ""
    elif cancelled:
        final_approval_at_value = format_timestamp(final_approval_at)
        actual_start_value = ""
        actual_end_value = ""
    else:
        remaining_seconds = max(
            (resolved_at - final_approval_at).total_seconds(),
            30 * 60,
        )
        actual_start = final_approval_at + timedelta(
            seconds=rng.uniform(0.50, 0.90) * remaining_seconds
        )
        if features["planned_downtime_minutes"] > 0:
            duration_minutes = features["planned_downtime_minutes"] * rng.uniform(0.6, 1.35)
        else:
            duration_minutes = rng.uniform(20, 150)
        duration_minutes = clip(duration_minutes, 15, 12 * 60)
        actual_end = actual_start + timedelta(minutes=duration_minutes)
        if actual_end > resolved_at - timedelta(minutes=5):
            actual_end = resolved_at - timedelta(minutes=5)
        if actual_end <= actual_start:
            actual_end = actual_start + timedelta(minutes=15)
        final_approval_at_value = format_timestamp(final_approval_at)
        actual_start_value = format_timestamp(actual_start)
        actual_end_value = format_timestamp(actual_end)

    closed_at = resolved_at + timedelta(hours=rng.uniform(1, 40))
    record_last_updated_at = closed_at + timedelta(minutes=rng.randint(5, 800))

    if closure_code == "Backed Out":
        rollback_performed = "Yes"
    elif closure_code == "Failed":
        rollback_performed = "Yes" if rng.random() < 0.7 else "No"
    elif closure_code == "Cancelled":
        rollback_performed = "No"
    else:
        rollback_performed = "Yes" if rng.random() < 0.04 else "No"

    if closure_code == "Cancelled":
        post_implementation_review = "Not Applicable"
    elif closure_code in {"Failed", "Backed Out", "Successful with Issues"}:
        post_implementation_review = pick(
            rng, ["Findings Raised", "Pending Actions"], [0.8, 0.2]
        )
    else:
        post_implementation_review = pick(
            rng, ["Passed", "Waived", "Findings Raised"], [0.72, 0.16, 0.12]
        )

    if cancelled:
        actual_effort = round(rng.uniform(0.2, 2.5), 1)
    else:
        actual_effort = features["estimated_effort_hours"] * rng.lognormvariate(0.04, 0.25)
        actual_effort *= 1 + 0.012 * cycle_time_days
        actual_effort = round(clip(actual_effort, 0.5, 200.0), 1)

    sla_breached = (
        "Yes" if cycle_time_days > SLA_DAYS[features["priority"]] else "No"
    )
    resolution_summary = (
        f"{features['change_type']} {features['change_category'].lower()} change "
        f"for {features['application_name']} in {features['environment']}. "
        f"Closure recorded as {closure_code}."
    )

    return {
        "final_approval_at": final_approval_at_value,
        "actual_start_at": actual_start_value,
        "actual_end_at": actual_end_value,
        "resolved_at": format_timestamp(resolved_at),
        "closed_at": format_timestamp(closed_at),
        "closure_code": closure_code,
        "resolution_summary": resolution_summary,
        "post_implementation_review": post_implementation_review,
        "actual_effort_hours": actual_effort,
        "rollback_performed": rollback_performed,
        "sla_breached": sla_breached,
        "record_last_updated_at": format_timestamp(record_last_updated_at),
    }


def generate_change_request(rng: random.Random, index: int) -> dict:
    change_type = pick(rng, CHANGE_TYPES, [0.34, 0.46, 0.12, 0.08])
    change_category = pick(rng, CHANGE_CATEGORIES, [0.30, 0.18, 0.14, 0.12, 0.12, 0.14])
    change_driver = pick(rng, CHANGE_DRIVERS, [0.28, 0.22, 0.12, 0.10, 0.12, 0.16])
    environment = pick(rng, ENVIRONMENTS, [0.48, 0.18, 0.20, 0.14])
    complexity = sample_complexity(rng, change_driver)
    customer_impact = sample_customer_impact(rng, environment, change_type)
    risk_level = sample_risk_level(
        rng, complexity, environment, customer_impact, change_type
    )
    approval_path = sample_approval_path(
        rng, change_type, risk_level, complexity, environment
    )
    dependency_type, dependency_count = sample_dependency(rng, complexity)
    created_at = sample_created_at(rng)
    priority = sample_priority(rng, change_type, change_driver)

    features = {
        "created_at": created_at,
        "change_type": change_type,
        "change_category": change_category,
        "change_driver": change_driver,
        "environment": environment,
        "business_unit": pick(
            rng, BUSINESS_UNITS, [0.18, 0.16, 0.12, 0.10, 0.16, 0.12, 0.16]
        ),
        "region": pick(rng, REGIONS, [0.40, 0.28, 0.22, 0.10]),
        "requester_role": pick(
            rng, REQUESTER_ROLES, [0.28, 0.18, 0.16, 0.14, 0.12, 0.12]
        ),
        "priority": priority,
        "risk_level": risk_level,
        "complexity": complexity,
        "approval_path": approval_path,
        "required_approver_count": sample_required_approvers(rng, approval_path),
        "assignment_group": sample_assignment_group(rng, change_category),
        "assignee_experience_months": sample_experience_months(rng),
        "application_name": rng.choice(APPLICATIONS),
        "application_criticality": sample_application_tier(
            rng, environment, customer_impact
        ),
        "dependency_type": dependency_type,
        "dependency_count": dependency_count,
        "affected_system_count": sample_affected_systems(
            rng, change_category, dependency_count
        ),
        "implementation_window": sample_implementation_window(
            rng, change_type, environment
        ),
        "requested_implementation_date": sample_requested_date(
            rng, created_at, change_type, priority
        ),
        "rollback_plan_ready": sample_rollback_plan(rng, environment, risk_level),
        "testing_scope": sample_testing_scope(rng, complexity, environment),
        "customer_impact": customer_impact,
        "estimated_effort_hours": sample_effort_hours(rng, complexity),
        "planned_downtime_minutes": sample_downtime_minutes(
            rng, change_category, environment, customer_impact
        ),
        "description_length": sample_description_length(rng, complexity),
    }
    features["assignment_backlog_size"] = sample_backlog(
        rng, features["assignment_group"]
    )

    cycle_time_days = generate_cycle_time_days(rng, features)
    leakage = build_resolution_fields(rng, features, cycle_time_days)
    email_local = (
        f"{rng.choice(GIVEN_NAMES)}.{rng.choice(SURNAMES)}{rng.randint(1, 97)}"
    )

    row = {
        "cr_id": f"CHG{1_000_000 + index:07d}",
        "requester_email": f"{email_local}@change-example.internal",
        "created_at": format_timestamp(created_at),
        "cycle_time_days": cycle_time_days,
        **{name: features[name] for name in LEGITIMATE_FEATURES if name != "created_at"},
        **leakage,
    }
    return {column: row[column] for column in COLUMNS}


def validate_schema() -> None:
    if len(LEGITIMATE_FEATURES) != 29:
        raise RuntimeError(
            f"Expected 29 legitimate features, found {len(LEGITIMATE_FEATURES)}"
        )
    if len(LEAKAGE_COLUMNS) != 14:
        raise RuntimeError(f"Expected 14 leakage columns, found {len(LEAKAGE_COLUMNS)}")
    if len(COLUMNS) != 44:
        raise RuntimeError(f"Expected 44 columns, found {len(COLUMNS)}")
    if len(set(COLUMNS)) != 44:
        raise RuntimeError("Column names are not unique")
    expected = set(LEGITIMATE_FEATURES) | set(LEAKAGE_COLUMNS) | {TARGET}
    if expected != set(COLUMNS):
        missing = expected - set(COLUMNS)
        extra = set(COLUMNS) - expected
        raise RuntimeError(f"Column groups do not match schema: {missing=} {extra=}")
    if len(HOUR_WEIGHTS) != 24:
        raise RuntimeError("Hour weights must cover 24 hours")


def generate_dataset(rng: random.Random) -> list[dict]:
    return [generate_change_request(rng, index) for index in range(ROW_COUNT)]


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    validate_schema()
    rng = random.Random(RANDOM_SEED)
    rows = generate_dataset(rng)
    if len(rows) != ROW_COUNT or any(len(row) != 44 for row in rows):
        raise RuntimeError("Generated dataset does not have 10,000 rows and 44 columns")
    write_csv(rows, OUTPUT_PATH)

    print(f"Saved {OUTPUT_PATH}")
    print("Column names:")
    for name in COLUMNS:
        print(name)
    print(f"Shape: ({len(rows)}, {len(COLUMNS)})")


if __name__ == "__main__":
    main()
