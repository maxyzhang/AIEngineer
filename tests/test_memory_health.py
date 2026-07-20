from memory_metrics import (
    generate_memory_health_warnings,
    generate_memory_recommendations,
)


def make_health_summary(
    *,
    total_memories: int = 10,
    average_importance: float = 5.0,
    average_access_count: float = 2.0,
    stale_memories: int = 0,
    high_value_memories: int = 3,
    never_accessed_memories: int = 0,
) -> dict[str, int | float]:
    return {
        "total_memories": total_memories,
        "average_importance": average_importance,
        "average_access_count": average_access_count,
        "stale_memories": stale_memories,
        "high_value_memories": high_value_memories,
        "never_accessed_memories": never_accessed_memories,
    }

def test_empty_memory_store_returns_warning() -> None:
    health = make_health_summary(
        total_memories=0,
        high_value_memories=0,
    )

    warnings = generate_memory_health_warnings(health)

    assert len(warnings) == 1
    assert (
        "no persistent long-tem memories are avaiable"
    )


def test_high_stale_ratio_returns_warning() -> None:
    health = make_health_summary(
        total_memories=10,
        stale_memories=4,
    )

    warnings = generate_memory_health_warnings(health)

    assert any(
        "stale" in warning.lower()
        for warning in warnings
    )


def test_high_never_accessed_ratio_returns_warning() -> None:
    health = make_health_summary(
        total_memories=10,
        never_accessed_memories=4,
    )

    warnings = generate_memory_health_warnings(health)

    assert any(
        "never been accessed" in warning.lower()
        for warning in warnings
    )


def test_low_average_importance_returns_warning() -> None:
    health = make_health_summary(
        average_importance=3.9,
    )

    warnings = generate_memory_health_warnings(health)

    assert any(
        "importance" in warning.lower()
        for warning in warnings
    )


def test_missing_high_value_memories_returns_warning() -> None:
    health = make_health_summary(
        high_value_memories=0,
    )

    warnings = generate_memory_health_warnings(health)

    assert any(
        "high-value" in warning.lower()
        for warning in warnings
    )


def test_healthy_memory_store_returns_no_warnings() -> None:
    health = make_health_summary()

    warnings = generate_memory_health_warnings(health)

    assert warnings == []

def test_stale_memories_recommend_decay_or_gc() -> None:
    health = make_health_summary(
        stale_memories=3,
    )

    warnings = generate_memory_health_warnings(health)

    recommendations = generate_memory_recommendations(
        health,
        warnings,
    )

    assert any(
        "decay" in recommendation.lower()
        or "garbage collection" in recommendation.lower()
        for recommendation in recommendations
    )


def test_never_accessed_memories_recommend_reviewing_extraction() -> None:
    health = make_health_summary(
        never_accessed_memories=3,
    )

    warnings = generate_memory_health_warnings(health)

    recommendations = generate_memory_recommendations(
        health,
        warnings,
    )

    assert any(
        "extraction" in recommendation.lower()
        for recommendation in recommendations
    )


def test_low_importance_recommends_adjusting_rules() -> None:
    health = make_health_summary(
        average_importance=3.5,
    )

    warnings = generate_memory_health_warnings(health)

    recommendations = generate_memory_recommendations(
        health,
        warnings,
    )

    assert any(
        "importance" in recommendation.lower()
        for recommendation in recommendations
    )


def test_missing_high_value_memories_recommends_threshold_review() -> None:
    health = make_health_summary(
        high_value_memories=0,
    )

    warnings = generate_memory_health_warnings(health)

    recommendations = generate_memory_recommendations(
        health,
        warnings,
    )

    assert any(
        "threshold" in recommendation.lower()
        or "reinforcement" in recommendation.lower()
        for recommendation in recommendations
    )


def test_healthy_memory_store_returns_healthy_recommendation() -> None:
    health = make_health_summary()
    warnings: list[str] = []

    recommendations = generate_memory_recommendations(
        health,
        warnings,
    )

    assert len(recommendations) == 1
    assert (
        "no immediate action is required"
        in recommendations[0].lower()
    )
