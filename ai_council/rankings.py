from __future__ import annotations

import math
from numbers import Real
from typing import Any


def ranking_ids(value: Any, *, accept_id_objects: bool = False) -> list[str]:
    """Return participant IDs from a supported ranking representation.

    Preferred representation is an ordered JSON list of participant IDs:
    ["P1", "P2", "P3"]. For robustness, we also support maps from
    participant ID to ordinal rank, such as {"P1": 1, "P2": 2, "P3": 3}.
    """

    if isinstance(value, list):
        ids: list[str] = []
        for item in value:
            if isinstance(item, str):
                ids.append(item)
            elif accept_id_objects and isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
            else:
                return []
        return ids

    if isinstance(value, dict):
        ranked_items: list[tuple[float, str]] = []
        for participant_id, rank in value.items():
            parsed_rank = _numeric_rank(rank)
            if not isinstance(participant_id, str) or parsed_rank is None:
                return []
            ranked_items.append((parsed_rank, participant_id))
        return [participant_id for _, participant_id in sorted(ranked_items)]

    return []


def is_supported_ranking_shape(value: Any) -> bool:
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(participant_id, str) and _numeric_rank(rank) is not None
            for participant_id, rank in value.items()
        )
    return False


def duplicate_rank_positions(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    positions = [_numeric_rank(rank) for rank in value.values()]
    duplicate_positions = {
        position
        for position in positions
        if position is not None and positions.count(position) > 1
    }
    return [str(position).removesuffix(".0") for position in sorted(duplicate_positions)]


def kendall_tau_against_prior(ranking: list[str], prior_ranks: dict[str, int | float]) -> float | None:
    concordant = 0
    discordant = 0
    for i, left in enumerate(ranking):
        for right in ranking[i + 1 :]:
            if left not in prior_ranks or right not in prior_ranks:
                continue
            left_prior = prior_ranks[left]
            right_prior = prior_ranks[right]
            if left_prior == right_prior:
                continue
            if left_prior < right_prior:
                concordant += 1
            else:
                discordant += 1
    return _tau(concordant, discordant)


def kendall_tau_between(left_ranking: list[str], right_ranking: list[str]) -> float | None:
    left_positions = {participant_id: index for index, participant_id in enumerate(left_ranking)}
    right_positions = {participant_id: index for index, participant_id in enumerate(right_ranking)}
    common = [participant_id for participant_id in left_ranking if participant_id in right_positions]
    concordant = 0
    discordant = 0
    for i, left in enumerate(common):
        for right in common[i + 1 :]:
            left_order = left_positions[left] - left_positions[right]
            right_order = right_positions[left] - right_positions[right]
            if left_order == 0 or right_order == 0:
                continue
            if left_order * right_order > 0:
                concordant += 1
            else:
                discordant += 1
    return _tau(concordant, discordant)


def spearman_rho_against_prior(
    ranking: list[str],
    prior_ranks: dict[str, int | float],
) -> float | None:
    comparable = [participant_id for participant_id in ranking if participant_id in prior_ranks]
    if len(comparable) < 2:
        return None
    predicted = {participant_id: index for index, participant_id in enumerate(comparable, start=1)}
    expected_order = sorted(comparable, key=lambda participant_id: prior_ranks[participant_id])
    expected = {participant_id: index for index, participant_id in enumerate(expected_order, start=1)}
    return _pearson(
        [float(predicted[participant_id]) for participant_id in comparable],
        [float(expected[participant_id]) for participant_id in comparable],
    )


def score_r_squared_against_prior(
    scores: Any,
    prior_scores: dict[str, int | float],
) -> float | None:
    if not isinstance(scores, dict):
        return None
    comparable = [
        participant_id
        for participant_id in scores
        if participant_id in prior_scores and _numeric_rank(scores[participant_id]) is not None
    ]
    if len(comparable) < 2:
        return None
    correlation = _pearson(
        [float(_numeric_rank(scores[participant_id])) for participant_id in comparable],
        [float(prior_scores[participant_id]) for participant_id in comparable],
    )
    return correlation * correlation if correlation is not None else None


def _numeric_rank(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _tau(concordant: int, discordant: int) -> float | None:
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)
