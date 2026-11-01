from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndexProgressEvent:
    phase: str
    document_name: str
    document_index: int
    documents_total: int
    complete: bool = False
    completed_units: int = 0
    total_units: int = 0
    unit: str = ""
    detail: str = ""

    @property
    def stage_fraction(self) -> float:
        if self.complete:
            return 1.0
        if self.total_units:
            return max(0.0, min(self.completed_units / self.total_units, 1.0))
        return 0.0

    @property
    def overall_fraction(self) -> float:
        if not self.documents_total:
            return 0.0
        if self.phase == "hashing":
            return 0.1 * (self.document_index - 1 + self.stage_fraction) / self.documents_total
        stages = {"reading": (0.0, 0.15), "chunking": (0.15, 0.35),
                  "embedding": (0.5, 0.45), "writing": (0.95, 0.05), "skipped": (1.0, 0.0)}
        start, weight = stages.get(self.phase, (0.0, 0.0))
        return min(1.0, max(0.1, 0.1 + 0.9 * (
            self.document_index - 1 + start + weight * self.stage_fraction
        ) / self.documents_total))


@dataclass
class IndexProgressTracker:
    latest: IndexProgressEvent | None = None
    overall_fraction: float = 0.0
    samples: list[tuple[int, float]] = field(default_factory=list)

    def observe(self, event: IndexProgressEvent, timestamp: float) -> None:
        previous = self.latest
        if previous is None or (event.phase, event.document_index, event.unit) != (
            previous.phase, previous.document_index, previous.unit
        ):
            self.samples.clear()
        self.latest = event
        self.overall_fraction = max(self.overall_fraction, event.overall_fraction)
        # Completed embedding batches, excluding the first model-loading batch,
        # give a more useful estimate than timing queue consumption in the UI.
        if event.unit == "passages" and event.completed_units > 0 and (
            not self.samples or event.completed_units > self.samples[-1][0]
        ):
            self.samples.append((event.completed_units, timestamp))
            self.samples[:] = self.samples[-8:]

    def throughput(self, now: float) -> float | None:
        if len(self.samples) < 3:
            return None
        first_units, first_time = self.samples[0]
        last_units, last_time = self.samples[-1]
        # A stalled batch should lower the displayed throughput rather than leave
        # a stale optimistic estimate on screen.
        duration = max(last_time, now) - first_time
        return (last_units - first_units) / duration if duration > 0 else None

    def remaining_seconds(self, now: float) -> float | None:
        event = self.latest
        rate = self.throughput(now)
        if event is None or not rate or event.completed_units >= event.total_units:
            return None
        return (event.total_units - event.completed_units) / rate
