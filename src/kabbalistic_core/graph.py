"""Project-owned operational Tree-of-Life graph."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.resources import files

from .models import CoreId, NodeKind, Sefirah, World


class InvalidTransitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node: Sefirah
    kind: NodeKind
    world: World
    core: CoreId
    living_function: str
    computational_role: str
    observable_effect: str
    imbalance_signals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PathSpec:
    path_id: str
    source: Sefirah
    target: Sefirah
    transform: str
    world_from: World
    world_to: World
    polarity: str | None = None


class GraphDefinition:
    """Explicit operational graph; not a claim about a canonical 22-path lineage."""

    def __init__(self, version: str, nodes: dict[Sefirah, NodeSpec], paths: tuple[PathSpec, ...]) -> None:
        self.version = version
        self.nodes = dict(nodes)
        self.paths = tuple(paths)
        self._path_index = {(path.source, path.target): path for path in self.paths}
        self.validate()

    @classmethod
    def load_default(cls) -> "GraphDefinition":
        resource = files("kabbalistic_core.data").joinpath("tree_of_life.json")
        data = json.loads(resource.read_text(encoding="utf-8"))
        nodes = {
            Sefirah(item["id"]): NodeSpec(
                node=Sefirah(item["id"]),
                kind=NodeKind(item.get("kind", "sefirah")),
                world=World(item["world"]),
                core=CoreId(item["core"]),
                living_function=item["living_function"],
                computational_role=item["computational_role"],
                observable_effect=item["observable_effect"],
                imbalance_signals=tuple(item["imbalance_signals"]),
            )
            for item in data["nodes"]
        }
        paths = tuple(
            PathSpec(
                path_id=item["id"],
                source=Sefirah(item["source"]),
                target=Sefirah(item["target"]),
                transform=item["transform"],
                world_from=World(item["world_from"]),
                world_to=World(item["world_to"]),
                polarity=item.get("polarity"),
            )
            for item in data["paths"]
        )
        return cls(data["version"], nodes, paths)

    def validate(self) -> None:
        required = set(Sefirah)
        missing = required.difference(self.nodes)
        if missing:
            raise ValueError(f"Graph is missing nodes: {sorted(node.value for node in missing)}")
        if len(self._path_index) != len(self.paths):
            raise ValueError("Graph contains duplicate source/target paths")
        path_ids = [path.path_id for path in self.paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("Graph contains duplicate path IDs")
        if self.nodes[Sefirah.DAAT].kind is not NodeKind.GATE:
            raise ValueError("Da'at must be declared as a gate")
        for path in self.paths:
            if path.source not in self.nodes or path.target not in self.nodes:
                raise ValueError(f"Path references an unknown node: {path.path_id}")
            if self.nodes[path.source].world != path.world_from:
                raise ValueError(f"Path world_from does not match node: {path.path_id}")
            if self.nodes[path.target].world != path.world_to:
                raise ValueError(f"Path world_to does not match node: {path.path_id}")
            if Sefirah.DAAT in (path.source, path.target):
                raise ValueError("Da'at cannot participate in ordinary graph traversal")

        required_edges = set(zip(self.main_cycle, self.main_cycle[1:])) | {
            (Sefirah.MALKHUT, Sefirah.YESOD),
            (Sefirah.YESOD, Sefirah.KETER),
        }
        missing_edges = required_edges.difference(self._path_index)
        if missing_edges:
            rendered = ", ".join(
                f"{source.value}->{target.value}"
                for source, target in sorted(missing_edges, key=lambda pair: (pair[0].value, pair[1].value))
            )
            raise ValueError(f"Graph is missing required operational edges: {rendered}")

        reachable = {Sefirah.KETER}
        changed = True
        while changed:
            changed = False
            for path in self.paths:
                if path.source in reachable and path.target not in reachable:
                    reachable.add(path.target)
                    changed = True
        if Sefirah.MALKHUT not in reachable:
            raise ValueError("Malkhut is not reachable from Keter")

    def transition(self, source: Sefirah, target: Sefirah) -> PathSpec:
        try:
            return self._path_index[(source, target)]
        except KeyError as exc:
            raise InvalidTransitionError(
                f"Undeclared transition in {self.version}: {source.value} -> {target.value}"
            ) from exc

    def node(self, node: Sefirah) -> NodeSpec:
        return self.nodes[node]

    @property
    def main_cycle(self) -> tuple[Sefirah, ...]:
        return (
            Sefirah.KETER,
            Sefirah.CHOKHMAH,
            Sefirah.BINAH,
            Sefirah.CHESED,
            Sefirah.GEVURAH,
            Sefirah.TIFERET,
            Sefirah.NETZACH,
            Sefirah.HOD,
            Sefirah.YESOD,
            Sefirah.MALKHUT,
        )
