"""Small runtime-only dataclass support that avoids importing stdlib re."""
from __future__ import annotations

from collections.abc import Callable


_MISSING = object()

import sys

class _FieldDefault:
    __slots__ = ("factory",)

    def __init__(self, factory: Callable[[], object]):
        self.factory = factory


def field(*, default_factory: Callable[[], object]) -> _FieldDefault:
    return _FieldDefault(default_factory)


class _FactoryMarker:
    def __repr__(self) -> str:
        return "<factory>"


_FACTORY_MARKER = _FactoryMarker()


class FrozenInstanceError(AttributeError):
    pass


def dataclass(
    *, frozen: bool = False, repr: bool = True
) -> Callable[[type], type]:
    def decorate(record: type) -> type:
        names = tuple(record.__annotations__)
        factories: dict[str, Callable[[], object]] = {}
        signature = ["self"]
        namespace: dict[str, object] = vars(sys.modules[record.__module__]).copy()
        for index, name in enumerate(names):
            default = record.__dict__.get(name, _MISSING)
            if isinstance(default, _FieldDefault):
                factories[name] = default.factory
                signature.append(f"{name}=_factory_marker")
            elif default is _MISSING:
                signature.append(name)
            else:
                namespace[f"_default_{index}"] = default
                signature.append(f"{name}=_default_{index}")

        if "__init__" not in record.__dict__:
            namespace["_factory_marker"] = _FACTORY_MARKER
            namespace["_factories"] = factories
            init_lines = [f"def __init__({', '.join(signature)}):"]
            for name in names:
                if name in factories:
                    init_lines.append(
                        f"    if {name} is _factory_marker: "
                        f"{name} = _factories[{name!r}]()"
                    )
                if frozen:
                    init_lines.append(
                        f"    object.__setattr__(self, {name!r}, {name})"
                    )
                else:
                    init_lines.append(f"    self.{name} = {name}")
            if "__post_init__" in record.__dict__:
                init_lines.append("    self.__post_init__()")
            else:
                init_lines.append("    pass")
            exec("\n".join(init_lines), namespace)
            initializer = namespace["__init__"]
            initializer.__annotations__ = {
                name: record.__annotations__[name] for name in names
            }
            initializer.__annotations__["return"] = None
            setattr(record, "__init__", initializer)
        setattr(record, "__match_args__", names)

        if repr:
            def record_repr(self: object) -> str:
                values = ", ".join(
                    f"{name}={getattr(self, name)!r}" for name in names
                )
                return f"{type(self).__qualname__}({values})"

            setattr(record, "__repr__", record_repr)

        def record_eq(self: object, other: object) -> object:
            if type(self) is not type(other):
                return NotImplemented
            return all(getattr(self, name) == getattr(other, name) for name in names)

        setattr(record, "__eq__", record_eq)
        if frozen:
            def frozen_setattr(self: object, name: str, value: object) -> None:
                raise FrozenInstanceError(f"cannot assign to field {name!r}")

            def frozen_delattr(self: object, name: str) -> None:
                raise FrozenInstanceError(f"cannot delete field {name!r}")

            setattr(record, "__setattr__", frozen_setattr)
            setattr(record, "__delattr__", frozen_delattr)
            setattr(
                record,
                "__hash__",
                lambda self: hash(tuple(getattr(self, name) for name in names)),
            )
        else:
            setattr(record, "__hash__", None)

        for name in factories:
            if hasattr(record, name):
                delattr(record, name)
        return record

    return decorate
