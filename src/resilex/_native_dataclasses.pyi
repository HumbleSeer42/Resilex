from collections.abc import Callable
from typing import Any, TypeVar, dataclass_transform

_T = TypeVar("_T")

class FrozenInstanceError(AttributeError): ...

def field(*, default_factory: Callable[[], Any]) -> Any: ...

@dataclass_transform(field_specifiers=(field,))
def dataclass(*, frozen: bool = ..., repr: bool = ...) -> Callable[[_T], _T]: ...
