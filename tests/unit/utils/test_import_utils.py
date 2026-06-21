from abc import abstractmethod
from dataclasses import dataclass

from openhands.app_server.utils.import_utils import get_impl


class Shape:
    @abstractmethod
    def get_area(self):
        """Get the area of this shape"""


@dataclass
class Square(Shape):
    length: float

    def get_area(self):
        return self.length**2


def test_get_impl():
    ShapeImpl = get_impl(Shape, f'{Shape.__module__}.Square')
    shape = ShapeImpl(5)
    assert shape.get_area() == 25
