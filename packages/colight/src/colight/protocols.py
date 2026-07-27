"""Base class for state-aware objects in Colight.

Objects that participate in Colight's state management and serialization
system should inherit from the Collector base class.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from colight.widget import CollectedState


class Collector:
    """Base class for objects that participate in state collection.

    Objects implementing this class can collect state information and
    determine their own serialization behavior.
    """

    def collect(self, collector: "CollectedState") -> Any:
        """Collect state information and return final serialized representation.

        Args:
            collector: The CollectedState instance to add state information to

        Returns:
            The final serialized form of this object. Common return values:
            - None: Object disappears from serialization
            - dict: Reference or other JSON-serializable structure
            - list/str/number: Direct JSON values

        Note: This method must return the FINAL serialized form. The result
        will NOT be processed further by to_json().
        """
        raise NotImplementedError("Subclasses must implement collect()")
