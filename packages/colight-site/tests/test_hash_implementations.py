"""Tests comparing hash implementations across modules."""

import hashlib
import libcst as cst
from colight_live.incremental_executor import IncrementalExecutor
from colight_site.model import Block, BlockInterface, Element, TagSet


def create_test_block_with_elements(elements_spec):
    """Create a block with specified elements."""
    elements = []
    for kind, content in elements_spec:
        if kind == "PROSE":
            elements.append(Element(kind=kind, content=content, lineno=1))
        else:  # STATEMENT or EXPRESSION
            module = cst.parse_module(content)
            for stmt in module.body:
                elements.append(Element(kind=kind, content=stmt, lineno=1))

    return Block(
        elements=elements,
        tags=TagSet(),
        start_line=1,
        id=1,
        interface=BlockInterface(provides=[], requires=[]),
    )


def hash_block_content(block):
    """Shared implementation for hashing block content."""
    content_parts = []
    for elem in block.elements:
        content_parts.append(f"{elem.kind}:{elem.get_source()}")
    content = "\n".join(content_parts)
    return hashlib.sha256(content.encode()).hexdigest()


def test_incremental_executor_hash_consistency():
    """Test that IncrementalExecutor's two hash methods are consistent."""
    executor = IncrementalExecutor()

    # Create a test block
    block = create_test_block_with_elements(
        [
            ("PROSE", "# This is a comment"),
            ("STATEMENT", "x = 42"),
            ("EXPRESSION", "x * 2"),
        ]
    )

    # Get content hash using _get_content_hash
    content_hash = executor._get_content_hash(block)

    # Manually compute what _compute_cache_key would compute for content
    content_parts = []
    for elem in block.elements:
        content_parts.append(f"{elem.kind}:{elem.get_source()}")
    content = "\n".join(content_parts)
    manual_hash = hashlib.sha256(content.encode()).hexdigest()

    # They should be identical
    assert content_hash == manual_hash, "Hash methods produce different results"


def test_shared_hash_function_matches_incremental_executor():
    """Test that our shared hash function matches IncrementalExecutor."""
    executor = IncrementalExecutor()

    test_blocks = [
        create_test_block_with_elements([("STATEMENT", "x = 1")]),
        create_test_block_with_elements(
            [
                ("PROSE", "# Comment"),
                ("STATEMENT", "y = 2"),
            ]
        ),
        create_test_block_with_elements(
            [
                ("STATEMENT", "def foo():\n    pass"),
                ("EXPRESSION", "foo()"),
            ]
        ),
    ]

    for block in test_blocks:
        executor_hash = executor._get_content_hash(block)
        shared_hash = hash_block_content(block)

        assert executor_hash == shared_hash, "Hash mismatch for block"


def test_json_generator_vs_incremental_executor_content():
    """Test that json_generator and incremental_executor hash content the same way."""
    # Create blocks with different element types
    blocks = [
        create_test_block_with_elements(
            [
                ("PROSE", "# This is prose"),
                ("STATEMENT", "x = 10"),
            ]
        ),
        create_test_block_with_elements(
            [
                ("EXPRESSION", "2 + 2"),
            ]
        ),
        create_test_block_with_elements(
            [
                ("STATEMENT", "def func():\n    return 42"),
                ("EXPRESSION", "func()"),
                ("PROSE", "# Another comment"),
            ]
        ),
    ]

    executor = IncrementalExecutor()

    for block in blocks:
        # IncrementalExecutor's way
        executor_hash = executor._get_content_hash(block)

        # JSON generator's way (simulated)
        content_parts = []
        for elem in block.elements:
            if elem.kind == "PROSE":
                # JSON generator uses the raw prose text
                content_parts.append(elem.get_source())
            else:
                # JSON generator strips the code
                content_parts.append(elem.get_source().strip())

        json_content = "\n".join(content_parts)
        json_hash = hashlib.sha256(json_content.encode()).hexdigest()

        # These will likely be different due to different formatting
        # This test documents the difference
        print(f"Executor hash: {executor_hash[:16]}")
        print(f"JSON gen hash: {json_hash[:16]}")
        print(f"Content differs: {executor_hash != json_hash}")
        print()


def test_interface_hash_differences():
    """Test how interface hashing differs between implementations."""
    # The JSON generator has special logic for interface hashing
    # that differs from the incremental executor

    block = Block(
        elements=[
            Element(
                kind="STATEMENT",
                content=cst.parse_module("def foo(): pass").body[0],
                lineno=1,
            ),
            Element(
                kind="EXPRESSION", content=cst.parse_module("foo()").body[0], lineno=1
            ),
        ],
        tags=TagSet(),
        start_line=1,
        id=1,
        interface=BlockInterface(provides=["foo"], requires=[]),
    )

    # JSON generator's interface hash logic (simulated)
    interface_parts = []
    for elem in block.elements:
        code = elem.get_source().strip()
        if elem.kind == "STATEMENT":
            # Only includes function/class definitions
            if code.startswith(("def ", "class ", "async def ")):
                parts = code.split()
                if len(parts) > 1:
                    interface_parts.append(f"stmt:{parts[1].split('(')[0]}")
        else:  # EXPRESSION
            interface_parts.append(f"expr:{code}")

    interface_text = "\n".join(sorted(interface_parts))
    json_interface_hash = hashlib.sha256(interface_text.encode()).hexdigest()[:16]

    print(f"Interface parts: {interface_parts}")
    print(f"Interface hash: {json_interface_hash}")

    # The incremental executor doesn't have a separate interface hash
    # It includes everything in the content hash


def test_propose_unified_hash_function():
    """Test a proposed unified hash function for block content."""

    def unified_hash_block_content(block, include_kind=True):
        """Unified hash function for block content.

        Args:
            block: The block to hash
            include_kind: Whether to include element kind in hash (for stricter matching)
        """
        content_parts = []
        for elem in block.elements:
            source = elem.get_source().strip()  # Always strip for consistency
            if include_kind:
                content_parts.append(f"{elem.kind}:{source}")
            else:
                content_parts.append(source)

        content = "\n".join(content_parts)
        return hashlib.sha256(content.encode()).hexdigest()

    # Test with various blocks
    block = create_test_block_with_elements(
        [
            ("PROSE", "  # Indented comment  "),
            ("STATEMENT", "x = 1"),
            ("EXPRESSION", "x + 1"),
        ]
    )

    hash_with_kind = unified_hash_block_content(block, include_kind=True)
    hash_without_kind = unified_hash_block_content(block, include_kind=False)

    # These should be different
    assert hash_with_kind != hash_without_kind

    print(f"Hash with kind: {hash_with_kind[:16]}")
    print(f"Hash without kind: {hash_without_kind[:16]}")
