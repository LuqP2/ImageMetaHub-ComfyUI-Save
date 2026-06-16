"""
Compatibility shim for MetaHub Save Node metadata helpers.

The implementation is kept in metadata_utils_impl.py; this module exposes the
same public API expected by the save nodes while pinning the published node
version to 1.1.3.
"""

try:
    from . import metadata_utils_impl as _impl
except ImportError:
    import metadata_utils_impl as _impl

METAHUB_SAVE_NODE_VERSION = "1.1.3"
_impl.METAHUB_SAVE_NODE_VERSION = METAHUB_SAVE_NODE_VERSION

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

globals()["METAHUB_SAVE_NODE_VERSION"] = METAHUB_SAVE_NODE_VERSION
__all__ = [name for name in globals() if not name.startswith("_")]
