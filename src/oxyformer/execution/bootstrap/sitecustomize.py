"""Runner-owned startup hook inherited by standard Python child interpreters.

Only worker.execute adds this directory to PYTHONPATH. The parent request pins
both values; scientific imports must be checked before multiprocessing unpickles
its target (including forkserver preloads), not just in the stage entry point.
"""
import os
import traceback
import uuid
from multiprocessing import connection

try:
    from oxyformer.execution.imports import install_tracked_imports
    install_tracked_imports(os.environ['OXYFORMER_IMPORT_REPO'],
                            os.environ['OXYFORMER_IMPORT_COMMIT'])
except BaseException:
    # Python normally prints a sitecustomize exception and continues startup.
    # That would silently discard the import check in precisely the failing case.
    traceback.print_exc()
    os._exit(1)


# Linux filesystem socket names have a 108-byte limit. Attempt roots can be
# longer before multiprocessing adds its temporary directory/socket suffix.
# Its standard authenticated AF_UNIX connections already support abstract
# addresses; use those for overlong generated names without writing elsewhere.
_original_address = connection.arbitrary_address


def _attempt_address(family):
    address = _original_address(family)
    if family == 'AF_UNIX' and len(os.fsencode(address)) >= 108:
        return '\0oxyformer-' + uuid.uuid4().hex
    return address


connection.arbitrary_address = _attempt_address
