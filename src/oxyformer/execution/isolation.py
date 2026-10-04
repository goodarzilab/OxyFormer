"""Linux stage worker: attempt-owned writes without changing upstream permissions.

Landlock enforces content and directory-entry protection in the worker and its
descendants. A seccomp notification supervisor checks metadata operations that
Landlock does not mediate. This protects honest-but-faulty stages, not a hostile
same-UID process racing path resolution or modifying the supervisor's memory.
The CLI forks before importing scientific modules or initializing accelerators.
"""
import array
import ctypes as C
import errno
import os
from pathlib import Path
import platform
import select
import signal
import socket
import stat
import tempfile

from oxyformer.contracts import StageResult
from oxyformer.provenance import require


class Filter(C.Structure):
    _fields_ = [('code', C.c_ushort), ('jt', C.c_ubyte), ('jf', C.c_ubyte), ('k', C.c_uint)]


class Program(C.Structure):
    _fields_ = [('length', C.c_ushort), ('filters', C.POINTER(Filter))]


class Syscall(C.Structure):
    _fields_ = [('number', C.c_int), ('arch', C.c_uint), ('ip', C.c_ulonglong),
               ('args', C.c_ulonglong * 6)]


class Notification(C.Structure):
    _fields_ = [('id', C.c_ulonglong), ('pid', C.c_uint), ('flags', C.c_uint),
               ('call', Syscall)]


class Response(C.Structure):
    _fields_ = [('id', C.c_ulonglong), ('value', C.c_longlong), ('error', C.c_int),
               ('flags', C.c_uint)]


class PathRule(C.Structure):
    _pack_ = 1
    _fields_ = [('access', C.c_ulonglong), ('parent_fd', C.c_int)]


# Native Linux x86-64 syscall numbers: (path argument, dirfd argument, flags
# argument, default no-follow). None path means an fd operation; -1 dirfd means
# the current working directory. The filter rejects other syscall ABIs.
METADATA = {
    90: (0, -1, None, False),     # chmod
    91: (None, 0, None, False),   # fchmod
    92: (0, -1, None, False),     # chown
    93: (None, 0, None, False),   # fchown
    94: (0, -1, None, True),     # lchown
    132: (0, -1, None, False),   # utime
    188: (0, -1, None, False),   # setxattr
    189: (0, -1, None, True),    # lsetxattr
    190: (None, 0, None, False), # fsetxattr
    197: (0, -1, None, False),   # removexattr
    198: (0, -1, None, True),    # lremovexattr
    199: (None, 0, None, False), # fremovexattr
    235: (0, -1, None, False),   # utimes
    260: (1, 0, 4, False),      # fchownat
    261: (1, 0, None, False),   # futimesat (NULL path means fd)
    268: (1, 0, None, False),   # fchmodat
    280: (1, 0, 3, False),      # utimensat (NULL path means fd)
    452: (1, 0, 3, False),      # fchmodat2
}


def checked(value):
    if value < 0:
        error = C.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def restrict_writes(writable_roots):
    """Called only in a fresh single-threaded child, before stage imports."""
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64',
            'stage isolation requires Linux x86_64')
    libc = C.CDLL(None, use_errno=True)
    libc.syscall.restype = C.c_long
    require(checked(libc.syscall(444, 0, 0, 1)) >= 3, 'Landlock ABI 3 or newer required')
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    # WRITE_FILE, REMOVE_*, MAKE_*, REFER, TRUNCATE. Read/execute are unrestricted.
    rights = (1 << 1) | sum(1 << bit for bit in range(4, 15))
    mask = C.c_ulonglong(rights)
    rules = checked(libc.syscall(444, C.byref(mask), C.sizeof(mask), 0))
    try:
        # Device I/O is needed by provisioned accelerators. Grant only existing
        # character devices, never directories, regular files or shared storage.
        paths = [(path, rights) for path in writable_roots]
        for root, _, files in os.walk('/dev'):
            for name in files:
                path = Path(root) / name
                if not path.is_symlink() and stat.S_ISCHR(path.stat().st_mode):
                    paths.append((path, 1 << 1))
        for path, access in paths:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                checked(libc.syscall(445, rules, 1, C.byref(rule), 0))
            finally:
                os.close(fd)
        checked(libc.syscall(446, rules, 0))
    finally:
        os.close(rules)
    # BPF: check architecture, reject x32, notify metadata/ioctl, disallow
    # io_uring (which could otherwise issue uninspected metadata operations).
    instructions = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, 0xc000003e),
                    Filter(0x06, 0, 0, 0x50000 | errno.ENOSYS),
                    Filter(0x20, 0, 0, 0), Filter(0x35, 0, 1, 0x40000000),
                    Filter(0x06, 0, 0, 0x50000 | errno.ENOSYS)]
    for number in (*METADATA, 16, 425):
        action = (0x50000 | errno.ENOSYS) if number == 425 else 0x7fc00000
        instructions.extend([Filter(0x15, 0, 1, number), Filter(0x06, 0, 0, action)])
    instructions.append(Filter(0x06, 0, 0, 0x7fff0000))
    filters = (Filter * len(instructions))(*instructions)
    program = Program(len(filters), filters)
    # Check kernel structure sizes before interpreting notification messages.
    sizes = (C.c_ushort * 3)()
    checked(libc.syscall(317, 3, 0, C.byref(sizes)))
    require(tuple(sizes) == (C.sizeof(Notification), C.sizeof(Response), C.sizeof(Syscall)),
            'unsupported seccomp notification ABI')
    return checked(libc.syscall(317, 1, 8, C.byref(program)))


def worker_path(pid, descriptor):
    descriptor = C.c_int(descriptor).value
    name = 'cwd' if descriptor == -100 else f'fd/{descriptor}'
    value = os.readlink(f'/proc/{pid}/{name}').removesuffix(' (deleted)')
    require(value.startswith('/'), 'unresolvable worker path')
    return Path(value)


def metadata_allowed(notification, writable_roots):
    """Resolve the stopped caller's path, including cwd, dirfd and symlinks."""
    call, pid = notification.call, notification.pid
    args = call.args
    if call.number == 16:  # ioctl: devices and fd-only operations are legitimate.
        mode = os.stat(f'/proc/{pid}/fd/{args[0]}').st_mode
        if stat.S_ISCHR(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            return True
        if args[1] in (0x5451, 0x5450, 0x5421, 0x541b, 0x80086601, 0x80087601):
            return True  # FIOCLEX/FIONCLEX/FIONBIO/FIONREAD, FS_IOC_GETFLAGS/GETVERSION
        target = worker_path(pid, args[0]).resolve()
    else:
        path_arg, fd_arg, flags_arg, nofollow = METADATA[call.number]
        descriptor = -100 if fd_arg == -1 else args[fd_arg]
        if path_arg is None or not args[path_arg]:
            target = worker_path(pid, descriptor).resolve()
        else:
            fd = os.open(f'/proc/{pid}/mem', os.O_RDONLY)
            try:
                data = os.pread(fd, 4096, args[path_arg])
            finally:
                os.close(fd)
            require(b'\0' in data, 'unterminated metadata path')
            value = os.fsdecode(data.split(b'\0', 1)[0])
            path = Path(value)
            if not path.is_absolute():
                path = worker_path(pid, descriptor) / path
            if flags_arg is not None:
                nofollow |= bool(args[flags_arg] & 0x100)  # AT_SYMLINK_NOFOLLOW
            target = path.parent.resolve() / path.name if nofollow else path.resolve()
    return any(target.is_relative_to(root) for root in writable_roots)


def service_notification(listener, writable_roots):
    libc = C.CDLL(None, use_errno=True)
    event = Notification()
    receive = 0xc0000000 | (C.sizeof(event) << 16) | (ord('!') << 8)
    send = 0xc0000000 | (C.sizeof(Response) << 16) | (ord('!') << 8) | 1
    if libc.ioctl(listener, receive, C.byref(event)) < 0:
        if C.get_errno() in (errno.ENOENT, errno.EINTR):
            return
        checked(-1)
    try:
        allowed = metadata_allowed(event, writable_roots)
    except (OSError, ValueError):
        allowed = False
    response = Response(event.id, 0, 0 if allowed else -errno.EACCES, 1 if allowed else 0)
    if libc.ioctl(listener, send, C.byref(response)) < 0 and C.get_errno() != errno.ENOENT:
        checked(-1)


def isolated_stage(request, invoke, dependency_roots):
    """Run a stage with unchanged request paths; return only a serialized result."""
    out = Path(request.output_dir)
    writable_roots = [out]
    # POSIX semaphores/shared memory are transient IPC, not output artifacts or
    # caches. Permit the host's tmpfs only when it cannot contain or hard-link
    # any dependency/attempt file. Persisted artifacts still must be under out.
    shm = Path('/dev/shm')
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    if (not shm.is_symlink() and shm.is_dir()
            and any(line.split()[4] == str(shm) and ' - tmpfs ' in line for line in mounts)
            and all(root.stat().st_dev != shm.stat().st_dev for root in (out, *dependency_roots))):
        writable_roots.append(shm)
    # Output aliases must not grant writes to some other attempt's inode.
    for root, _, files in os.walk(out):
        for name in files:
            path = Path(root) / name
            require(path.is_symlink() or path.stat().st_nlink == 1,
                    'hard-linked file in writable attempt')
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        ready = False
        try:
            os.setsid()
            # Only the protocol socket and safe standard streams survive. In
            # particular, no writable upstream handle can bypass Landlock.
            for name in os.listdir('/proc/self/fd'):
                fd = int(name)
                if fd > 2 and fd != child.fileno():
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            for fd in (0, 1, 2):
                try:
                    path = worker_path(os.getpid(), fd).resolve()
                    if any(path.is_relative_to(root) for root in dependency_roots):
                        null = os.open('/dev/null', os.O_RDWR)
                        os.dup2(null, fd)
                        os.close(null)
                except (OSError, ValueError):
                    pass  # pipes and terminals cannot alias upstream files
            os.chdir(out)
            tempfile.tempdir = None  # discard the parent's cached /tmp choice
            libc = C.CDLL(None, use_errno=True)
            checked(libc.prctl(36, 1, 0, 0, 0))  # PR_SET_CHILD_SUBREAPER
            listener = restrict_writes(writable_roots)
            child.sendmsg([b'L'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                   array.array('i', [listener]))])
            os.close(listener)
            ready = True
            result = invoke()
            require(isinstance(result, StageResult), 'stage did not return StageResult')
        except BaseException as exc:
            result = StageResult(request_hash=request.content_hash,
                                 status='fail' if ready else 'blocked', artifacts=(),
                                 message=str(exc).strip() or type(exc).__name__)
        try:
            # Reap ordinary children and orphaned descendants before the result
            # is returned to the parent for artifact verification/publication.
            while True:
                try:
                    os.waitpid(-1, 0)
                except ChildProcessError:
                    break
            if not ready:
                child.sendall(b'R')
            child.sendall(result.to_json().encode())
        finally:
            child.close()
            os._exit(0)
    child.close()
    listener = None
    chunks = []
    try:
        header, controls, _, _ = parent.recvmsg(1, socket.CMSG_SPACE(array.array('i').itemsize))
        if header == b'L':
            for level, kind, data in controls:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors = array.array('i')
                    descriptors.frombytes(data)
                    require(len(descriptors) == 1, 'unexpected worker listener count')
                    listener = descriptors[0]
            require(listener is not None, 'stage isolation listener missing')
        else:
            require(header == b'R', 'stage worker exited before initialization')
        poller = select.poll()
        poller.register(parent, select.POLLIN)
        if listener is not None:
            poller.register(listener, select.POLLIN)
        while True:
            events = dict(poller.poll())
            if parent.fileno() in events:
                data = parent.recv(65536)
                if not data:
                    break
                chunks.append(data)
            if listener is not None and listener in events:
                # HUP means every restricted task has exited. RECV would block
                # forever here: a readable notification fd is not always data.
                if events[listener] & select.POLLHUP:
                    poller.unregister(listener)
                elif events[listener] & select.POLLIN:
                    service_notification(listener, writable_roots)
        _, status = os.waitpid(pid, 0)
        require(os.waitstatus_to_exitcode(status) == 0, 'stage worker terminated abnormally')
        return StageResult.from_json(b''.join(chunks).decode())
    except BaseException:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        raise
    finally:
        parent.close()
        if listener is not None:
            os.close(listener)
