"""Read only receipt-declared members from a verified acquisition tar."""
from hashlib import sha256
from pathlib import Path
import shutil
import tarfile
from oxyformer.provenance import require, relative_artifact_path


def member_hash(archive, member):
    relative_artifact_path(member)
    with tarfile.open(archive, 'r:') as tar:
        info = tar.getmember(member)
        require(info.isfile(), 'payload member must be a regular file')
        with tar.extractfile(info) as stream:
            digest = sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def extract_member(archive, member, destination, expected_hash):
    """Destination is an explicitly assigned private filename; never extractall."""
    relative_artifact_path(member)
    with tarfile.open(archive, 'r:') as tar:
        info = tar.getmember(member)
        require(info.isfile(), 'payload member must be a regular file')
        with tar.extractfile(info) as source, Path(destination).open('xb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
    from oxyformer.provenance import file_hash
    require(file_hash(destination) == expected_hash, 'extracted resource hash mismatch')
