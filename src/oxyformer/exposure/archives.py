"""Read only receipt-declared members from a verified acquisition tar."""
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
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


def gdal_archive_uri(archive, kind):
    """Address local immutable archives independently of their filename suffix.

    Literal braces would confuse GDAL's archive boundary. Its URL-encoded
    /vsicached? wrapper (GDAL >= 3.8) preserves the real path identity with a
    bounded 1 MiB cache per open file, rather than using reusable descriptor
    aliases or copying payloads. The prescribed raster/vector runtimes support
    this wrapper. No filesystem output is created.
    https://gdal.org/en/stable/user/virtual_file_systems.html
    """
    require(kind in ('tar', 'zip'), 'unsupported archive handler')
    path = str(Path(archive).resolve())
    if '{' in path or '}' in path:
        path = '/vsicached?chunk_size=65536&cache_size=1048576&file=' + quote(path, safe='/')
    return f'/vsi{kind}/{{{path}}}'
