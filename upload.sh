from google.cloud import storage
import sys
import os
from pathlib import Path

def upload_file(bucket, source_file_path, destination_blob_name):
    """Uploads a single file to the given GCS bucket."""
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_path)
        print(f"Uploaded {source_file_path} -> gs://{bucket.name}/{destination_blob_name}")
    except Exception as e:
        print(f"Error uploading {source_file_path} to gs://{bucket.name}/{destination_blob_name}: {e}")
        raise

def upload_directory(bucket, source_dir, destination_prefix):
    """Recursively uploads a directory to GCS under the given prefix."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Source {source_dir} is not a directory")
    for root, _, files in os.walk(source_dir):
        root_path = Path(root)
        for fname in files:
            local_path = root_path / fname
            rel_path = local_path.relative_to(source_dir).as_posix()
            if destination_prefix:
                blob_name = f"{destination_prefix.rstrip('/')}/{rel_path}"
            else:
                blob_name = rel_path
            upload_file(bucket, str(local_path), blob_name)

def find_latest_subdir(project_name):
    base = Path("out") / project_name
    if not base.exists() or not base.is_dir():
        return None
    # find immediate subdirectories
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # sort by name and pick the last (replicates `ls | sort | tail -n1`)
    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    return subdirs_sorted[-1]

def read_object_id_from_gcskey():
    key_path = Path.home() / "gcskey"
    if not key_path.exists():
        return None
    try:
        content = key_path.read_text().strip()
    except Exception:
        return None
    return content if content else None

def main(argv):
    if len(argv) not in (2, 3):
        print("Usage: python3 script.py <ProjectName> [bucket_name]")
        sys.exit(1)

    project_name = argv[1]
    bucket_name = argv[2] if len(argv) == 3 else "swaptube-out"

    destination_prefix = read_object_id_from_gcskey()
    if destination_prefix is None:
        print("Error: object ID not found. Expected a non-empty file at ~/gcskey containing the destination prefix.")
        sys.exit(1)

    ultimate_subdir = find_latest_subdir(project_name)
    if ultimate_subdir is None:
        print(f"No output directory found for project {project_name}. Expected directories under out/{project_name}/")
        sys.exit(1)

    storage_client = storage.Client()
    try:
        bucket = storage_client.bucket(bucket_name)
        # validate bucket exists by fetching metadata
        if not bucket.exists():
            print(f"Error: bucket '{bucket_name}' does not exist or is not accessible.")
            sys.exit(1)
    except Exception as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        sys.exit(1)

    try:
        upload_directory(bucket, ultimate_subdir, destination_prefix)
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

    print("Upload completed successfully.")

if __name__ == "__main__":
    main(sys.argv)
