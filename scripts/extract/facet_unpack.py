from pathlib import Path


def launch(src_dir: Path) -> None:
    from samplelib import PackedFaceset
    PackedFaceset.unpack(src_dir)
