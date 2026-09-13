import tempfile
from pathlib import Path

from train.training import available_memory_mib


def main():
    with tempfile.TemporaryDirectory() as directory:
        meminfo = Path(directory) / "meminfo"
        meminfo.write_text(
            "MemTotal:       16384000 kB\nMemAvailable:    2098176 kB\n",
            encoding="ascii",
        )
        assert available_memory_mib(meminfo) == 2049

        meminfo.write_text("MemTotal: 16384000 kB\n", encoding="ascii")
        assert available_memory_mib(meminfo) is None

    print("memory guard tests passed")


if __name__ == "__main__":
    main()
