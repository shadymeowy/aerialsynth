"""List the DLLs the Windows binaries (or the extensions in wheels) import, and fail unless each
is a system DLL (in System32) or python3.dll: no Visual C++ runtime (vcruntime / msvcp /
api-ms-win-crt-*: the C runtime is linked statically), no HDF5 or zlib DLL.

    python pe_deps.py dist/aerialsynth-*.whl
    python pe_deps.py terrain.exe aerialsynth.dll
"""

import os
import re
import sys
import zipfile

import pefile

SYSTEM32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
FORBIDDEN = re.compile(r"^(vcruntime|msvcp|concrt|ucrtbase|api-ms-win-crt-|.*hdf5|zlib)", re.I)


def binaries(paths):
    for path in paths:
        if path.endswith(".whl"):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.lower().endswith((".pyd", ".dll")):
                        yield f"{path}!{name}", z.read(name)
        else:
            with open(path, "rb") as f:
                yield path, f.read()


def main(paths):
    bad = []
    for name, data in binaries(paths):
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        dlls = sorted({e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])}, key=str.lower)
        print(f"{name} imports:")
        for dll in dlls:
            system = os.path.exists(os.path.join(SYSTEM32, dll))
            ok = (system or dll.lower().startswith("python3")) and not FORBIDDEN.match(dll)
            print(f"  {dll}{'' if ok else '   <-- not a system DLL'}")
            if not ok:
                bad.append(f"{name}: {dll}")
    if bad:
        sys.exit("non-system DLL dependencies:\n  " + "\n  ".join(bad))
    print("only system DLLs")


if __name__ == "__main__":
    main(sys.argv[1:])
