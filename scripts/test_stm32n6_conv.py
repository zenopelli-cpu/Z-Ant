#!/usr/bin/env python3
"""Compile and run the STM32 N6 convolution shim with clang for smoke testing."""

from __future__ import annotations

import ctypes
import math
import shutil
import ctypes
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"
CONV_SOURCES = [
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
]
CMSIS_STUB = REPO_ROOT / "tests/fixtures/cmsis_stub"
CMSIS_STUB_SOURCES = (
    CMSIS_STUB / "arm_convolve_s8.c",
    CMSIS_STUB / "arm_dot_prod_f32.c",
)


def build_shared_object(
    tmpdir: Path,
    *,
    label: str = "default",
    macros: tuple[str, ...] = (),
    extra_sources: tuple[Path, ...] = (),
    include_dirs: tuple[Path, ...] = (),
) -> Path:
    lib_path = tmpdir / f"libstm32n6_conv_{label}.so"
    compiler = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("no C compiler (clang/cc/gcc) found in PATH")
    compile_cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c11",
        "-o",
        str(lib_path),
    ]
    for directory in include_dirs:
        compile_cmd.extend(["-I", str(directory)])
    for macro in macros:
        compile_cmd.append(f"-D{macro}")
    for src in CONV_SOURCES:
        compile_cmd.append(str(src))
    for src in extra_sources:
        compile_cmd.append(str(src))
    subprocess.run(compile_cmd, check=True)
    return lib_path


def _prepare_common(lib: ctypes.CDLL) -> dict[str, ctypes._CFuncPtr]:
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_size_p = ctypes.POINTER(ctypes.c_size_t)

    signature = [
        c_float_p,
        c_size_p,
        c_float_p,
        c_size_p,
        c_float_p,
        c_size_p,
        c_float_p,
        ctypes.c_size_t,
        c_size_p,
        c_size_p,
        c_size_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]

    def configure(name: str) -> ctypes._CFuncPtr:
        func = getattr(lib, name)
        func.restype = ctypes.c_bool
        func.argtypes = signature
        return func

    lib.zant_stm32n6_reset_test_state.restype = None
    lib.zant_stm32n6_reset_test_state.argtypes = []
    lib.zant_stm32n6_cmsis_was_used.restype = ctypes.c_bool
    lib.zant_stm32n6_cmsis_was_used.argtypes = []
    lib.zant_stm32n6_cmsis_invocation_count.restype = ctypes.c_size_t
    lib.zant_stm32n6_cmsis_invocation_count.argtypes = []
    lib.zant_stm32n6_ethos_was_used.restype = ctypes.c_bool
    lib.zant_stm32n6_ethos_was_used.argtypes = []

    funcs = {
        "reference": configure("zant_stm32n6_conv_f32"),
        "helium": getattr(lib, "zant_stm32n6_conv_f32_helium", None),
        "ethos": getattr(lib, "zant_stm32n6_conv_f32_ethos", None),
        "reset": lib.zant_stm32n6_reset_test_state,
        "cmsis_flag": lib.zant_stm32n6_cmsis_was_used,
        "cmsis_count": lib.zant_stm32n6_cmsis_invocation_count,
        "cmsis_selftest": getattr(lib, "zant_stm32n6_cmsis_s8_selftest", None),
        "ethos_flag": lib.zant_stm32n6_ethos_was_used,
    }

    if funcs["helium"] is not None:
        funcs["helium"].restype = ctypes.c_bool
        funcs["helium"].argtypes = signature
    if funcs["ethos"] is not None:
        funcs["ethos"].restype = ctypes.c_bool
        funcs["ethos"].argtypes = signature
    if funcs["cmsis_selftest"] is not None:
        funcs["cmsis_selftest"].restype = ctypes.c_bool
        funcs["cmsis_selftest"].argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

    tensors = {
        "input": (ctypes.c_float * 9)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        "input_shape": (ctypes.c_size_t * 4)(1, 1, 3, 3),
        "weights": (ctypes.c_float * 4)(1.0, 0.0, 0.0, 1.0),
        "weight_shape": (ctypes.c_size_t * 4)(1, 1, 2, 2),
        "output": (ctypes.c_float * 4)(),
        "output_shape": (ctypes.c_size_t * 4)(1, 1, 2, 2),
        "stride": (ctypes.c_size_t * 2)(1, 1),
        "pads": (ctypes.c_size_t * 4)(0, 0, 0, 0),
        "dilations": (ctypes.c_size_t * 2)(1, 1),
    }

    funcs["args"] = (
        ctypes.cast(tensors["input"], c_float_p),
        ctypes.cast(tensors["input_shape"], c_size_p),
        ctypes.cast(tensors["weights"], c_float_p),
        ctypes.cast(tensors["weight_shape"], c_size_p),
        ctypes.cast(tensors["output"], c_float_p),
        ctypes.cast(tensors["output_shape"], c_size_p),
        ctypes.POINTER(ctypes.c_float)(),
        ctypes.c_size_t(0),
        ctypes.cast(tensors["stride"], c_size_p),
        ctypes.cast(tensors["pads"], c_size_p),
        ctypes.cast(tensors["dilations"], c_size_p),
        ctypes.c_size_t(1),
        ctypes.c_size_t(1),
        ctypes.c_size_t(1),
    )
    funcs["output_tensor"] = tensors["output"]
    funcs["cmsis_selftest_output"] = (ctypes.c_float * 4)()
    return funcs


def _check_output(output_tensor, *, abs_tol: float = 1e-6, rel_tol: float = 1e-6) -> None:
    expected = (6.0, 8.0, 12.0, 14.0)
    for idx, value in enumerate(expected):
        if not math.isclose(output_tensor[idx], value, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(
                f"mismatch at index {idx}: got {output_tensor[idx]!r}, expected {value!r}"
            )


def run_reference(tmpdir: Path) -> None:
    lib_path = build_shared_object(tmpdir, label="reference")
    lib = ctypes.CDLL(str(lib_path))
    funcs = _prepare_common(lib)
    funcs["reset"]()
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS invocation counter should reset to zero")
    if not funcs["reference"](*funcs["args"]):
        raise RuntimeError("reference convolution failed")
    _check_output(funcs["output_tensor"])
    if funcs["cmsis_flag"]():
        raise AssertionError("CMSIS path should not be marked for reference build")
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS call count should be zero for reference build")
    if funcs["ethos_flag"]():
        raise AssertionError("Ethos path should not be marked for reference build")


def run_helium(tmpdir: Path) -> None:
    lib_path = build_shared_object(
        tmpdir,
        label="helium",
        macros=("ZANT_HAS_CMSIS_DSP=1",),
        extra_sources=CMSIS_STUB_SOURCES,
        include_dirs=(CMSIS_STUB,),
    )
    lib = ctypes.CDLL(str(lib_path))
    funcs = _prepare_common(lib)
    funcs["reset"]()
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS invocation counter should reset to zero")
    if funcs["helium"] is None:
        raise AssertionError("helium symbol missing")
    if not funcs["helium"](*funcs["args"]):
        raise RuntimeError("CMSIS convolution failed")
    _check_output(funcs["output_tensor"], abs_tol=1e-1, rel_tol=1e-2)
    if funcs["cmsis_flag"]():
        raise AssertionError("CMSIS flag should remain false before self-test")
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS call count should be zero before self-test")

    if funcs["cmsis_selftest"] is None:
        raise AssertionError("CMSIS s8 self-test symbol missing")
    output = funcs["cmsis_selftest_output"]
    if not funcs["cmsis_selftest"](
        ctypes.cast(output, ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(len(output)),
    ):
        raise RuntimeError("CMSIS s8 self-test failed")
    _check_output(output)
    if not funcs["cmsis_flag"]():
        raise AssertionError("CMSIS flag not raised after self-test")
    if funcs["cmsis_count"]() == 0:
        raise AssertionError("CMSIS call count not incremented after self-test")
    if funcs["ethos_flag"]():
        raise AssertionError("Ethos flag should be false during Helium test")


def run_ethos(tmpdir: Path) -> None:
    lib_path = build_shared_object(
        tmpdir,
        label="ethos",
        macros=("ZANT_HAS_ETHOS_U=1", "ZANT_HAS_CMSIS_DSP=1"),
        extra_sources=CMSIS_STUB_SOURCES,
        include_dirs=(CMSIS_STUB,),
    )
    lib = ctypes.CDLL(str(lib_path))
    funcs = _prepare_common(lib)
    funcs["reset"]()
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS invocation counter should reset to zero")
    if funcs["ethos"] is None:
        raise AssertionError("ethos symbol missing")
    if not funcs["ethos"](*funcs["args"]):
        raise RuntimeError("Ethos convolution failed")
    _check_output(funcs["output_tensor"])
    if not funcs["ethos_flag"]():
        raise AssertionError("Ethos flag not raised")
    if funcs["cmsis_count"]() != 0:
        raise AssertionError("CMSIS call count should remain zero in Ethos test")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        run_reference(tmpdir)
        run_helium(tmpdir)
        run_ethos(tmpdir)
    print("STM32N6 convolution shim passed comprehensive smoke tests")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"failed to build test shim: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - surface any failure clearly
        print(exc, file=sys.stderr)
        sys.exit(1)
