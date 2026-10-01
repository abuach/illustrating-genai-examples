"""Models writing the code under the code.

The Efficiency chapter of *Programming Generative AI* made a model run fast.
This one makes the model *write* fast code: hand it a slow numerical routine and
ask it to optimize it without breaking it. The catch is that "optimize" hides
two demands at once, and they pull against each other. A kernel that's fast but
wrong is worthless, and a kernel that's correct but slow didn't do its job. So
we measure both halves: correctness against a trusted reference, and the real
speedup against the baseline we're trying to beat.

The routine is the pairwise squared-distance matrix, the inner loop under every
nearest-neighbour search and clustering pass (and under the retrieval the
Augmentation chapter leaned on). Three implementations matter. ``reference`` is
the obvious, plainly-correct version, used as the correctness oracle. ``naive``
is the same idea written the way most people first reach for it. ``baseline`` is
the optimized library version, the bar a generated kernel actually has to clear,
the local stand-in for the heavily-tuned library a GPU kernel has to beat.

The code model is qwen2.5-coder; gpt-oss reasons at length but rarely emits a
clean compiled kernel. Timing is hardware-dependent and runs are
nondeterministic, so capture the KERNEL_* constants once on a quiet machine with
scripts/_lowlevel_probe.py and freeze the notebook cells.
"""
import importlib.util
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import ollama

KERNEL_MODEL = "qwen2.5-coder:latest"


# ── The routine, three ways ───────────────────────────────────────────────────

def make_data(n: int = 600, d: int = 96, seed: int = 0):
    """A random point cloud: n rows of d numbers each."""
    return np.random.default_rng(seed).standard_normal((n, d))


def reference(X):
    """Plainly correct, used only to check the model's answer (never timed)."""
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def naive(X):
    """The same broadcast most people write first. The easy bar to beat."""
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def baseline(X):
    """The optimized library version: the squared-norm identity over a matrix
    multiply, which hands the heavy lifting to multithreaded BLAS. This is the
    real bar, the local stand-in for the tuned library a GPU kernel must beat."""
    g = (X * X).sum(1)
    return g[:, None] + g[None, :] - 2.0 * (X @ X.T)


# ── Compiling and benching a model-written kernel ─────────────────────────────

def compile_kernel(code: str):
    """Write the model's code to a real module file and import its ``kernel``.

    A real file path (not exec of a string) is what lets numba's @njit(cache=...)
    work and mirrors the chapter's whole point: the model's code becomes the code
    under the code. Controlled local demo, same posture as run_z3 in reason.py."""
    path = Path(tempfile.gettempdir()) / f"_kernel_{uuid.uuid4().hex}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # noqa: S102 - controlled demo
    return mod.kernel


def _best_ms(fn, X, repeats: int = 7, slow_ms: float = 250.0) -> float:
    """Best wall-clock of ``repeats`` runs, in ms. Warm up first so a JIT
    compile (numba's first call) never lands inside the timed window. A candidate
    that runs slower than ``slow_ms`` (an un-compiled Python loop) is timed once
    and returned, so a hopeless kernel can't stall the whole search."""
    fn(X)
    t0 = time.perf_counter()
    fn(X)
    first = (time.perf_counter() - t0) * 1000
    if first > slow_ms:
        return first
    best = first
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def bench(code: str, n: int = 600, d: int = 96, seed: int = 0) -> dict:
    """Compile a model kernel and score it on the two axes that matter.

    Returns correctness (does it match the reference?) and speedup against both
    bars: the optimized ``baseline`` (the one that counts) and the ``naive``
    version (for context). A kernel that won't compile or crashes is an honest
    miss, not an error that stops the search."""
    X = make_data(n, d, seed)
    ref = reference(X)
    opt_ms = _best_ms(baseline, X)
    naive_ms = _best_ms(naive, X)
    try:
        kernel = compile_kernel(code)
        out = kernel(X)
        correct = bool(np.allclose(out, ref, atol=1e-6))
        cand_ms = _best_ms(kernel, X)
    except Exception as exc:
        return {"correct": False, "compiled": False, "error": _short(exc),
                "speedup": 0.0, "vs_naive": 0.0,
                "baseline_ms": round(opt_ms, 2), "candidate_ms": None}
    return {"correct": correct, "compiled": True, "error": None,
            "speedup": round(opt_ms / cand_ms, 2),
            "vs_naive": round(naive_ms / cand_ms, 2),
            "baseline_ms": round(opt_ms, 2),
            "naive_ms": round(naive_ms, 2),
            "candidate_ms": round(cand_ms, 2)}


def _short(exc) -> str:
    s = str(exc).strip().splitlines()
    return (s[0] if s else repr(exc))[:90]


# ── Asking the model for a kernel, and the search loop ────────────────────────

_TASK = (
    "Write an optimized Python function `kernel(X)` that returns the pairwise "
    "squared Euclidean distance matrix of the rows of a 2-D NumPy array X: "
    "result[i, j] = sum_k (X[i, k] - X[j, k]) ** 2. It must match this reference "
    "exactly:  ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1).  Make it as fast "
    "as you can. You may use numba (from numba import njit, prange). Define numpy "
    "as np. Output only the Python code, nothing else.")


def _emit(model: str, prompt: str) -> str:
    """Ask for kernel code; strip any markdown fence around it."""
    msg = ollama.chat(model=model, options={"num_predict": 600},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = msg.get("content") or ""
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (fenced.group(1) if fenced else text).strip()


def propose_kernel(model: str = KERNEL_MODEL) -> tuple:
    """One shot: the model writes a kernel, we bench it. Returns (code, result)."""
    code = _emit(model, _TASK)
    return code, bench(code)


def optimize_kernel(model: str = KERNEL_MODEL, max_tries: int = 4) -> list:
    """Generate, profile, regenerate: feed each result back and ask for faster.

    The model sees what went wrong (won't compile / wrong answer / correct but
    slower than the baseline and by how much) and tries again. Returns the trace,
    one row per attempt, so the search can be charted."""
    trace, feedback = [], ""
    for attempt in range(1, max_tries + 1):
        code = _emit(model, _TASK + feedback)
        r = bench(code)
        trace.append({"attempt": attempt, "code": code, "correct": r["correct"],
                      "speedup": r["speedup"], "error": r["error"]})
        if not r["compiled"]:
            feedback = (f"\n\nYour code didn't compile: {r['error']}. Import "
                        "everything you use (from numba import njit, prange) and "
                        "pass fastmath=True as a boolean, not a set. Fix it.")
        elif not r["correct"]:
            feedback = ("\n\nYour kernel ran but its answer was wrong (it didn't "
                        "match the reference). Fix the math, keep it fast.")
        elif r["speedup"] >= 1.0:
            break
        else:
            feedback = (f"\n\nCorrect, but only {r['speedup']}x the baseline's "
                        "speed (slower). The baseline uses a matrix multiply "
                        "across all CPU cores. Make it faster: decorate with "
                        "@njit(parallel=True, fastmath=True) and write the outer "
                        "loop as `for i in prange(n)` (from numba import prange).")
    return trace


def fast1(model: str = KERNEL_MODEL, n: int = 20) -> dict:
    """Local fast_1: over n one-shot attempts, how many are correct, and how many
    are correct AND faster than the optimized baseline."""
    correct = beat = 0
    for _ in range(n):
        _, r = propose_kernel(model)
        correct += int(r["correct"])
        beat += int(r["correct"] and r["speedup"] >= 1.0)
    return {"n": n, "correct": correct, "beat": beat,
            "fast1_pct": round(100 * beat / n)}


# ── Baked from scripts/_lowlevel_probe.py (qwen2.5-coder + numba) ──────────────
# A real one-shot, kept verbatim. The model didn't reach for a timid serial loop:
# it parallelized the outer loop with prange and halved the work by exploiting
# symmetry (each pair computed once, then mirrored). It's correct and beats the
# naive broadcast nearly twelve-fold, and it still loses to the tuned BLAS
# baseline five-fold. Correct and fast-looking isn't the same as faster.
KERNEL_ONESHOT = {
    "code": (
        "import numpy as np\n"
        "from numba import njit, prange\n\n"
        "@njit(parallel=True)\n"
        "def kernel(X):\n"
        "    n = X.shape[0]\n"
        "    result = np.zeros((n, n))\n"
        "    for i in prange(n):\n"
        "        for j in range(i + 1, n):\n"
        "            diff = X[i] - X[j]\n"
        "            dist = 0.0\n"
        "            for k in range(diff.shape[0]):\n"
        "                dist += diff[k] ** 2\n"
        "            result[i, j] = dist\n"
        "            result[j, i] = dist\n"
        "    return result"),
    "correct": True,
    "vs_naive": 11.78,
    "speedup": 0.2,
    "naive_ms": 40.4,
    "baseline_ms": 0.68,
    "candidate_ms": 3.43,
}

# A real search trace: speedup against the optimized BLAS baseline per attempt,
# every attempt correct this run. The model tries parallelism, fastmath, and a
# different inner product each time, peaking at 0.97x (a whisker short) and
# bouncing around in between. It never crosses the line: against a multithreaded
# matrix multiply, an iterating code model gets close and stalls.
KERNEL_SEARCH = [
    {"attempt": 1, "correct": True, "speedup": 0.6,  "note": "parallel, scalar inner"},
    {"attempt": 2, "correct": True, "speedup": 0.97, "note": "+ fastmath, vectorized"},
    {"attempt": 3, "correct": True, "speedup": 0.17, "note": "np.empty, rewrote inner"},
    {"attempt": 4, "correct": True, "speedup": 0.56, "note": "more fastmath flags"},
    {"attempt": 5, "correct": True, "speedup": 0.25, "note": "dot product"},
]

# Local results over 8 one-shots, with the cited frontier number for context. Of
# eight tries, three compiled and matched the reference (the rest tripped on
# numba's API), and none of those also beat the tuned baseline. The frontier bar
# is from KernelBench (Ouyang et al. 2025): the best general models write a kernel
# that is correct AND faster than the baseline on under a fifth of GPU tasks.
# Caption must label the frontier bar as cited, not measured here.
KERNEL_REALITY = {
    "correct (local)": 38,
    "correct & faster (local)": 0,
    "frontier, KernelBench": 20,
}


def show_kernel_demo(demo: dict = KERNEL_ONESHOT):
    """The one-shot: the model's real kernel, then its measured scorecard."""
    from genai.agent import show_code, show_turn
    show_turn("you", "Optimize the pairwise squared-distance routine. "
                     "Make it fast, keep it correct.")
    show_code("qwen2.5-coder", demo["code"])
    verdict = "correct" if demo["correct"] else "WRONG"
    show_turn("BENCH", f"{verdict}; {demo['vs_naive']}x faster than the naive "
                       f"version, but {demo['speedup']}x the optimized baseline "
                       f"({demo['candidate_ms']} ms vs {demo['baseline_ms']} ms). "
                       "Correct, and still slower than the library.")


# ── Fast on the big matrix, slower on the small one ───────────────────────────
# The scoreboard timed the model's kernel at one size and reported one number. But
# the kernel is *parallel* (the model reached for prange), and parallelism isn't
# free: spawning threads is a fixed cost paid on every call, worth it only when
# there's enough work to spread. So the kernel's speed against its own serial twin
# isn't a number, it's a curve that crosses. On a big matrix the parallel version
# wins several-fold; on a small one the thread-launch cost dwarfs the work and the
# same kernel run serially is many times faster. The single-size benchmark sat on
# one side of that crossover. Timing is hardware-dependent, so SIZE_CURVE is
# captured once by scripts/_lowlevel_probe.py and the cell frozen.

# The model's kernel run serially: the same code with the parallel decorator and
# prange swapped for a plain njit loop, so the only thing measured is the
# parallelism itself.
SERIAL_ONESHOT = (KERNEL_ONESHOT["code"]
                  .replace("from numba import njit, prange", "from numba import njit")
                  .replace("@njit(parallel=True)", "@njit")
                  .replace("for i in prange(n)", "for i in range(n)"))


def size_curve(ns=(16, 64, 128, 600, 2048), d: int = 96) -> list:
    """Time the model's parallel kernel and its serial twin across matrix sizes.
    Returns one row per size with each version's best milliseconds, so the winner
    can be read off. Hardware-dependent; capture once and bake."""
    parallel = compile_kernel(KERNEL_ONESHOT["code"])
    serial = compile_kernel(SERIAL_ONESHOT)
    out = []
    for n in ns:
        X = make_data(n, d)
        out.append({"n": n,
                    "serial_ms": round(_best_ms(serial, X), 3),
                    "parallel_ms": round(_best_ms(parallel, X), 3)})
    return out


# Captured by scripts/_lowlevel_probe.py (qwen2.5-coder's kernel vs its serial twin,
# numba, quiet machine). At n=16 the parallel version is ~15x SLOWER than the same
# code run serially; by n=600 it's ~4x faster. The winner flips between 64 and 128.
SIZE_CURVE = [
    {"n": 16, "serial_ms": 0.01, "parallel_ms": 0.149},
    {"n": 64, "serial_ms": 0.146, "parallel_ms": 0.182},
    {"n": 128, "serial_ms": 0.586, "parallel_ms": 0.256},
    {"n": 600, "serial_ms": 12.589, "parallel_ms": 2.978},
    {"n": 2048, "serial_ms": 167.986, "parallel_ms": 47.004},
]


def show_size_curve(curve: list = None) -> None:
    """The model's parallel kernel against its serial twin across sizes: serial wins
    on the small matrices, parallel on the big ones, and the crossover between them
    is the break-even size the single-number scoreboard hid."""
    from genai.agent import show_turn
    curve = curve or SIZE_CURVE
    show_turn("SETUP", "the model's kernel is parallel (prange); time it against the "
                       "same code run serially, across matrix sizes")
    for row in curve:
        s, p = row["serial_ms"], row["parallel_ms"]
        if p > s:
            verdict = ("serial wins, barely" if p / s < 1.5
                       else f"serial wins, parallel {p / s:.0f}x slower")
        else:
            verdict = f"parallel wins, {s / p:.0f}x faster"
        show_turn(f"n={row['n']}", f"serial {s:g} ms, parallel {p:g} ms  ->  {verdict}")
    serial_wins = [c["n"] for c in curve if c["parallel_ms"] > c["serial_ms"]]
    parallel_wins = [c["n"] for c in curve if c["parallel_ms"] <= c["serial_ms"]]
    if serial_wins and parallel_wins:
        show_turn("CROSSOVER", f"the winner flips between n={max(serial_wins)} and "
                  f"n={min(parallel_wins)}: below it, spawning threads costs more "
                  "than it saves")


# ── One rung lower: the model writes the chip (RTL) ───────────────────────────
# The kernel ran *on* silicon. RTL (register-transfer level, written in Verilog)
# *describes* the silicon: a hardware module that gets synthesized into gates.
# The two-bar shortfall reappears, one level down. Correctness is a testbench (does the
# module match a reference on every input?), the easy bar. The hard bar is the
# hardware quality the model can't see: how many gates it costs (area) and how
# many gate-delays sit on its longest path (timing). Correct-but-bloated is the
# RTL twin of correct-but-slow. Tools: iverilog (simulate the testbench) and
# yosys (synthesize and measure); both run locally, neither is on PyPI as a
# normal import, so the demo is captured by the probe and baked.

IVERILOG = shutil.which("iverilog") or "/opt/homebrew/bin/iverilog"
VVP = shutil.which("vvp") or "/opt/homebrew/bin/vvp"
YOSYS = shutil.which("yowasp-yosys") or "yowasp-yosys"

RTL_MODEL = "qwen2.5-coder:latest"

RTL_SPEC = (
    "Write a synthesizable Verilog module with exactly this interface:\n"
    "  module popcount(input [7:0] x, output [3:0] y);\n"
    "Set y to the population count of x: the number of bits of x that are 1. "
    "Output only the Verilog module, no testbench, no explanation.")

# The correctness oracle: instantiate the candidate as popcount and check all 256
# inputs against an independent reference computed in the testbench.
RTL_TESTBENCH = r"""
module tb;
  reg [7:0] x; wire [3:0] y; integer i, gold, fails = 0;
  popcount dut(.x(x), .y(y));
  initial begin
    for (i = 0; i < 256; i = i + 1) begin
      x = i[7:0]; #1;
      gold = (i&1)+((i>>1)&1)+((i>>2)&1)+((i>>3)&1)
           + ((i>>4)&1)+((i>>5)&1)+((i>>6)&1)+((i>>7)&1);
      if (y !== gold[3:0]) fails = fails + 1;
    end
    if (fails == 0) $display("PASS"); else $display("FAIL %0d", fails);
  end
endmodule
"""

# The tuned baseline: a balanced adder tree, the shorter critical path. This is
# the area/timing bar the model's RTL has to clear, the hardware twin of the BLAS
# kernel above.
RTL_REFERENCE = (
    "module popcount(input [7:0] x, output [3:0] y);\n"
    "  wire [1:0] a = x[0]+x[1], b = x[2]+x[3], c = x[4]+x[5], d = x[6]+x[7];\n"
    "  wire [2:0] ab = a + b, cd = c + d;\n"
    "  assign y = ab + cd;\n"
    "endmodule")


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def run_testbench(module_code: str) -> dict:
    """Simulate the module against the testbench with iverilog. Returns whether
    it compiled, whether it passed all 256 inputs, and how many it missed."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write(tmp, "dut.v", module_code)
        _write(tmp, "tb.v", RTL_TESTBENCH)
        comp = subprocess.run([IVERILOG, "-g2012", "-o", str(tmp / "a.out"),
                               str(tmp / "dut.v"), str(tmp / "tb.v")],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return {"compiled": False, "correct": False, "fails": None,
                    "error": _short_text(comp.stderr)}
        run = subprocess.run([VVP, str(tmp / "a.out")], capture_output=True,
                             text=True)
        out = run.stdout
        if "PASS" in out:
            return {"compiled": True, "correct": True, "fails": 0, "error": None}
        m = re.search(r"FAIL\s+(\d+)", out)
        return {"compiled": True, "correct": False,
                "fails": int(m.group(1)) if m else None, "error": None}


def synth_metrics(module_code: str) -> dict:
    """Synthesize to gates with yosys and measure the two hardware costs the
    model can't see: gate count (area) and longest path length (timing)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write(tmp, "dut.v", module_code)
        script = ("read_verilog -sv dut.v; hierarchy -auto-top; proc; flatten; "
                  "opt; techmap; opt; simplemap; opt; stat; ltp")
        r = subprocess.run([YOSYS, "-p", script],
                           capture_output=True, text=True, cwd=str(tmp))
        text = r.stdout + r.stderr
        # Sum the per-type gate breakdown ("  15   $_AND_"); the longest path
        # length is the timing proxy.
        gate_lines = re.findall(r"^\s*(\d+)\s+\$_\w+_\s*$", text, re.M)
        depth = re.search(r"length=(\d+)", text)
        return {"gates": sum(int(g) for g in gate_lines) if gate_lines else None,
                "depth": int(depth.group(1)) if depth else None}


def _short_text(s: str) -> str:
    line = next((ln for ln in (s or "").splitlines() if ln.strip()), "")
    return line.strip()[:90]


def _emit_verilog(model: str, prompt: str) -> str:
    """Ask for a Verilog module; strip any fence and trailing prose."""
    msg = ollama.chat(model=model, options={"num_predict": 500},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = msg.get("content") or ""
    fenced = re.search(r"```(?:verilog|systemverilog)?\n(.*?)```", text, re.S)
    code = (fenced.group(1) if fenced else text).strip()
    end = code.rfind("endmodule")
    return code[:end + len("endmodule")] if end != -1 else code


def propose_rtl(model: str = RTL_MODEL) -> dict:
    """The model writes the module; we test it and synthesize it."""
    code = _emit_verilog(model, RTL_SPEC)
    tb = run_testbench(code)
    metrics = synth_metrics(code) if tb["correct"] else {"gates": None,
                                                          "depth": None}
    return {"code": code, **tb, **{f"model_{k}": v for k, v in metrics.items()}}


# ── Baked from scripts/_lowlevel_probe.py (qwen2.5-coder + iverilog + yosys) ───
# The model's real Verilog, kept verbatim: a correct popcount written as a flat
# loop that adds the bits one at a time. It passes all 256 inputs, and synthesis
# shows the catch it couldn't: more gates and a longer critical path than the
# tuned adder tree. Correct, and still worse hardware. The ref_* fields are the
# reference's measured cost.
RTL_DEMO = {
    "code": (
        "module popcount(input [7:0] x, output reg [3:0] y);\n"
        "always @(*) begin\n"
        "    y = 4'b0;\n"
        "    for (integer i = 0; i < 8; i = i + 1) begin\n"
        "        y = y + (x[i] ? 1'b1 : 1'b0);\n"
        "    end\n"
        "end\n"
        "endmodule"),
    "correct": True,
    "model_gates": 40,
    "model_depth": 10,
    "ref_gates": 34,
    "ref_depth": 8,
}


def show_rtl_demo(demo: dict = RTL_DEMO):
    """The model writes the chip: its real Verilog, the testbench verdict, and
    the hardware cost it couldn't see next to the tuned reference."""
    from genai.agent import show_code, show_turn
    show_turn("you", "Write a Verilog popcount module: count the 1 bits of an "
                     "8-bit input. Make it correct.")
    show_code("qwen2.5-coder", demo["code"])
    show_turn("TEST", "PASS, all 256 inputs match the reference."
                      if demo["correct"] else "FAIL, the module is wrong.")
    show_turn("SYNTH", f"correct, but {demo['model_gates']} gates and a "
                       f"{demo['model_depth']}-deep critical path, against the "
                       f"tuned adder tree's {demo['ref_gates']} gates and "
                       f"{demo['ref_depth']}. Correct, and still slower silicon.")


# ── Fast on paper, wrong in the corners ───────────────────────────────────────
# baseline() went fast by the squared-norm identity, |a-b|^2 = |a|^2 + |b|^2 -
# 2 a.b, which turns the whole distance matrix into one matmul BLAS races through.
# In exact arithmetic it equals reference(). In floating point it subtracts large
# nearly-equal quantities, and catastrophic cancellation can lose every
# significant digit, even printing a *negative* squared distance, which can't
# exist. On the random cloud a kernel is checked against, this is invisible: the
# two agree to 1e-13. It only shows on inputs the single check never tries, points
# far from the origin or spanning a wide range of magnitudes. Everything here is
# deterministic numpy, so the cell runs live.

FAST_KERNEL_SRC = ("g = (X * X).sum(1)\n"
                   "return g[:, None] + g[None, :] - 2.0 * (X @ X.T)")


def corner_battery():
    """Inputs the single random-cloud check never tries: a normal cloud (the
    check itself), a large-magnitude cloud, a wide dynamic range, and a cluster
    sitting far from the origin. Returns ``[(name, X)]``, deterministic."""
    rng = np.random.default_rng(1)
    return [
        ("random cloud (the check)", make_data()),
        ("cloud scaled by 1e6", make_data(80) * 1e6),
        ("wide range of magnitudes",
         np.vstack([rng.standard_normal((20, 8)) * 1e6,
                    rng.standard_normal((20, 8)) * 1e-3])),
        ("cluster far from origin", 1e7 + rng.standard_normal((40, 32))),
    ]


def corner_report(fast=baseline, exact=reference):
    """Score the fast kernel against the exact reference across the battery.
    Returns ``[(name, agrees, neg)]``: whether the two match, and how many
    'squared distances' came out negative (a value that cannot exist)."""
    out = []
    for name, X in corner_battery():
        f, r = fast(X), exact(X)
        agrees = bool(np.allclose(r, f, rtol=1e-4, atol=1e-6))
        neg = int((f < -1e-6).sum())
        out.append((name, agrees, neg))
    return out


def show_corner_cases() -> None:
    """The fast kernel against the exact reference on the corner battery: it
    matches on the random cloud the check uses and comes apart on the corners,
    printing impossible negative squared distances."""
    from genai.agent import show_code
    show_code("the fast kernel (BLAS identity)", FAST_KERNEL_SRC)
    print(f"{'input':26}{'matches exact?':>15}{'neg. dists':>12}")
    for name, agrees, neg in corner_report():
        print(f"{name:26}{('yes' if agrees else 'NO'):>15}{(str(neg) if neg else '-'):>12}")
