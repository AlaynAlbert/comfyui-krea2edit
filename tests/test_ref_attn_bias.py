"""Verifica _ref_attn_bias tras el parche ref_boost_out, extrayendo la funcion
REAL del __init__.py del nodo (sin importar comfy).

Run standalone:  python tests/test_ref_attn_bias.py
"""
import ast, math, re, sys
from pathlib import Path
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "__init__.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ref_attn_bias")
ns = {"torch": torch, "F": F, "math": math}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<x>", "exec"), ns)
bias_fn = ns["_ref_attn_bias"]

TXT, SL, TGT = 3, 16, 5           # 3 text tokens, 1 ref de 16 tokens (4x4), 5 target
HW = [(4, 4)]
ROWS0 = TXT + SL
# mascara: 4 tokens de "cara" (esquina sup-izq 2x2), resto "cuerpo".
# Un MASK de ComfyUI es (B,H,W) -> 3D. Con 2D, mask[:1] corta la primera FILA.
m = torch.zeros(1, 4, 4); m[0, :2, :2] = 1.0
FACE = [TXT + i for i in (0, 1, 4, 5)]
BODY = [c for c in range(TXT, TXT + SL) if c not in FACE]


def run(boosts, mask, out=1.0):
    return bias_fn(boosts, mask, TXT, [SL], TGT, HW, "cpu", torch.float32, boost_out=out)


def chk(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    return cond


ok = True
L = ROWS0 + TGT

# --- retrocompat: sin mascara, b=0.65 (el regimen de produccion) ---
b = run([0.65], None)
ok &= chk("sin mascara: todo el ref a log(0.65)",
          torch.allclose(b[0, 0, ROWS0:, TXT:TXT + SL], torch.tensor(math.log(0.65))))
ok &= chk("sin mascara: columnas de texto intactas",
          torch.all(b[0, 0, ROWS0:, :TXT] == 0))
ok &= chk("sin mascara: filas de texto/ref intactas (solo sesga target->ref)",
          torch.all(b[0, 0, :ROWS0, :] == 0))

# --- retrocompat: mascara + b=2.0, out=1.0 (lo que se ejecuto en los pilotos) ---
b = run([2.0], m, out=1.0)
ok &= chk("mask b=2.0 out=1.0: cara a log(2)",
          torch.allclose(b[0, 0, ROWS0:, FACE], torch.tensor(math.log(2.0))))
ok &= chk("mask b=2.0 out=1.0: CUERPO A 0 (neutro) <- el fallo diagnosticado",
          torch.all(b[0, 0, ROWS0:, BODY] == 0))

# --- retrocompat: la fila que era un no-op ---
b = run([1.0], m, out=1.0)
ok &= chk("mask b=1.0 out=1.0: bias TODO cero (mascara nunca leida)",
          torch.all(b == 0))

# --- NUEVO: mascara invertida por escalar. cara neutra, cuerpo suprimido ---
b = run([1.0], m, out=0.65)
ok &= chk("NUEVO b=1.0 out=0.65: cara a 0 (neutro)",
          torch.all(b[0, 0, ROWS0:, FACE] == 0))
ok &= chk("NUEVO b=1.0 out=0.65: cuerpo a log(0.65)",
          torch.allclose(b[0, 0, ROWS0:, BODY], torch.tensor(math.log(0.65))))

# --- NUEVO: el objetivo real de 4.2 — cara 2.0 / cuerpo 0.65 ---
b = run([2.0], m, out=0.65)
ok &= chk("NUEVO b=2.0 out=0.65: cara a log(2.0)",
          torch.allclose(b[0, 0, ROWS0:, FACE], torch.tensor(math.log(2.0))))
ok &= chk("NUEVO b=2.0 out=0.65: cuerpo a log(0.65)",
          torch.allclose(b[0, 0, ROWS0:, BODY], torch.tensor(math.log(0.65))))
ok &= chk("NUEVO b=2.0 out=0.65: texto intacto",
          torch.all(b[0, 0, ROWS0:, :TXT] == 0))

# --- out sin mascara no debe hacer nada (no hay "fuera de la mascara") ---
b = run([1.0], None, out=0.65)
ok &= chk("out=0.65 sin mascara: bias todo cero (no aplica)", torch.all(b == 0))

def test_ref_attn_bias():
    """pytest: every check above must hold."""
    assert ok


if __name__ == "__main__":
    print("\n=> TODO OK" if ok else "\n=> HAY FALLOS")
    sys.exit(0 if ok else 1)
