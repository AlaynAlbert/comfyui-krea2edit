"""Mutaciones de moss_ref_boost: quita (o rompe) a proposito cada linea clave en una
copia temporal del paquete y comprueba que tests/test_moss_ref_boost.py FALLA.
Si alguna mutacion pasa los tests, ese test no prueba su arreglo.

Run standalone:  python tests/mutar_moss_ref_boost.py
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests" / "test_moss_ref_boost.py"

# (fichero, texto original, mutacion, que deberia detectar)
MUTATIONS = [
    ("core.py", "bias[:, :, :n_img, cols] = math.log(max(b, 1e-4))",
     "bias[:, :, :, cols] = math.log(max(b, 1e-4))", "solo filas de imagen (texto/refs intactos)"),
    ("core.py", "offs = [n_img + n_txt]", "offs = [n_img]", "columnas de la ref tras [imagen | texto]"),
    ("core.py", "if b == 1.0 and b_out == 1.0:", "if b == 1.0:", "no saltar la ref con mascara y b=1"),
    ("core.py", "bias[:, :, :n_img, off:off + sl] = math.log(max(b_out, 1e-4))", "pass",
     "ref_boost_out fuera de la mascara"),
    ("core.py", "if lk == n_k and lq in (n_img + n_txt, n_k):\n            dtype",
     "if True:\n            dtype", "el override solo sesga los bloques DiT"),
    ("core.py", "mask = bias if mask is None else mask + bias", "mask = bias",
     "sumar a la mascara nativa (image_strength)"),
    ("core.py", "target = preferred(kwargs.get(\"transformer_options\"), skip_reshape, lq, lk) or func",
     "target = func", "conservar la atencion preferida de cada bloque"),
    ("core.py", "to = transformer_options.copy()", "to = transformer_options",
     "no ensuciar el transformer_options del llamador"),
    ("core.py", "if refs:\n        hw", "if False:\n        hw", "preferir ref_latents (rejilla por ref)"),
    ("core.py", "if mask is not None:\n                warn_once", "if False:\n                warn_once",
     "ignorar la mascara sin rejilla (kv_cache nativo)"),
    ("core.py", "if ref_boost_a != 1.0:", "if ref_boost_a != ref_boost:",
     "no avisar de ref_boost_a cuando sigue a 1.0"),
    ("nodes.py", "            return (model,)\n", "            pass\n",
     "con 1.0 devolver el mismo modelo"),
]


def main():
    bad = 0
    for fname, old, new, what in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "moss_ref_boost"
            shutil.copytree(ROOT / "moss_ref_boost", pkg)
            src = (pkg / fname).read_text(encoding="utf-8")
            if src.count(old) != 1:
                print(f"?? {fname}: el texto a mutar aparece {src.count(old)} veces: {old!r}")
                bad += 1
                continue
            (pkg / fname).write_text(src.replace(old, new), encoding="utf-8")
            env = dict(os.environ, MOSS_REF_BOOST_DIR=str(pkg))
            r = subprocess.run([sys.executable, str(TEST)], env=env, capture_output=True, text=True)
            fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
            caught = r.returncode != 0
            print(f"{'OK  ' if caught else 'VIVA'} {what}: "
                  f"{len(fails)} FAIL{'' if fails else ' (' + (r.stderr.strip().splitlines() or ['?'])[-1] + ')'}")
            bad += not caught
    print("\n=> TODAS LAS MUTACIONES CAZADAS" if not bad else f"\n=> {bad} MUTACIONES VIVAS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
