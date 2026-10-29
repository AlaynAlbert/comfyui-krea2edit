"""Prototipo moss_ref_boost: tests en CPU con tensores de juguete, sin comfy.

Carga moss_ref_boost/core.py y el paquete (nodes.py) por ruta. Un MOSS de juguete
reproduce lo que importa del nativo (comfy/ldm/moss_image/model.py): secuencia
[imagen | texto | refs], K/V de las refs precalculadas a t=0 y anadidas como claves,
despacho de atencion al estilo wrap_attn (override > preferred > por defecto),
mascara aditiva del texto (image_strength) y salto de capa (SLG).

Run standalone:  python tests/test_moss_ref_boost.py
MOSS_REF_BOOST_DIR=<carpeta> carga otra copia (lo usa tests/mutar_moss_ref_boost.py).
"""
import importlib.util, math, os, sys, types
from pathlib import Path
import torch
import torch.nn.functional as F

PKG = Path(os.environ.get("MOSS_REF_BOOST_DIR",
                          Path(__file__).resolve().parents[1] / "moss_ref_boost"))


def _load_pkg():
    spec = importlib.util.spec_from_file_location("moss_ref_boost", PKG / "__init__.py",
                                                  submodule_search_locations=[str(PKG)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["moss_ref_boost"] = pkg
    spec.loader.exec_module(pkg)
    return pkg, sys.modules["moss_ref_boost.core"], sys.modules["moss_ref_boost.nodes"]


pkg, core, nodes = _load_pkg()


def chk(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    return bool(cond)


ok = True
IMG, TXT = 6, 3                   # 6 tokens de imagen (2x3), 3 de texto
L = math.log

# ---------------------------------------------------------------- 1. ref_bias
# una ref 4x4 (16 tokens); mascara de "cara" 2x2 arriba a la izquierda
R, HW = 16, [(4, 4)]
m = torch.zeros(1, 4, 4); m[0, :2, :2] = 1.0
OFF = IMG + TXT
FACE = [OFF + i for i in (0, 1, 4, 5)]
BODY = [c for c in range(OFF, OFF + R) if c not in FACE]
NK = IMG + TXT + R


def bias(boosts, mask=None, out=1.0, nq=IMG + TXT, lens=(R,), hw=HW):
    return core.ref_bias(boosts, mask, IMG, TXT, list(lens), hw, nq, "cpu", torch.float32,
                         boost_out=out)


b = bias([0.65])
ok &= chk("bias: forma (1,1,img+txt,img+txt+ref) con K/V en cache", b.shape == (1, 1, IMG + TXT, NK))
ok &= chk("bias: filas de imagen x columnas de la ref = log(0.65)",
          torch.allclose(b[0, 0, :IMG, OFF:], torch.tensor(L(0.65))))
ok &= chk("bias: filas de texto intactas", torch.all(b[0, 0, IMG:] == 0))
ok &= chk("bias: columnas de imagen y texto intactas", torch.all(b[0, 0, :, :OFF] == 0))
b = bias([0.65], nq=NK)
ok &= chk("bias en secuencia (kv_cache off): filas de la ref intactas",
          b.shape == (1, 1, NK, NK) and torch.all(b[0, 0, IMG:] == 0)
          and torch.allclose(b[0, 0, :IMG, OFF:], torch.tensor(L(0.65))))
ok &= chk("bias: todo a 1.0 -> None (nada que sesgar)", bias([1.0]) is None)
b = core.ref_bias(core.boost_list(2, 0.65, 2.5), None, IMG, TXT, [4, 16], [(2, 2), (4, 4)],
                  IMG + TXT, "cpu", torch.float32)
ok &= chk("dos refs: la primera (escena) a log(ref_boost_a), la ultima a log(ref_boost)",
          torch.allclose(b[0, 0, :IMG, OFF:OFF + 4], torch.tensor(L(2.5)))
          and torch.allclose(b[0, 0, :IMG, OFF + 4:], torch.tensor(L(0.65))))
b = bias([2.0], m, out=0.65)
ok &= chk("mascara b=2.0 out=0.65: cara log(2), cuerpo log(0.65)",
          torch.allclose(b[0, 0, :IMG, FACE], torch.tensor(L(2.0)))
          and torch.allclose(b[0, 0, :IMG, BODY], torch.tensor(L(0.65))))
b = bias([2.0], m)
ok &= chk("mascara b=2.0 out=1.0: cuerpo neutro (comportamiento krea2edit)",
          torch.all(b[0, 0, :IMG, BODY] == 0))
b = bias([1.0], m, out=0.65)
ok &= chk("mascara b=1.0 out=0.65: NO se salta la ref (el arreglo de ref_boost_out)",
          b is not None and torch.all(b[0, 0, :IMG, FACE] == 0)
          and torch.allclose(b[0, 0, :IMG, BODY], torch.tensor(L(0.65))))
ok &= chk("mascara sin rejilla de la ref (kv_cache nativo): se ignora",
          torch.allclose(bias([2.0], m, out=0.65, hw=None)[0, 0, :IMG, OFF:], torch.tensor(L(2.0))))

# softmax(s + log b) == reponderar softmax(s) por b en las columnas de la ref y renormalizar
torch.manual_seed(0)
s = torch.randn(IMG + TXT, NK, dtype=torch.float64)
w = torch.ones_like(s); w[:IMG, OFF:] = 1.7
p = s.softmax(-1) * w
ok &= chk("semantica: el bias multiplica la atencion a la ref y renormaliza",
          torch.allclose((s + bias([1.7]).double()[0, 0]).softmax(-1), p / p.sum(-1, keepdim=True)))

# ---------------------------------------------------------------- 2. layout
x = torch.zeros(1, 4, 2, 3); ctx = torch.zeros(1, TXT, 8)
lay = core.ref_layout(x, ctx, {"ref_latents": [torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 1, 4, 4)]})
ok &= chk("layout ostris/kv off: un bloque y una rejilla por ref (4D y 5D)",
          lay == (6, 3, [4, 16], [(2, 2), (4, 4)]))
lay = core.ref_layout(x, ctx, {"ref_kv": torch.zeros(1, 5, 2, 20, 8)})
ok &= chk("layout nativo kv_cache: un solo bloque de Lr claves, sin rejilla", lay == (6, 3, [20], None))
lay = core.ref_layout(x, ctx, {"ref_latents": [torch.zeros(1, 4, 4, 4)], "ref_kv": torch.zeros(1, 5, 2, 16, 8)})
ok &= chk("layout: con ref_latents y ref_kv (ostris sobre master) manda ref_latents",
          lay == (6, 3, [16], [(4, 4)]))
ok &= chk("layout: sin refs -> None", core.ref_layout(x, ctx, {}) is None)


# ---------------------------------------------------------------- 3. override
def attn_basic(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kw):
    """attention_basic de comfy, en pequeno: (B,N,H*D) o (B,H,N,D) con skip_reshape."""
    if not skip_reshape:
        B, N, C = q.shape
        q, k, v = (t.view(B, -1, heads, C // heads).transpose(1, 2) for t in (q, k, v))
    sim = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if mask is not None:
        sim = sim + mask
    out = sim.softmax(-1) @ v
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).flatten(2)


H, D = 2, 4
torch.manual_seed(1)
qd, kd, vd = torch.randn(1, IMG + TXT, H * D), torch.randn(1, NK, H * D), torch.randn(1, NK, H * D)
ov = core.make_attention_override([2.0], None, IMG, TXT, [R], HW)
ok &= chk("override: llamada de bloque DiT = atencion con el bias",
          torch.equal(ov(attn_basic, qd, kd, vd, H), attn_basic(qd, kd, vd, H, mask=bias([2.0]))))
qt = torch.randn(1, TXT, H * D)
ok &= chk("override: bloque de texto pasa intacto (bit a bit)",
          torch.equal(ov(attn_basic, qt, qt, qt, H), attn_basic(qt, qt, qt, H)))
qr = torch.randn(1, R, H * D)
ok &= chk("override: pasada solo-refs (precalculo K/V) pasa intacta (bit a bit)",
          torch.equal(ov(attn_basic, qr, qr, qr, H), attn_basic(qr, qr, qr, H)))
cm = torch.zeros(1, 1, 1, NK); cm[..., IMG:OFF] = L(0.5)          # image_strength en el texto
ok &= chk("override: suma su bias a la mascara nativa (image_strength), no la pisa",
          torch.equal(ov(attn_basic, qd, kd, vd, H, mask=cm),
                      attn_basic(qd, kd, vd, H, mask=cm + bias([2.0]))))
q4, k4, v4 = (t.view(1, -1, H, D).transpose(1, 2) for t in (qd, kd, vd))
ok &= chk("override: layout (B,H,N,D) del parche de ostris (skip_reshape=True)",
          torch.allclose(ov(attn_basic, q4, k4, v4, H, skip_reshape=True),
                         attn_basic(qd, kd, vd, H, mask=bias([2.0])), atol=1e-6))
seen = {}


def prev(func, *a, **kw):
    seen["func"], seen["mask"] = func, kw.get("mask")
    return func(*a, **kw)


ov2 = core.make_attention_override([2.0], None, IMG, TXT, [R], HW, prev_override=prev)
ov2(attn_basic, qd, kd, vd, H)
ok &= chk("override: encadena un override previo y le pasa la mascara sesgada",
          seen.get("func") is attn_basic and torch.equal(seen["mask"], bias([2.0])))

# ---------------------------------------------------------------- 4. MOSS de juguete
calls = {"pref": 0}


def preferred_fn(*a, **kw):
    calls["pref"] += 1
    return attn_basic(*a, **kw)


def dispatch(q, k, v, heads, mask=None, transformer_options=None, preferred=None):
    """wrap_attn: override > preferred_attention.function > funcion por defecto."""
    to = transformer_options or {}
    if "optimized_attention_override" in to:
        return to["optimized_attention_override"](attn_basic, q, k, v, heads, mask=mask,
                                                  transformer_options=to, _inside_attn_wrapper=True)
    if preferred is not None and preferred.function is not None:
        return preferred.function(q, k, v, heads, mask=mask, transformer_options=to)
    return attn_basic(q, k, v, heads, mask=mask)


class Blk(torch.nn.Module):
    def __init__(self, dim, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv = torch.nn.Parameter(torch.randn(dim, 3 * dim, generator=g) * 0.3)
        self.o = torch.nn.Parameter(torch.randn(dim, dim, generator=g) * 0.3)
        self.comfy_attention = types.SimpleNamespace(function=preferred_fn)

    def forward(self, x, to, ref_kv=None, kv_out=None, mask=None):
        q, k, v = (x @ self.qkv).chunk(3, -1)
        if kv_out is not None:
            kv_out[:, 0].copy_(k)
            kv_out[:, 1].copy_(v)
        if ref_kv is not None:                      # model.py:78-80
            k, v = torch.cat((k, ref_kv[:, 0]), 1), torch.cat((v, ref_kv[:, 1]), 1)
        return x + dispatch(q, k, v, H, mask=mask, transformer_options=to,
                            preferred=self.comfy_attention) @ self.o


class ToyMOSS(torch.nn.Module):
    def __init__(self, dim=H * D, nl=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(Blk(dim, 10 + i) for i in range(nl))
        self.text_embedder = types.SimpleNamespace(blocks=[Blk(dim, 99)])

    def preprocess_refs(self, refs):
        x = torch.cat([r.flatten(2).transpose(1, 2) for r in refs], 1)
        kv = x.new_empty((x.shape[0], len(self.layers), 2) + x.shape[1:])   # model.py:205
        for i, blk in enumerate(self.layers):       # sin transformer_options, como extra_conds
            x = blk(x, {}, kv_out=kv[:, i])
        return kv

    def forward(self, x, t, context, transformer_options={}, ref_kv=None, ref_latents=None,
                context_attention_bias=None):
        to = transformer_options.copy()
        txt = self.text_embedder.blocks[0](context, to)
        h = torch.cat([x.flatten(2).transpose(1, 2), txt]
                      + [r.flatten(2).transpose(1, 2) for r in (ref_latents or [])], 1)
        nkv = 0 if ref_kv is None else ref_kv.shape[3]
        mask = None
        if context_attention_bias is not None:
            mask = F.pad(context_attention_bias, (x.shape[-2] * x.shape[-1], h.shape[1] - x.shape[-2] * x.shape[-1] - context.shape[1] + nkv))[:, None, None]
        skip = to.get("patches_replace", {}).get("dit", {})
        for i, blk in enumerate(self.layers):
            to["block_index"] = i
            if ("single_block", i) in skip:
                continue
            h = blk(h, to, None if ref_kv is None else ref_kv[:, i], mask=mask)
        return h[:, :x.shape[-2] * x.shape[-1]]


class ToyBase:
    """BaseModel.apply_model: ejecuta los wrappers APPLY_MODEL como WrapperExecutor."""
    def __init__(self, dit):
        self.diffusion_model = dit
        self.wrappers = []

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options={}, **kw):
        def run(idx, *a, **k):
            if idx == len(self.wrappers):
                xx, tt, _, ctx, _, to = a
                return self.diffusion_model(xx, tt, ctx, to, **k)
            ex = lambda *a2, **k2: run(idx + 1, *a2, **k2)
            ex.class_obj = self
            return self.wrappers[idx](ex, *a, **k)
        return run(0, x, t, c_concat, c_crossattn, control, transformer_options, **kw)


class ToyPatcher:
    def __init__(self, base):
        self.model = base

    def clone(self):
        c = ToyPatcher(ToyBase(self.model.diffusion_model))
        c.model.wrappers = list(self.model.wrappers)
        return c

    def add_wrapper_with_key(self, kind, key, fn):
        assert kind == "apply_model"
        self.model.wrappers.append(fn)


fake = types.ModuleType("comfy.patcher_extension")
fake.WrappersMP = types.SimpleNamespace(APPLY_MODEL="apply_model")
sys.modules.setdefault("comfy", types.ModuleType("comfy")).patcher_extension = fake
sys.modules["comfy.patcher_extension"] = fake

torch.manual_seed(2)
dit = ToyMOSS()
xin, ctx = torch.randn(1, H * D, 2, 3), torch.randn(1, TXT, H * D)
ref = torch.randn(1, H * D, 4, 4)
kv = dit.preprocess_refs([ref])
kv_copy = kv.clone()
base = ToyPatcher(ToyBase(dit))
plain = base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv)
node = nodes.MOSSImageRefBoost()

(mdef,) = node.patch(base)
ok &= chk("nodo con 1.0 por defecto: devuelve EL MISMO modelo (sin clon ni wrapper)",
          mdef is base and base.model.wrappers == [])
out = mdef.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv)
ok &= chk("nodo con 1.0 por defecto: salida bit a bit igual a sin nodo", torch.equal(out, plain))

to_seen = {}


def spy(executor, *a, **k):
    to_seen["to"] = a[5]
    return executor(*a, **k)


wdef = core.make_apply_model_wrapper()
to0 = {"x": 1}
base.model.wrappers = [wdef, spy]
out = base.model.apply_model(xin, 0.5, None, ctx, None, to0, ref_kv=kv)
ok &= chk("wrapper neutro: mismo transformer_options y salida bit a bit",
          to_seen["to"] is to0 and "optimized_attention_override" not in to0 and torch.equal(out, plain))
base.model.wrappers = []

(mb,) = node.patch(base, ref_boost=2.0)
calls["pref"] = 0
to1 = {}
boosted = mb.model.apply_model(xin, 0.5, None, ctx, None, to1, ref_kv=kv)
ok &= chk("ref_boost=2.0 cambia la salida", not torch.allclose(boosted, plain))
ok &= chk("ref_boost=2.0: el transformer_options del llamador queda limpio",
          "optimized_attention_override" not in to1)
ok &= chk("ref_boost=2.0: la cache K/V de las refs no se toca",
          torch.equal(kv, kv_copy))
ok &= chk("ref_boost=2.0: cada bloque sigue usando su atencion preferida (3 DiT + 1 texto)",
          calls["pref"] == 4)


# referencia: el mismo juguete pasando a mano el bias como mascara de cada bloque DiT
def manual(boost, cab=None):
    to = {}
    txt = dit.text_embedder.blocks[0](ctx, to)
    h = torch.cat([xin.flatten(2).transpose(1, 2), txt], 1)
    mask = core.ref_bias([boost], None, IMG, TXT, [R], None, IMG + TXT, "cpu", torch.float32)
    if cab is not None:
        mask = F.pad(cab, (IMG, R))[:, None, None] + mask
    for i, blk in enumerate(dit.layers):
        h = blk(h, to, kv[:, i], mask=mask)
    return h[:, :IMG]


ok &= chk("extremo a extremo: wrapper+override == mascara log(b) puesta a mano (kv_cache)",
          torch.equal(boosted, manual(2.0)))
cab = torch.tensor([[L(0.5), 0.0, 0.0]])
plain_cab = base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv, context_attention_bias=cab)
b_cab = mb.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv, context_attention_bias=cab)
ok &= chk("extremo a extremo con image_strength nativo: los dos sesgos se suman",
          torch.equal(b_cab, manual(2.0, cab)) and not torch.equal(b_cab, plain_cab))

# kv_cache off: refs en la secuencia
plain_in = base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_latents=[ref])
b_in = mb.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_latents=[ref])
(m1,) = node.patch(base, ref_boost=1.0)
ok &= chk("kv_cache off: 1.0 identico, 2.0 cambia",
          torch.equal(m1.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_latents=[ref]), plain_in)
          and not torch.allclose(b_in, plain_in))

# SLG: capa 1 saltada en la pasada negativa
slg = {"patches_replace": {"dit": {("single_block", 1): None}}}
calls["pref"] = 0
b_slg = mb.model.apply_model(xin, 0.5, None, ctx, None, slg, ref_kv=kv)
ok &= chk("SLG (capa saltada): corre, sesga las demas capas y no toca el dict del llamador",
          calls["pref"] == 3 and not torch.allclose(b_slg, boosted)
          and "optimized_attention_override" not in slg)

msgs = []
base.model.wrappers = [core.make_apply_model_wrapper(ref_boost=2.0, log=msgs.append)]
o2 = base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv)
ok &= chk("kv_cache nativo, ref_boost=2.0 con ref_boost_a=1.0: igual que el nodo y sin avisos",
          torch.equal(o2, boosted) and msgs == [])
base.model.wrappers = [core.make_apply_model_wrapper(ref_boost=2.0, ref_boost_a=1.5, log=msgs.append)]
base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv)
ok &= chk("kv_cache nativo con ref_boost_a != 1: avisa de que se ignora",
          any("ref_boost_a" in s for s in msgs))
msgs = []
w = core.make_apply_model_wrapper(ref_boost=1.0, ref_boost_out=0.65, ref_boost_mask=m,
                                  log=msgs.append)
base.model.wrappers = [w]
o = base.model.apply_model(xin, 0.5, None, ctx, None, {}, ref_kv=kv)
ok &= chk("kv_cache nativo + mascara: se ignora con aviso y no cambia nada",
          torch.equal(o, plain) and any("ref_boost_mask" in s for s in msgs))
base.model.wrappers = []


def test_moss_ref_boost():
    """pytest: every check above must hold."""
    assert ok


if __name__ == "__main__":
    print("\n=> TODO OK" if ok else "\n=> HAY FALLOS")
    sys.exit(0 if ok else 1)
