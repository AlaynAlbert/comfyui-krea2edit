# MOSS-Image Ref Boost (prototipo)

Port a MOSS-Image del `ref_boost` de comfyui-krea2edit. Es un nodo aparte; el nodo
krea2edit de la raíz no lo carga ni cambia.

- **Qué hace.** Suma `log(b)` a los logits de atención de los tokens de **imagen** (el
  objetivo) hacia las **claves de cada referencia**, en los 42 bloques DiT. Es lo mismo
  que multiplicar por `b` la atención a esa referencia y renormalizar.
- **Qué no toca.** Ni las filas de texto, ni las de la propia referencia, ni la pasada
  que precalcula la caché K/V de las referencias (t=0, refs aisladas). La caché sigue
  siendo exactamente la que vio la LoRA.
- **Por defecto (todo a 1,0)** el nodo devuelve el mismo objeto `MODEL` que recibe: sin
  clon ni wrapper. La salida es idéntica bit a bit a no tener el nodo.

## Entradas

| entrada | por defecto | efecto |
|---|---|---|
| `ref_boost` | 1,0 | la ÚLTIMA referencia (la única en edición de una imagen) |
| `ref_boost_a` | 1,0 | las referencias anteriores (la escena en dos imágenes) |
| `ref_boost_mask` | — | región de la última referencia donde va `ref_boost` (p. ej. la cara) |
| `ref_boost_out` | 1,0 | la última referencia FUERA de la máscara (última entrada) |

## Cableado

```
Load Diffusion Model (moss_image) -> Load LoRA -> [edición] -> MOSS-Image Ref Boost
    -> SkipLayerGuidanceDiTSimple (single_layers 22) -> KSampler
```

`[edición]` puede ser cualquiera de las dos vías:

- **Nativa (ComfyUI master).** No hay nodo de modelo: el `TextEncodeMOSSImageEdit`
  (`edit_format` ai-toolkit) va al `positive` y al `negative`.
- **Ostris.** `MOSSImageOstrisEditModelPatch` (con `kv_cache` encendido o apagado).

El orden respecto a SLG da igual.

## Límites conocidos

- **Vía nativa con `kv_cache`.** Las referencias llegan como un solo bloque de claves
  (`ref_kv`), sin la división por referencia ni su rejilla. Por eso:
  - `ref_boost` se aplica a todas;
  - `ref_boost_a`, `ref_boost_mask` y `ref_boost_out` se ignoran, con aviso.

  Con el parche de Ostris, o con `kv_cache` apagado, funcionan todos.
- **Kernel de atención.** Con un valor distinto de 1, la atención de los bloques DiT
  lleva una máscara densa `(1, 1, Lq, Lk)`, unos 75 MB a 1024² con una referencia de
  1024². Eso obliga a un kernel con máscara, más lento. `image_strength` ya paga lo
  mismo. No se ha medido en GPU.
- **Atención preferida.** El nodo devuelve a cada bloque nativo la atención que
  declara (`ComfyAttention`), que ComfyUI descarta en cuanto hay un override. No está
  verificado con un modelo que la declare (comfy_kitchen_int8 / sol).
- **cond y uncond.** El sesgo va en las dos pasadas, como en krea2edit. Sesgar solo la
  positiva queda como variante sin probar.

## Instalación

Copia o enlaza esta carpeta en `ComfyUI/custom_nodes/`:

- Windows: `mklink /D <ComfyUI>\custom_nodes\moss_ref_boost <repo>\moss_ref_boost`
- Linux / macOS: `ln -s`

## Tests

```
python tests/test_moss_ref_boost.py      # CPU, tensores de juguete, sin comfy
python tests/mutar_moss_ref_boost.py     # cada línea clave quitada debe hacer fallar el test
```
