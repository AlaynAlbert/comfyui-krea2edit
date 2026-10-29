# Contexto para el agente de investigación (rama `ref-boost-out`)

Este fork es de `lbouaraba/comfyui-krea2edit`. La rama `ref-boost-out` es el upstream **v2.0.0** (`13ef60f`, 16-10-2026) más **nuestro parche local**, el mismo que usamos en producción. El upstream no tiene commits posteriores a `13ef60f` a fecha de 29-10-2026. Compruébalo con `git fetch upstream` o en la página de commits del original.

Aquí va todo lo que necesitas saber de nuestra instalación, porque no tienes acceso a ella.

## 1. Nuestro parche: `ref_boost_out` (fichero `__init__.py`)

`git diff 13ef60f -- __init__.py` lo enseña completo. Contenido:

1. **Entrada nueva `ref_boost_out`** (FLOAT, 1,0 por defecto, va la ÚLTIMA en la lista de entradas) en `Krea2EditModelPatch`.
   - Es el escalar de la **última** referencia (la persona en los flujos de dos imágenes) para los tokens que quedan **fuera** de `ref_boost_mask`.
   - `ref_boost` sigue aplicándose **dentro** de la máscara.
   - Sin el parche, la máscara solo podía *añadir* boost a su región, y el resto se quedaba neutro a 1,0. Ese 1,0 es **más** que el `ref_boost` global típico de 0,65, así que una pasada con máscara dejaba pasar más cuerpo y ropa de la referencia que una sin máscara.
2. **`_ref_attn_bias(..., boost_out=1.0)`:** el sesgo es aditivo sobre los logits de atención target→ref, `log(b)`. Con máscara y `boost_out ≠ 1`, primero pone toda la referencia a `log(boost_out)` y después la región de la máscara a `log(ref_boost)`.
3. **Arreglo de un fallo:** el `if b == 1.0: continue` original se saltaba la referencia entera cuando `ref_boost` valía 1,0, y la máscara ni se leía. Ahora solo se salta si `b == 1.0` **y** `boost_out == 1.0`.
4. **Aviso** por consola si `ref_boost_out ≠ 1` sin máscara conectada (no tiene efecto).
5. **Orden de entradas:** el de upstream, con `ref_boost_out` al final, para que los workflows guardados con la v2 pura carguen alineados. Si upstream añade entradas, `ref_boost_out` tiene que seguir siendo la última.
6. **Retrocompatibilidad:** con los valores por defecto, las imágenes son **idénticas bit a bit** a la v1.2.5 y a la v2.0.0 sin parche. Lo validamos con recetas reales: 3/3 recetas de doble referencia, 12/12 con máscara + `ref_boost_out`, 30/30 repeticiones estables y 0 OOM.
7. **Test:** `tests/test_ref_attn_bias.py` (`python tests/test_ref_attn_bias.py`).
   - Saca la función `_ref_attn_bias` REAL del `__init__.py` con `ast`, sin importar comfy, y comprueba 12 condiciones: retrocompatibilidad, máscara + `boost_out` y «sin máscara no hace nada».
   - Con el `__init__.py` de upstream sin parche, falla.
   - Nota: en Windows, `pytest` no recoge los tests de este repo (tampoco los de upstream), porque la raíz tiene un `__init__.py`. Se ejecutan sueltos.

**Qué aprendimos con el parche:** el experimento que lo motivó salió **negativo**. Buscaba «tirar de la cara de la referencia y soltar el cuerpo» con una máscara facial y `ref_boost` 2,0 dentro / 0,65 fuera, y no quitaba la ropa que se colaba de la referencia. Lo que sí funciona es subir el `ref_boost` **global** por encima de 1, sin máscara. Antes solo se había barrido entre 0,45 y 1,0. El parche se mantiene porque es inerte por defecto y es la palanca que permite hacer la pregunta, pero no está en producción.

## 2. Cómo usamos el nodo (recetas que no se pueden romper)

- **LoRA de identidad:** `krea2_identity_edit_v1_2` (conradlocke). La «Identity Edit V2» del mismo autor **perdió 12-0** frente a la v1.2 en nuestro duelo ciego del 18-10.
- **Doble referencia escena + persona:**
  - `ref_boost_a` de 1,0 a 2,5 en la ranura A y `ref_boost` 0,65 en la última referencia;
  - grounding de 512 o 768 px;
  - `fit_mode` fit;
  - base Krea 2 Turbo cuantizada (int8 «convrot» mixto);
  - unos 9 pasos, `euler_ancestral` / `sgm_uniform`, cfg 1.
- **Hallazgo:** en doble referencia, **la pose de cabeza la calca la referencia de la ranura A** (83 de 85 tomas con el mismo giro, medido con InsightFace). Ni el prompt, ni subir la fuerza de A de 1 a 2,5, ni la ranura B la mueven. Hoy lo resolvemos con varias referencias de cara en poses distintas.
- **Edición con «latente propio»:** la foto buena va en `latent_image` (no un latente vacío), denoise 0,65, `ref_boost_a` 1,2 y un prompt «change ONLY …». Cambia expresión, fondo o luz sin tocar el cuerpo: 12/12 aceptadas. Por encima de 0,75 se mueve el cuerpo.
- **Problemas abiertos que el nodo podría atacar:**
  1. la pose de cabeza copiada de la referencia;
  2. la ropa de la referencia que se cuela;
  3. el parecido.

## 3. MOSS-Image (el modelo al que queremos llevar el enfoque)

- **Modelo:** OpenMOSS, 16-10-2026. DiT de **un solo flujo**, 5B, 42 capas. Encoder Qwen3.5-9B congelado, con torre de visión. VAE de FLUX.2. Apache 2.0. En ComfyUI es nativo (`MOSSImage`); usamos pesos int8 «convrot» de Base y SFT y el encoder int8 de Comfy-Org, que incluye la visión.
- **Nodo de edición de Ostris:** https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit (17-10). Es el equivalente de este nodo para MOSS:
  - la referencia entra por la visión de Qwen3.5 (`Picture N:`, hasta 384² px);
  - y además como latentes limpios a t=0 detrás del texto, cada uno con su índice RoPE;
  - con caché K/V opcional, que aísla las referencias y va unas 2 veces más rápido.
- **Parche que le hicimos al nodo de Ostris** (`nodes.py`, `_pe`): ComfyUI quitó `MOSSImage.pe_embedder` (PR #17172, 18-10) y calcula la RoPE de otra forma. Sustituimos:

  ```python
  def _pe(dit, ids):
      if hasattr(dit, "pe_embedder"):
          return dit.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()
      import comfy.ldm.flux.math
      ids = ids.unsqueeze(0)
      emb = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000)
                       for i, d in enumerate(dit.rope_axes_dim)], dim=-3)
      return emb.unsqueeze(1).transpose(1, 2).contiguous()
  ```

  Es idéntico al original: diferencia 0,0 en nuestra comprobación.
- **ComfyUI master** (commit `fd7639f47` al 29-10) trae además edición nativa de MOSS: `TextEncodeMOSSImageEdit` (PR #17296) e `image_strength` (#17320).
- **Lo que medimos en MOSS:**
  - Sin una LoRA de edición entrenada con pares, el nodo de Ostris **repinta la referencia 6/6** e ignora la escena del prompt.
  - Entrenamos una LoRA de edición v1 en MOSS Base con ai-toolkit (`edit: true`, `kv_cache: true`, unos 450 pares de recorte de cara → foto de la misma persona). Resultado: **«escena sí, identidad no»**: sigue el prompt pero no se parece.
- **La hipótesis que nos interesa:** ¿un sesgo de atención por referencia al estilo de `ref_boost` / `ref_boost_a` / `ref_boost_out` puede, en MOSS…
  - … con `ref_boost < 1` sin LoRA, frenar el «repinta la referencia» y dejar la escena?
  - … con `ref_boost > 1` y la LoRA v1, subir el parecido?

  Hay que ver:
  - si los bloques de MOSS admiten una máscara o un sesgo aditivo de atención;
  - qué pasa con la caché K/V, que aísla las referencias;
  - qué pasa con SLG (skip-layer guidance en la capa 22, que usamos).

## 4. Restricciones del trabajo

- Investigación y prototipo. No hay GPU en esta fase.
- Cualquier cambio de código va en ramas de este fork, con test, y **con valores por defecto tiene que dar lo mismo bit a bit** que sin el cambio.
- No abras PR al upstream ni a Ostris sin que lo pida el usuario.
- Nada de contenido: las pruebas se describen y se prerregistran, y las lanza el usuario en su máquina.
