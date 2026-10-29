# Estudio: upstream de krea2edit y `ref_boost` en MOSS-Image (29-10-2026)

Rama de trabajo: `moss-ref-boost`, que sale de `ref-boost-out` (`3bbd696`). No se toca ni
`main` ni `ref-boost-out`.

Fuentes leídas:

| fuente | versión |
|---|---|
| upstream `lbouaraba/comfyui-krea2edit` | `13ef60f` + refs `pull/*` |
| `krea2edit-trainer` | `56bf37b` |
| `ostris/ComfyUI-MOSS-Image-Ostris-Edit` | `8f9b50c` |
| ComfyUI master | `fd7639f` |
| `ostris/ai-toolkit` | `b69b983` (`extensions_built_in/diffusion_models/moss_image/`) |

Lo marcado **«no verificado»** no se ha podido comprobar aquí: no hay GPU ni pesos.

## Resumen

1. **Upstream.** No hay commits después de `13ef60f` (comprobado con `git fetch upstream`).
   Por eso no se crea `rebase-ref-boost-out`. Lo útil está en los **PR abiertos**:
   #22, #30 y #31, más #20, que es solo un cambio del README. Los tres chocan ya con
   la v2.0.0 pura, porque parten de la v1.2.5.
2. **El parche aguanta.** `upstream/main` = `13ef60f`, que es la base de `ref-boost-out`:
   `python tests/test_ref_attn_bias.py` sale 12/12 OK (ver §2).
3. **MOSS admite `ref_boost`.** Sus bloques ya aceptan una máscara aditiva de atención
   (`comfy/ldm/moss_image/model.py:62,82,140,148`), y ComfyUI la usa para
   `image_strength`. Una ref en caché K/V no estorba: el sesgo va en la pasada del
   objetivo, nunca en la que llena la caché. SLG en la capa 22 solo quita esa capa.
   - Prototipo: `moss_ref_boost/`. Con 1,0 devuelve el mismo `MODEL`, idéntico bit a bit.
   - Tests: 36 comprobaciones en CPU y 12 de 12 mutaciones cazadas.
4. **Prueba propuesta** (`docs/PRERREGISTRO_MOSS_REF_BOOST.md`). Piloto de 3 pares por
   brazo frente a su control, con la misma semilla. Recomendado: **3 brazos (E-2,5, E-1,5 y
   S-0,65) y 2 controles, 16 renders, unos 20 min de GPU**. Con los 6 brazos que se pidieron
   son 25 renders, unos 30 min.

## 1. Upstream: commits y PR

### 1.1 Commits de `upstream/main` posteriores a `13ef60f`

**Ninguno.** `git log 13ef60f..upstream/main` sale vacío (29-10, 22:10 UTC). Las etiquetas
son v1.2.4, v1.2.5 y v2.0.0 (`43a8b2a` → `13ef60f`).

### 1.2 PR de upstream

| PR | estado | qué cambia | entradas | ¿píxeles por defecto? | choque con nuestro parche | veredicto |
|---|---|---|---|---|---|---|
| #31 `5582c0f` (easis, 10-10) — `boost_attention: split` | abierto | Calcula `ref_boost` sin máscara densa: una atención completa más una sobre las filas del objetivo, con dos grupos de cabezas extra por ref sesgada; después reconstruye `out' = (AV + Σ(b−1)A(V·1_S)) / (1 + Σ(b−1)A·1_S)`. Mide de 70,1 a 46,8 s con dos refs (5060 Ti). | nueva `boost_attention` (`split` por defecto / `dense_bias`) | Con `ref_boost` = 1,0, no (no hay override). Con 0,65 (nuestra receta), **sí**, por redondeo de coma flotante; `dense_bias` deja lo de siempre. | **Sí, de fondo.** Su `_boost_sets` copia el fallo antiguo (`if b == 1.0: continue`) y no tiene `boost_out`. Además, el override supone conjuntos de columnas disjuntos: para portar `ref_boost_out` hay que dar «ref menos máscara» y «máscara» como dos conjuntos. Choca en 5 hunks de `__init__.py` (3 frente a la v2.0.0 pura). | **vigilar**: solo gana velocidad. Si upstream lo fusiona, hay que portar `boost_out` y fijar `dense_bias` mientras se valida. |
| #30 `673614b` (Ilya Egoshin, 9-10) — hasta 4 refs y `ref_boosts` por ref | abierto (lo reclama el issue #36) | `source_image_c/_d`, `image_c/_d` en el encode y `ref_boosts` como lista separada por comas, en orden de conexión, que pisa `ref_boost` y `ref_boost_a`. | 3 entradas nuevas al final del patch y 2 en el encode | No (todo vacío = antes). | Lógico no: `ref_boost_out` sigue siendo de la última ref. De orden, sí: sus entradas van «al final», igual que `ref_position*` (v2) y la nuestra. Al portarlo, `ref_boost_out` tiene que seguir la ÚLTIMA. 4 hunks (3 frente a la v2.0.0). | **vigilar, con prueba candidata**: ataca la **pose de cabeza calcada** (ver §1.4). Sin entrenar más allá de 2 refs. |
| #22 `82f3a14` (thipokKub, 16-08) — Flash Attention + `ref_boost` | abierto | Con `--use-flash-attention`, el sesgo se rehace desde el `lse` de flash, sin máscara. Antes caía a SDPA con máscara: aviso en cada bloque y pasos ~2 veces más lentos. | ninguna | Solo con flash activo y `ref_boost` ≠ 1 (error máximo 5e-4). | Mismo problema de `_bias_to_sets` (lee la máscara densa: funcionaría con `boost_out`, porque agrupa por valor de logit). Choca en `CHANGELOG` y `pyproject`. | **no**, salvo que usemos `--use-flash-attention` (no consta en las recetas). |
| #20 `98c1f92` | abierto | Enlace del README | — | No | No | no |
| #9, #15 | cerrados (#15 ya está dentro: `target_latent`) | — | — | — | — | — |

Choques comprobados con `git merge` de prueba sobre `ref-boost-out` y sobre
`upstream/main`. Las ramas temporales se borraron y no se subió nada.

### 1.3 Issues con algo aplicable

| issue | qué aporta | problema nuestro | veredicto |
|---|---|---|---|
| #35 (26-10) | Si el ancho o el alto no es múltiplo de 16, salen rayas en el borde derecho o inferior. La causa es `pad_to_patch_size(..., padding_mode="replicate")` (`__init__.py:239,254`); con `circular`, como en el `_forward` nativo de Krea 2, desaparecen. Se ve sobre todo con `beside`. | ninguno de los tres; es un artefacto | **vigilar**. Nuestras recetas usan lados múltiplos de 16 (no verificado en todas). Si se prueba `beside`, mantener ×16. |
| #32 (11-10) | `SamplerSASolver` (SamplerCustom), `ref_boost` 0,9, 8 pasos, `simple`, para seguir mejor la instrucción en ediciones «Change/Replace X». | parecido y ropa, de forma indirecta | **vigilar**: anécdota sin medida. |
| #14 (22-07) | El autor: la expresión forma parte de lo que se copia. Sugiere `ref_boost` de 0,7 a 0,9, describir la cara en concreto y `ref_boost_mask` para frenar solo la cara. El que lo reporta: funciona con algunas semillas, no de forma fiable. | pose y expresión calcadas | **vigilar**. Encaja con nuestro hallazgo: la máscara no separa de forma fiable. |
| #10 (20-07) | `isolate_identities` (aislar la atención entre refs) y enrutado de caras. El autor lo probó en su banco de «fuga entre dos personas» y ayudó «poco», así que lo quitó. | ropa (fuga entre refs) | **no**: ya medido como débil por el autor. |
| #7 (18-07) | Con la ControlNet de profundidad de Krea 2, la identidad se tuerce o vuelve a la fuente; el autor sugiere bajar el peso o acortar el tramo. | pose de cabeza (forzarla con profundidad) | **vigilar**: sin cifras. |
| #11, #36, #27 | Roles fijos escena/persona, más refs (→ #30) y dos caras sobre dos personas (sin respuesta). | — | vigilar |

### 1.4 Ya está en la v2.0.0 y no consta que se haya probado: `ref_position_a = beside`

La v2.0.0 (`13ef60f`, nuestra base) trae `ref_position` y `ref_position_a`
(`__init__.py:56`, `_place_refs`). Con **`beside`**, la referencia sigue pasando
identidad, pero deja de estar «clavada» al lienzo de salida en RoPE.

- **Por qué importa.** Nuestro hallazgo es que la pose de cabeza la calca la ranura A
  (83 de 85 tomas). Encaja con un anclaje espacial: con `aligned`, cada token del
  objetivo tiene «enfrente» al mismo token de A.
- **Qué falta.** `docs/CONTEXTO_AGENTE.md` no dice si se probó `ref_position_a` =
  `beside` (no verificado).
- **Por qué es la primera palanca.** Es la candidata más barata contra la pose
  calcada: no hay código que escribir.
  - Vigilar #35: con `beside`, lados ×16.
  - Con `aligned` intacto, los píxeles son idénticos a los de hoy.

### 1.5 Entrenador público (`krea2edit-trainer` `56bf37b`)

No trae mejoras del nodo. Lo útil son **guardas de datos** que aplican tal cual a nuestra
**LoRA de edición MOSS v1**, entrenada con ai-toolkit `moss_image`, `edit: true` y
`use_raw_control_images`. Atacan el **parecido** («escena sí, identidad no»):

- **Fallo 1: pares sueltos.** ai-toolkit empareja por el nombre del fichero, sin la
  extensión. Un objetivo sin fuente se queda con su caption y **se entrena como T2I, sin
  avisar** (README del entrenador, «Dataset layout»).
  - El entrenador de Krea lo audita al arrancar.
  - El arch `moss_image` de ai-toolkit no lo hace: busqué `audit|pair` y no hay nada
    (no verificado en el resto de ai-toolkit).
- **Fallo 2: volteos.** Con `flip_x`/`flip_y`, ai-toolkit voltea el objetivo pero no
  las imágenes de control en crudo, y desincroniza cada par.
  - `moss_image.py:198` activa `use_raw_control_images` para `edit`.
  - No he visto ninguna guarda contra el volteo en `moss_image.py` (no verificado).
- **Fallo 3: estirar referencias.** Estirar referencias de proporciones mezcladas le
  enseñó a la v1 de Krea un «antiaplastado» en las caras (`krea2_edit.py:502`).
  - En MOSS no aplica: `_encode_ref_latents` (`moss_image.py:423-452`) conserva la
    proporción.
- **Acción sin GPU.** Comprobar en `runs/moss_image/20261021_lora_moss_edit_ref_v1/`
  (en `utilidades`) que los 454 pares emparejan todos y que `flip_x`/`flip_y` están a
  `false`. Si fallan, el «identidad no» tiene una causa de datos, no de modelo.
  → **adoptar como comprobación**: CPU, minutos.

## 2. Estado del parche sobre `upstream/main`

- `upstream/main` = `13ef60f` = base de `ref-boost-out`, así que no hay nada que rebasar.
  No se crea `rebase-ref-boost-out`: el encargo la pedía solo si había commits nuevos.
- Con torch 2.14.1 CPU (conda-forge, sin GPU ni comfy):
  - `python tests/test_ref_attn_bias.py`: **12/12 OK** en `ref-boost-out`. Con el
    `__init__.py` de `13ef60f` cae (`TypeError: ... 'boost_out'`): el test sí prueba el parche.
  - `python tests/test_pre_encode.py` (upstream): **5/5 PASS**.
  - `python tests/test_wrapper_signature.py` (upstream): **no se puede ejecutar aquí**, porque
    importa `comfy` desde la carpeta de ComfyUI en la que esté instalado el nodo (necesita
    `comfy_aimdo`, `comfy_kitchen`, etc.). Falla igual en la v2.0.0 pura, así que no es una
    regresión. Hay que ejecutarlo en la máquina del usuario.
- **Si upstream fusiona #30 o #31**, hay que tocar el parche:
  - **#30:** `ref_boost_out` se aplica a `boosts[-1]` después de `_resolve_boosts`.
    Hay que reordenar las entradas para que `ref_boost_out` quede la última.
  - **#31:** hay que portar `boost_out` a `_boost_sets`, con conjuntos disjuntos, y el
    arreglo del `if b == 1.0`. Mientras tanto, fijar `boost_attention = dense_bias`.

## 3. Tabla pieza por pieza: krea2edit ↔ MOSS

| pieza | krea2edit (Krea 2) | MOSS nativo (ComfyUI `fd7639f`) | MOSS Ostris (`8f9b50c`) |
|---|---|---|---|
| **Secuencia** | `[texto \| refs… \| objetivo]` (`__init__.py:271`) | `[imagen \| texto \| refs]` (`model.py:253`); con caché, `[imagen \| texto]` y las K/V de las refs se añaden como claves (`model.py:78-80`) | igual que el nativo (`nodes.py:309`, `:369`, `_attention` `:203-205`) |
| **Timestep de las refs** | el mismo `tvec` ruidoso que el objetivo (`__init__.py:265,300`) | t=0, con su propia modulación (`model.py:37-53`, `:240-243`) | t=0 (`nodes.py:305-306`, `:338-339`) |
| **RoPE por referencia** | 3 ejes `(frame, h, w)`; texto en 0, objetivo en frame 0, ref i en frame i+1 (`__init__.py:279-287`); `ref_position*` mueve h/w (`:56`) | 4 ejes `(r, f, h, w)`; ref i en `r=i+1`, `f=0`, rejilla centrada (`model.py:186-191`); imagen y texto desplazados `f_offset=−text_len` para dejar las refs en f=0 (`model.py:233`); las refs no se escalan con Scale ROPE (`:193-195`) | `r=i+1`, `f=text_len`, igual que la imagen (`nodes.py:143-152,182`); relativo, equivale al nativo |
| **Grounding por visión** | Qwen3-VL con plantilla propia; `grounding_px` 768, entrenado de 384 a 768 (`__init__.py:517-523`) | Qwen3.5 `Picture N:`, ≤ 384² (`nodes_moss_image.py:127-132`); **`image_strength`** = `log(s)` sobre los tokens de visión dentro del texto (`:135-145` → `context_attention_bias`, `model_base.py:2843`) | Qwen3.5 `Picture N:`, ≤ 384² (`nodes.py:52,111-129`); **no lleva `image_strength`**: el `forward` de Ostris descarta `context_attention_bias` (`nodes.py:461-484`) |
| **Caché K/V de refs** | no; las refs atienden a todo en cada paso | sí, para `ai-toolkit` (`index_timestep_zero_isolated`) y `Moss-Image-Edit`: se calculan una vez en `extra_conds` (`model_base.py:2853-2857`, `preprocess_refs` `model.py:198-211`); con `kv_cache off`, refs en la secuencia, a t=0, atendiendo a todo | sí con `kv_cache` encendido (por defecto): pasada de solo refs y caché por huella (`nodes.py:324-351,378-387,467-484`) |
| **SLG en la capa 22** | — (cfg 1) | `patches_replace` `("single_block", 22)` en la pasada negativa (`model.py:264-267`, `nodes_slg.py:132-151`); `ref_kv` se precalcula **sin** saltar la 22 | se salta la 22 también en la pasada de solo refs, y la huella incluye las capas saltadas (`nodes.py:278-280,331-332`): **no es bit a bit igual al nativo con SLG** |
| **¿Sesgo o máscara aditiva?** | sí: `block(..., attn_bias)` (`__init__.py:300`) | **sí**: `Block.forward(..., mask)` → `optimized_attention(..., mask=mask)` (`model.py:82,140,148`); hoy solo para `context_attention_bias`, de forma `(B,1,1,Lk)` (`model.py:245-249`) | no recibe máscara; llama a `optimized_attention(..., skip_reshape=True, transformer_options=…)` (`nodes.py:207-209`) |
| **Dónde inyectar el sesgo por ref** | en `_ref_attn_bias` (filas del objetivo × columnas de cada ref) | en las llamadas de atención DiT con `Lk = img+txt+Σref`, columnas `[img+txt, Lk)`, filas `[0, img)`. Vía `optimized_attention_override` (`attention.py:215-245`) instalado por un wrapper `APPLY_MODEL`: sirve para las dos vías sin tocarlas | ídem: el override llega por `transformer_options` |

## 4. Viabilidad del sesgo en MOSS

### 4.1 ¿Se puede? Sí

- **La máscara ya está ahí.** Los bloques de MOSS aceptan una máscara aditiva y ComfyUI
  ya la usa en `image_strength` (`log` sobre las claves de visión). Lo nuevo es la parte
  de las **columnas de referencia** y restringirla a las **filas de imagen**, como en
  krea2edit.
- **La caché K/V.**
  - Nativo: la caché se llena en `extra_conds` **sin** `transformer_options`, así que el
    override no la ve.
  - Ostris: su pasada de solo refs tiene `Lq = Lk = Lr` y el override la deja pasar
    intacta (test «pasada solo-refs … bit a bit»).
  - En los dos casos, la caché es exactamente la de la LoRA y el sesgo solo cambia cuánto
    miran las filas del objetivo a esas claves. La caché de Ostris se indexa por contenido
    de las refs y no depende de `ref_boost`, así que sigue siendo válida.
- **SLG en la capa 22.** Esa capa no se ejecuta en la pasada negativa. Las otras 41
  llevan el sesgo en las dos pasadas, igual que krea2edit, que lo aplica en todas.
  - Sesgar solo la positiva haría que la CFG (3) amplificara la diferencia: variante
    **sin probar**, fuera del piloto.

### 4.2 Prototipo `moss_ref_boost/` (rama `moss-ref-boost`)

- **Ficheros.**
  - `moss_ref_boost/core.py`: torch puro, sin comfy.
  - `nodes.py`: el nodo `MOSSImageRefBoost`.
  - `README.md`.
- **Entradas:** `ref_boost`, `ref_boost_a`, `ref_boost_mask` y `ref_boost_out`, este
  último al final. Semántica idéntica a krea2edit, incluido el arreglo del
  `if b == 1.0`.
- **Mecanismo.** Un wrapper `APPLY_MODEL` saca el layout de cada llamada: `H·W` de `x`,
  la longitud del texto de `c_crossattn`, y las refs de `ref_latents` (rejilla por ref) o
  de `ref_kv` (un bloque). Después instala un `optimized_attention_override` en una
  **copia** de `transformer_options`. El override:
  - suma el sesgo solo en las llamadas DiT (`Lk == img+txt+Σref`), encima de la máscara
    nativa si la hay (`image_strength`);
  - deja intactas las llamadas de texto y la pasada de solo refs;
  - encadena un override previo;
  - devuelve a cada bloque nativo su `ComfyAttention` preferida, que `wrap_attn` descarta
    en cuanto hay override (`attention.py:229-242`).
- **Por defecto.** Con todo a 1,0, `patch()` devuelve **el mismo objeto `MODEL`**: no hay
  clon ni wrapper, así que la salida es idéntica bit a bit.
- **Tests:** `tests/test_moss_ref_boost.py`, 36 comprobaciones.
  - El sesgo: filas, columnas, dos refs, máscara + `out`, el arreglo y la semántica
    «multiplicar y renormalizar».
  - El layout de las dos vías.
  - El override: pasa intacto lo que no es DiT, suma con `image_strength`, layout de
    Ostris y encadenado.
  - Un MOSS de juguete de extremo a extremo:
    - por defecto, el mismo objeto y bit a bit igual;
    - wrapper + override = máscara `log(b)` puesta a mano, con y sin `image_strength`;
    - la caché K/V no se toca;
    - se conserva la atención preferida;
    - con `kv_cache` apagado;
    - con SLG.
- **Mutaciones:** `tests/mutar_moss_ref_boost.py` quita o rompe cada una de las 12 líneas
  clave en una copia y comprueba que el test cae: **12/12 cazadas**. Una salió de la
  propia revisión: el aviso de `ref_boost_a` saltaba con `ref_boost_a` = 1,0 en la vía
  nativa con caché; está arreglado y tiene su test.

### 4.3 Límites del prototipo

- **Nativo con `kv_cache`.** Las refs llegan como un único bloque `ref_kv`, sin la
  división por ref ni la rejilla. Por eso:
  - `ref_boost` va a todas;
  - `ref_boost_a`, la máscara y `ref_boost_out` se ignoran, con aviso.

  Con Ostris o con `kv_cache` apagado, va todo. Para el piloto, con una sola ref, da igual.
- **Coste.** Con un valor ≠ 1, las atenciones DiT llevan una máscara densa
  `(1,1,Lq,Lk)`, unos 75 MB a 1 MP con una ref de 1 MP, y caen a un kernel con máscara.
  Es lo mismo que ya paga `image_strength`.
  - Sobrecoste no medido; estimo de +10 a +30 % por paso.
  - Si importa, el truco «split» de #31 se porta tal cual.
- **Atención preferida.** No verificado con pesos que declaren una `ComfyAttention`
  (comfy_kitchen_int8 / sol). Con los pesos convrot int8 que usamos, no consta que la
  declaren.
- **El override no se ve en `wrap_attn`.** Con `container_function` activo en el backend
  global, sus llamadas toman los tensores en lugar del contenedor. No cambia los números
  (no verificado en GPU).

## 5. Las dos hipótesis, a priori

- **H1: `ref_boost < 1` sin LoRA de edición frena «repinta la referencia».**
  - **Lo que hay.** El 21-10, R0 y R1 repintaron 6 de 6 con la ref a ~1 MP. La base de
    MOSS **nunca vio** referencias en la secuencia (`moss_image.py:189-191`): las K/V de
    la ref son tokens que el modelo no sabe usar, y copiarlos es la salida fácil.
  - **Expectativa.** Bajar `b` reduce su peso, pero lo más probable es que pase de
    «repinta» a «ignora la ref» sin un punto medio con parecido: la escena llega, la
    identidad no.
  - **Qué lo decidiría.** Una dosis donde la escena se siga en ≥ 2 de 3 y el parecido
    supere al C0 sin ref.
  - **Probabilidad estimada:** baja.
- **H2: `ref_boost > 1` con la LoRA v1 sube el parecido.**
  - **Lo que hay.**
    - La v1 sigue la escena 6 de 6, pero se aleja de la vara: 0,515 frente a 0,404 del
      C0 (`utilidades/runs/moss_image/20261021_lora_moss_edit_ref_v1/validacion/LEEME.md`).
    - En Krea2, subir el `ref_boost` global por encima de 1 fue justo lo que funcionó.
  - **Riesgo.** Que a 2,5 vuelva a repintar.
  - **Probabilidad estimada:** media. Merece el piloto.
- **Palancas sin código que conviene tener en cuenta** (fuera del piloto, a vigilar):
  - **`image_strength` nativo**, sobre los tokens de visión: es el canal semántico, no el
    de apariencia, y solo existe en la vía nativa.
  - **El checkpoint 1440 frente al 2880** de la v1.

## 6. Riesgos

1. **Mezcla de vías.** Ostris y nativo no son bit a bit iguales con SLG (§3, fila SLG).
   El piloto se hace entero por **una sola vía**: Ostris, que es la que midió el 21 y el
   22-10.
2. **Ostris sobre master.** El `extra_conds` nativo también precalcula `ref_kv` para las
   refs de Ostris (`model_base.py:2853-2857`, sin `reference_latents_method`). Es trabajo
   tirado y el `forward` de Ostris lo ignora. El prototipo prefiere `ref_latents`. No
   verificado en GPU que no cambie bytes: lo comprueba el control de bytes del
   prerregistro.
3. **Tarjeta.** Los controles del 21 y el 22-10 salieron de la 4070 Ti SUPER. Ningún duelo
   mezcla tarjetas (REGLAS_AGENTES), así que se renderizan de nuevo los controles en la
   5070 Ti.
4. **Kernel.** Con el sesgo, la atención usa máscara, y eso puede cambiar el backend
   frente al control. Hay que medir los segundos por imagen. La diferencia de píxeles por
   kernel es mucho menor que el efecto buscado, pero el control de bytes «nodo a 1,0» no
   la cubre (no verificado).
5. **Las PR de upstream** pueden fusionarse con otra semántica (#31). Si pasa, se revisa
   el parche.

## 7. Recomendación

1. **Upstream:**
   - **adoptar:** nada de código ahora;
   - **probar sin código:** `ref_position_a = beside` contra la pose calcada (lados ×16,
     #35);
   - **vigilar:** #30, por la pose (varias refs de cara con `ref_boosts`), y #31, por
     velocidad, con portado previsto de `boost_out`;
   - **no:** #22, #10.
2. **Comprobaciones de datos de la LoRA MOSS v1:** emparejado por nombre y `flip_x`/`flip_y`
   (§1.5). CPU, minutos. Van antes del piloto, porque una LoRA mal emparejada invalida H2.
3. **MOSS:** lanzar el piloto del prerregistro con la variante recomendada.
   - Primero E-2,5 y E-1,5 frente a E0 (LoRA v1 a 1,0, ck1440).
   - Sin LoRA, solo S-0,65 frente a S0.
   - 16 renders, unos 20 min de GPU.
   - Si E-1,5 o E-2,5 gana, se completa a 6 pares. Si no, el `ref_boost` en MOSS queda
     descartado y el parecido tiene que venir del entrenamiento.
