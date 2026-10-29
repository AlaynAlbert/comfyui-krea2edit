# Prerregistro: `ref_boost` en la edición de MOSS-Image (29-10-2026)

- **Estado:** borrador para el usuario. No se ha ejecutado nada.
- **Quién lo lanza:** el usuario, en su máquina.
- **Nodo:** `moss_ref_boost/` de la rama `moss-ref-boost`.
- **Contexto y viabilidad:** `docs/ESTUDIO_UPSTREAM_Y_MOSS.md`.
- **Aviso.** MOSS-Sira (D16) está **aplazado** desde el 23-10. Esta tanda solo tiene
  sentido si el usuario lo reabre para esta pregunta.

## 1. Pregunta

¿Un sesgo de atención del objetivo hacia la referencia (`ref_boost`, el mismo escalar
de krea2edit) cambia la edición de MOSS en la dirección buscada? Dos casos:

- **H2, con la LoRA de edición v1** (hoy da «escena sí, identidad no»): con
  `ref_boost` > 1, ¿sube el **parecido** sin perder la escena ni pasar a copiar la
  referencia?
- **H1, sin LoRA de edición** (hoy «repinta la referencia», 6 de 6): con `ref_boost` < 1,
  ¿deja de repintar y **sigue la escena**?

## 2. Antes de lanzar (sin GPU, obligatorio)

1. **Datos de la LoRA v1** (`utilidades/runs/moss_image/20261021_lora_moss_edit_ref_v1/`):
   - todos los objetivos tienen su referencia con el mismo nombre, sin contar la extensión;
   - `flip_x` y `flip_y` están a `false` en la config sellada.

   Si falla algo, **H2 no se prueba**: la «identidad no» tendría una causa de datos y el
   piloto no la arreglaría.
2. **Instalar el nodo.** Enlazar `moss_ref_boost/` en `custom_nodes` (ver su README) y
   reiniciar el ComfyUI del photoset con su `.bat` (`--disable-metadata`).
3. **Fijar por escrito, antes del primer render:**
   - el commit de ComfyUI;
   - Ostris `8f9b50c` + parche `_pe`;
   - el commit de `moss-ref-boost`;
   - los sha de las LoRA y de la referencia.

## 3. Montaje fijo (igual en todos los brazos)

| pieza | valor |
|---|---|
| Vía de edición | **Ostris** `MOSSImageOstrisEditModelPatch`, `kv_cache` encendido. Es la que se midió el 21 y el 22-10; nunca se mezcla con la nativa (estudio §6.1). |
| Modelo | MOSS Base int8 convrot (la base de la LoRA v1), encoder int8 Comfy-Org con visión, VAE FLUX.2 |
| Muestreo, SLG (capa 22), resolución | los de `validacion/val_edit.py` (no verificado aquí: se copian del script) |
| Referencia | Sira `03_portrait_face_clean.jpg`, ≤ 262 144 px y lados ×16, como en `val_edit.py` |
| Prompt | «The same woman as in Picture 1. » + la toma (`val_edit.py`) |
| Tomas del piloto | las **3 primeras** de `val_edit.py planes`, en su orden |
| Semilla | la misma por toma en todos los brazos |
| LoRA de edición v1 | **ck1440** a 1,0 (siguió la escena 6/6 el 22-10) |
| Tarjeta | la RTX 5070 Ti para todo, controles incluidos |
| Telemetría | `py -3.14 gpu_telemetria.py registrar --log <run>/GPU.tsv --pid-padre <PID>` |

El nodo va detrás del parche de Ostris y delante del SLG, con `ref_boost_a`,
`ref_boost_out` y la máscara sin tocar: hay una sola referencia.

## 4. Brazos

| brazo | LoRA v1 | `ref_boost` | papel | ¿se lanza? |
|---|---|---|---|---|
| **E0** | sí | 1,0 | control de la familia E (C0 = edición tal cual) | sí |
| **E-2,5** | sí | 2,5 | H2, dosis alta primero (D34) | sí |
| **E-1,5** | sí | 1,5 | H2 | sí |
| E-0,65 | sí | 0,65 | contraste. La predicción es menos parecido; no cambia ninguna decisión | **solo si el usuario lo pide** |
| **S0** | no | 1,0 | control de la familia S (lo que repintó 6/6) | sí |
| **S-0,65** | no | 0,65 | H1 | sí |
| S-1,5 / S-2,5 | no | 1,5 / 2,5 | S0 ya repinta 6/6; más sesgo solo puede repintar más. No cambian ninguna decisión | **solo si el usuario lo pide** |

Lo pedido son 6 brazos además de los 2 controles. **Recomiendo lanzar 3**: E-2,5, E-1,5
y S-0,65. Los demás se dejan anotados por si el usuario los quiere (§8).

**Control de bytes.** Antes del piloto se renderiza la toma 1 de E0 **sin el nodo**. Su
sha tiene que coincidir con el de E0 con el nodo a 1,0. Si no coincide, se para todo: el
nodo no es inerte.

## 5. Piloto y regla de parada

- **Par.** Brazo frente a su control (E-x frente a E0, S-x frente a S0), misma toma y
  misma semilla. **3 pares por brazo.**
- **Rejilla.** Se monta con `rejilla` (AGENTES), a ciegas, con las letras A/B
  equilibradas, y se verifica con `verify --estricto`. El cribado físico (`fisico/`) **solo
  informa**: es una comparativa A/B.
- **Ejes por par** (A / B / IGUAL):
  1. **sigue la escena**: lugar, ropa, encuadre y luz que pide el prompt;
  2. **parecido**: ¿es la persona de la referencia (Sira)?;
  3. **no copia la referencia**: no repite su encuadre, fondo, luz ni pose;
  4. **global**.
- **Regla de parada del piloto** (por brazo, con los 3 pares votados):
  - **si el control gana en «global» en ≥ 2 de 3, el brazo se para** y no se renderiza
    nada más de él;
  - además, un brazo E se para si pierde «sigue la escena» en ≥ 2 de 3: ha vuelto a
    repintar;
  - un brazo S se para si no gana «no copia la referencia» en ≥ 2 de 3.
- **Si sobrevive alguno:**
  - se completa a **6 pares**: las 3 tomas restantes de `val_edit.py`, más 3 renders del
    control;
  - la lectura prerregistrada es **«mejora»** si el brazo gana «global» en ≥ 4 de 6 **y**
    la mediana de ArcFace a la vara (`11_headshot_coleta_upright.jpg`) queda ≤ la del
    control − 0,03 (la misma vara que en `val_edit.py`);
  - si por el camino ya no puede llegar a 4 de 6, se para (D34).
- **Medidas** (`val_edit.py medir`): ArcFace a la vara y a la referencia que se pasa, y
  `fisico/`. Registran; no deciden. La decisión es del ojo.

## 6. Minutos de GPU

- **Base de cálculo:** ~51 s por render con referencia de 512² (`val_edit.py`, 22-10, en
  la 4070 Ti SUPER).
  - La 5070 Ti, sin medir: supongo lo mismo ±20 % (no verificado).
  - La máscara del sesgo: de +10 a +30 % por paso en los brazos (no medido).
  - Redondeo a ~1 min por render.

| tanda | renders | min de GPU |
|---|---|---|
| Control de bytes (E0 toma 1 sin nodo) | 1 | ~1 |
| Controles E0 + S0 (3 + 3) | 6 | ~5 |
| **Brazos recomendados** E-2,5, E-1,5, S-0,65 (3 × 3) | 9 | ~10 |
| Cargas de modelo/LoRA y precálculo de refs | — | ~3 |
| **Piloto recomendado** | **16** | **~20** |
| Brazos opcionales E-0,65, S-1,5, S-2,5 (3 × 3) | +9 | +~10 |
| Piloto con los 6 brazos pedidos | 25 | ~30 |
| Ampliación a 6 pares, por brazo superviviente (+3 brazo, +3 control la primera vez) | +3 a +6 | +3 a +6 |

Hay que comprobar el tiempo con `personaje estimar` si cubre MOSS (no verificado) y anotar
los segundos reales por render en el `LEEME` del run.

## 7. Qué decisión cambia

- **Si gana un brazo E** («mejora» a 6 pares):
  - `ref_boost` entra como candidato en la receta de edición MOSS;
  - el siguiente paso es la generalización: Tyxa, con la vara del núcleo ≤ 0,412, igual
    que la ENMIENDA 1 del 22-10;
  - se aplaza entrenar una LoRA de edición v2 (~3–4 h de GPU por entrenamiento).
- **Si no gana ningún brazo E:**
  - `ref_boost` queda **descartado en MOSS** para el parecido;
  - el parecido tiene que venir del entrenamiento: otra LoRA de edición, con pares
    revisados, o E+P con la v5;
  - el nodo se archiva sin usar.
- **Si S-0,65 gana:**
  - la vía «sin LoRA de edición» deja de estar cerrada;
  - siguiente pregunta: barrido fino entre 0,65 y 1 a 3 pares.
- **Si S-0,65 pierde:** H1 descartada; la edición sin LoRA queda cerrada para siempre,
  sin más tandas.
- **Ninguna otra decisión pendiente invalida la tanda**, salvo el §2.1 (datos de la v1),
  que por eso va antes.

## 8. Fuera de este piloto (anotado, no se lanza)

- **E-0,65, S-1,5 y S-2,5** (ver §4).
- **Sesgo solo en la pasada positiva** (cond), no en la negativa: la CFG lo amplifica.
- **`image_strength` nativo** sobre los tokens de visión. Solo existe en la vía nativa, y
  cambiar de vía es otro experimento.
- **La vía nativa (`TextEncodeMOSSImageEdit`)** frente a la de Ostris. Con SLG no son bit a
  bit iguales (estudio §6.1).
