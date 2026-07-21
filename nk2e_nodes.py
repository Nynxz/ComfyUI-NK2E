"""ComfyUI-NK2E: in-context image editing for Krea 2. Not affiliated with Krea."""

import comfy.ldm.common_dit
import comfy.patcher_extension
import node_helpers
import torch
from comfy.ldm.flux.layers import timestep_embedding
from comfy_api.latest import io
from einops import rearrange

# Keep off comfy's "reference_latents": the stock ReferenceLatent node writes it, and
# sharing the key would let unrelated nodes feed refs into NK2E and vice versa.
REF_COND_KEY = "nk2e_refs"
REF_TOKEN_KEY = "nk2e_ref_token"  # bumped per run so KSampler can't reuse a stale sample

WRAPPER_KEY_INCONTEXT = "nk2e_incontext_ref"
WRAPPER_KEY_LEGACY = "nk2e_incontext"


def _nk2e_log(message):
    print(f"[NK2E] {message}", flush=True)


def _shape_list(refs):
    refs = refs if isinstance(refs, (list, tuple)) else [refs]
    return [tuple(ref.shape) for ref in refs]


def _nk2e_incontext_forward(model, x, timesteps, context, transformer_options, refs, log_details=False):
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, _, H_orig, W_orig = x.shape
    patch = model.patch

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    # Each ref becomes its own token block, tagged 1..N on position axis 0 (target = 0),
    # so refs may differ in resolution from the target and from each other.
    refs = refs if isinstance(refs, (list, tuple)) else [refs]
    ref_toks, ref_grids, ref_log = [], [], []
    for i, ref in enumerate(refs):
        input_shape = tuple(ref.shape)
        ref = ref.to(device=x.device, dtype=x.dtype)
        if ref.ndim == 5:
            rb, rc, rt, rh, rw = ref.shape
            ref = ref.reshape(rb * rt, rc, rh, rw)
        if ref.shape[0] != bs:
            ref = ref[:1].repeat(bs, 1, 1, 1) if bs % ref.shape[0] else ref.repeat(bs // ref.shape[0], 1, 1, 1)
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref, (patch, patch))
        ref_grid = (ref.shape[-2] // patch, ref.shape[-1] // patch)
        ref_tok = rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        ref_grids.append(ref_grid)
        ref_toks.append(ref_tok)
        if log_details:
            ref_log.append(
                f"ref{i + 1}=input{input_shape} batched{tuple(ref.shape)} grid={ref_grid} toks={ref_tok.shape[1]}"
            )

    img = model.first(torch.cat((img, *ref_toks), dim=1))

    context = model._unpack_context(context)
    t = model.tmlp(timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(img.dtype))
    tvec = model.tproj(t)
    context = model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = model.txtmlp(context)

    txtlen, tgtlen = context.shape[1], h_ * w_
    combined = torch.cat((context, img), dim=1)
    device = combined.device

    if log_details:
        _nk2e_log(
            f"forward target={tuple(x.shape)} padded_target={(H, W)} target_grid={(h_, w_)} "
            f"target_toks={tgtlen} txt_toks={txtlen} total_refs={len(refs)} total_seq={combined.shape[1]} "
            + ", ".join(ref_log)
        )

    def grid(hh, ww, image_index):
        ids = torch.zeros(hh, ww, 3, device=device, dtype=torch.float32)
        ids[..., 0] = float(image_index)
        ids[..., 1] = torch.arange(hh, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(ww, device=device, dtype=torch.float32)[None, :]
        return ids.reshape(1, hh * ww, 3).repeat(bs, 1, 1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    pos = torch.cat((txtpos, grid(h_, w_, 0), *[grid(hh, ww, i + 1) for i, (hh, ww) in enumerate(ref_grids)]), dim=1)
    freqs = model.pe_embedder(pos)

    for block in model.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    out = model.last(combined, t)[:, txtlen : txtlen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=model.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, model.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _wrapped_forward(get_ref, tag):
    state = {"last_counter": None}

    def _wrapper(executor, *args, **kwargs):
        model = executor.class_obj
        if not hasattr(model, "txtfusion"):
            return executor(*args, **kwargs)
        ref = get_ref()
        if ref is None:
            return executor(*args, **kwargs)
        # Positional index is not stable: c9602625 inserted ref_latents ahead of
        # transformer_options in SingleStreamDiT.forward, so args[4] became the ref list.
        # Take it by keyword, else the first dict in the tail — works on both signatures.
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            transformer_options = next((a for a in args[3:] if isinstance(a, dict)), {})
        counter = _NK2E_REF.get("counter")
        log_details = state["last_counter"] != counter
        if log_details:
            _nk2e_log(f"{tag} active token={counter} x={tuple(args[0].shape)} refs={_shape_list(ref)}")
            state["last_counter"] = counter
        try:
            out = _nk2e_incontext_forward(
                model, args[0], args[1], args[2], transformer_options, ref, log_details=log_details
            )
        except Exception as e:
            _nk2e_log(f"{tag} forward failed ({type(e).__name__}: {e}); falling back")
            return executor(*args, **kwargs)
        return executor(*args, **kwargs) if out is None else out

    return _wrapper


# Refs ride a global rather than the conditioning so the wrapper can be registered
# once: changing the reference then doesn't reload the model.
_NK2E_REF = {"current": None, "counter": 0}


def _conditioning_refs(conditioning):
    """The longest ref chain on any cond entry, i.e. what the last Set Reference built up."""
    refs = []
    for _, cond in conditioning:
        cond_refs = cond.get(REF_COND_KEY)
        if cond_refs is not None and len(cond_refs) > len(refs):
            refs = cond_refs
    return list(refs)


def _get_ref(latent_in):
    raw = _NK2E_REF.get("current")
    return None if raw is None else [latent_in(r) for r in raw]


class NK2EInContextModelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NK2EInContextModelNode",
            display_name="NK2E In-Context (Model)",
            category="NK2E",
            description="In-context edit wrapper; reads the reference from NK2E Set Reference. "
            "Wire: LoraLoaderModelOnly -> here -> KSampler(model).",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output("MODEL")],
        )

    @classmethod
    def execute(cls, model):
        latent_in = model.model.process_latent_in
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY_INCONTEXT,
            _wrapped_forward(lambda: _get_ref(latent_in), "in-context"),
        )
        return io.NodeOutput(m)


class NK2ESetReferenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NK2ESetReferenceNode",
            display_name="NK2E Set Reference",
            category="NK2E",
            description="Adds one reference (VAEEncode of the source) for NK2E In-Context (Model). "
            "Insert on positive: TextEncode -> here -> more NK2E Set Reference nodes if needed -> "
            "KSampler(positive). Chain multiple nodes to accumulate multiple references.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("reference"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # NaN = never cache. Must run every prompt, or the global keeps a previous
        # run's refs and a single-ref graph silently samples with a stale extra ref.
        return float("nan")

    @classmethod
    def execute(cls, conditioning, reference):
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {REF_COND_KEY: [reference["samples"]]}, append=True
        )
        refs = _conditioning_refs(conditioning)
        _NK2E_REF["current"] = refs
        _NK2E_REF["counter"] += 1
        _nk2e_log(f"set reference token={_NK2E_REF['counter']} refs={len(refs)} shapes={_shape_list(refs)}")
        c = node_helpers.conditioning_set_values(conditioning, {REF_TOKEN_KEY: _NK2E_REF["counter"]})
        return io.NodeOutput(c)


# Legacy: reference baked into the cloned model (reloads on ref change). Prefer the pair above.
class NK2EInContextEditNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NK2EInContextEditNode",
            display_name="NK2E In-Context Edit",
            category="NK2E",
            description="Legacy single-node edit (reloads the model on reference change). "
            "reference = VAEEncode(source); KSampler gets an empty latent, denoise 1.0.",
            inputs=[io.Model.Input("model"), io.Latent.Input("reference")],
            outputs=[io.Model.Output("MODEL")],
        )

    @classmethod
    def execute(cls, model, reference):
        ref = model.model.process_latent_in(reference["samples"])
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY_LEGACY,
            _wrapped_forward(lambda: ref, "in-context(legacy)"),
        )
        return io.NodeOutput(m)
