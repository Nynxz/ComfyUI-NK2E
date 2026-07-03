"""ComfyUI-NK2E: in-context image editing for Krea 2. Not affiliated with Krea."""
import torch
from einops import rearrange

import comfy.patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy_api.latest import io
import node_helpers

WRAPPER_KEY = "nk2e_incontext"
MODEL_REF_KEY = "nk2e_incontext_ref"


def _nk2e_incontext_forward(model, x, timesteps, context, transformer_options, ref):
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

    ref = ref.to(device=x.device, dtype=x.dtype)
    if ref.ndim == 5:
        rb, rc, rt, rh, rw = ref.shape
        ref = ref.reshape(rb * rt, rc, rh, rw)
    if ref.shape[0] != bs:
        ref = ref[:1].repeat(bs, 1, 1, 1) if bs % ref.shape[0] else ref.repeat(bs // ref.shape[0], 1, 1, 1)
    ref = comfy.ldm.common_dit.pad_to_patch_size(ref, (patch, patch))
    hr_, wr_ = ref.shape[-2] // patch, ref.shape[-1] // patch
    refimg = rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    img = model.first(torch.cat((img, refimg), dim=1))

    context = model._unpack_context(context)
    t = model.tmlp(timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(img.dtype))
    tvec = model.tproj(t)
    context = model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = model.txtmlp(context)

    txtlen, tgtlen = context.shape[1], h_ * w_
    combined = torch.cat((context, img), dim=1)
    device = combined.device

    def grid(hh, ww, image_index):
        ids = torch.zeros(hh, ww, 3, device=device, dtype=torch.float32)
        ids[..., 0] = float(image_index)
        ids[..., 1] = torch.arange(hh, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(ww, device=device, dtype=torch.float32)[None, :]
        return ids.reshape(1, hh * ww, 3).repeat(bs, 1, 1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    pos = torch.cat((txtpos, grid(h_, w_, 0), grid(hr_, wr_, 1)), dim=1)  # target=0, ref=1
    freqs = model.pe_embedder(pos)

    for block in model.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    out = model.last(combined, t)[:, txtlen:txtlen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=model.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, model.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _wrapped_forward(get_ref, tag):
    state = {"logged": False}

    def _wrapper(executor, *args, **kwargs):
        model = executor.class_obj
        if not hasattr(model, "txtfusion"):
            return executor(*args, **kwargs)
        ref = get_ref()
        if ref is None:
            return executor(*args, **kwargs)
        transformer_options = args[4] if len(args) > 4 else kwargs.get("transformer_options", {})
        if not state["logged"]:
            print(f"[NK2E] {tag} active  x={tuple(args[0].shape)}  ref={tuple(ref.shape)}", flush=True)
            state["logged"] = True
        try:
            out = _nk2e_incontext_forward(model, args[0], args[1], args[2], transformer_options, ref)
        except Exception as e:
            print(f"[NK2E] {tag} forward failed ({type(e).__name__}: {e}); falling back", flush=True)
            return executor(*args, **kwargs)
        return executor(*args, **kwargs) if out is None else out

    return _wrapper


# Reference rides a side channel; the wrapper is registered once, so changing the
# reference does not reload the model.
_NK2E_REF = {"current": None, "counter": 0}


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
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               MODEL_REF_KEY, _wrapped_forward(lambda: _get_ref(latent_in), "in-context"))
        return io.NodeOutput(m)


def _get_ref(latent_in):
    raw = _NK2E_REF.get("current")
    return None if raw is None else latent_in(raw)


class NK2ESetReferenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NK2ESetReferenceNode",
            display_name="NK2E Set Reference",
            category="NK2E",
            description="Sets the reference (VAEEncode of the source) for NK2E In-Context (Model). "
                        "Insert on positive: TextEncode -> here -> KSampler(positive).",
            inputs=[io.Conditioning.Input("conditioning"), io.Latent.Input("reference")],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, reference):
        _NK2E_REF["current"] = reference["samples"]
        _NK2E_REF["counter"] += 1
        c = node_helpers.conditioning_set_values(conditioning, {"nk2e_ref_token": _NK2E_REF["counter"]})
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
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               WRAPPER_KEY, _wrapped_forward(lambda: ref, "in-context(legacy)"))
        return io.NodeOutput(m)
