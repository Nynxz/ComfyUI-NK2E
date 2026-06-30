"""ComfyUI-NK2E: in-context image editing for Krea 2. Community project, not affiliated with Krea.

Appends the reference image's latent as in-context tokens so a KreaEdit LoRA can edit it.
Load the LoRA (ComfyUI format) with the stock LoraLoaderModelOnly node. See README.
"""

import torch
from einops import rearrange

import comfy.patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy_api.latest import ComfyExtension, io, ui


WRAPPER_KEY = "nk2e_incontext"


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
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch, pw=patch)

    ref = ref.to(device=x.device, dtype=x.dtype)
    if ref.ndim == 5:
        rb, rc, rt, rh, rw = ref.shape
        ref = ref.reshape(rb * rt, rc, rh, rw)
    if ref.shape[0] != bs:
        ref = ref[:1].repeat(bs, 1, 1, 1) if bs % ref.shape[0] else ref.repeat(
            bs // ref.shape[0], 1, 1, 1)
    ref = comfy.ldm.common_dit.pad_to_patch_size(ref, (patch, patch))
    hr_, wr_ = ref.shape[-2] // patch, ref.shape[-1] // patch
    refimg = rearrange(
        ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    img = torch.cat((img, refimg), dim=1)
    img = model.first(img)

    context = model._unpack_context(context)
    t = model.tmlp(timestep_embedding(
        timesteps, model.tdim).unsqueeze(1).to(img.dtype))
    tvec = model.tproj(t)
    context = model.txtfusion(
        context, mask=None, transformer_options=transformer_options)
    context = model.txtmlp(context)

    txtlen = context.shape[1]
    tgtlen = h_ * w_
    combined = torch.cat((context, img), dim=1)
    device = combined.device

    def grid(hh, ww, image_index):
        ids = torch.zeros(hh, ww, 3, device=device, dtype=torch.float32)
        ids[..., 0] = float(image_index)
        ids[..., 1] = torch.arange(
            hh, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(
            ww, device=device, dtype=torch.float32)[None, :]
        return ids.reshape(1, hh * ww, 3).repeat(bs, 1, 1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    pos = torch.cat((txtpos, grid(h_, w_, 0), grid(
        hr_, wr_, 1)), dim=1)  # target=0, ref=1
    freqs = model.pe_embedder(pos)

    for block in model.blocks:
        combined = block(combined, tvec, freqs, None,
                         transformer_options=transformer_options)

    final = model.last(combined, t)
    out = final[:, txtlen:txtlen + tgtlen, :]  # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=model.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, model.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _make_wrapper(ref):
    state = {"logged": False}

    def _wrapper(executor, *args, **kwargs):
        model = executor.class_obj
        if not hasattr(model, "txtfusion"):
            return executor(*args, **kwargs)
        transformer_options = args[4] if len(
            args) > 4 else kwargs.get("transformer_options", {})
        if not state["logged"]:
            print(
                f"[NK2E] in-context active  x={tuple(args[0].shape)}  ref={tuple(ref.shape)}", flush=True)
            state["logged"] = True
        try:
            out = _nk2e_incontext_forward(
                model, args[0], args[1], args[2], transformer_options, ref)
        except Exception as e:
            print(
                f"[NK2E] in-context forward failed ({type(e).__name__}: {e}); falling back", flush=True)
            return executor(*args, **kwargs)
        return executor(*args, **kwargs) if out is None else out

    return _wrapper


class NK2EInContextEditNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NK2EInContextEditNode",
            display_name="NK2E In-Context Edit",
            category="NK2E",
            description="Edit the reference image in-context. reference = VAEEncode(source); KSampler gets an empty latent at the output size, denoise 1.0, sampler euler; instruction in the positive prompt.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("reference"),
            ],
            outputs=[
                io.Model.Output("MODEL"),
            ]
        )

    @classmethod
    def execute(cls, model, reference):
        ref = model.model.process_latent_in(reference["samples"])
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               WRAPPER_KEY, _make_wrapper(ref))
        return io.NodeOutput(m)
