import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import io
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFilter

import timm


LABEL_A = {0: "Real", 1: "AI-Generated"}
LABEL_B = {
    0: "Real",
    1: "SD21",
    2: "SDXL",
    3: "SD3",
    4: "DALLE3",
    5: "Midjourney",
}

class MultiTaskViT(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", num_a=2, num_b=6, dropout=0.10):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Linear(feat_dim, num_a)
        self.head_b = nn.Linear(feat_dim, num_b)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.drop(feats)
        return self.head_a(feats), self.head_b(feats)

def st_image_safe(img, caption=None):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


def list_checkpoints(checkpoints_dir="checkpoints") -> List[str]:
    if not os.path.isdir(checkpoints_dir):
        return []
    files = []
    for f in os.listdir(checkpoints_dir):
        if f.lower().endswith((".pth", ".pt")):
            files.append(os.path.join(checkpoints_dir, f))
    files.sort()
    return files


def topk(probs: np.ndarray, label_map: Dict[int, str], k=3):
    idx = probs.argsort()[::-1][:k]
    return [(label_map[int(i)], float(probs[int(i)])) for i in idx]


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def risk_band(p_ai: float) -> Tuple[str, str]:
    """
    Returns (label, emoji) for user-friendly risk messaging.
    """
    if p_ai < 0.40:
        return "Low AI likelihood", "🟢"
    if p_ai < 0.60:
        return "Uncertain / borderline", "🟡"
    if p_ai < 0.80:
        return "High AI likelihood", "🟠"
    return "Very high AI likelihood", "🔴"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class Loaded:
    model: nn.Module
    transform: nn.Module
    image_size: int
    backbone: str
    meta: Dict


def _pick_ckpt_key(d: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


@st.cache_resource
def load_model_cpu(ckpt_path: str, cpu_threads: int = 2) -> Loaded:
    cpu_threads = max(1, int(cpu_threads))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format not recognized (expected a dict).")

   
    state_key = _pick_ckpt_key(ckpt, ["model_state", "state_dict", "model"])
    if state_key is None:
        raise KeyError("Checkpoint missing model weights. Expected one of: model_state/state_dict/model")

    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    backbone = ckpt.get("backbone") or cfg.get("backbone") or ckpt.get("model_name") or "vit_base_patch16_224"
    image_size = int(ckpt.get("image_size") or cfg.get("image_size") or 224)

    model = MultiTaskViT(backbone_name=backbone, dropout=0.10)
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    meta = {
        "state_key": state_key,
        "epoch": ckpt.get("epoch"),
        "best_val_A": ckpt.get("best_val_A") or ckpt.get("val_A"),
        "best_val_B": ckpt.get("best_val_B") or ckpt.get("val_B"),
        "train_time": ckpt.get("train_time"),
    }

    return Loaded(model=model, transform=transform, image_size=image_size, backbone=backbone, meta=meta)


@torch.inference_mode()
def predict_pil(loaded: Loaded, pil_img: Image.Image) -> Tuple[int, np.ndarray, int, np.ndarray]:
    img = pil_img.convert("RGB")
    x = loaded.transform(img).unsqueeze(0)  # CPU
    logits_a, logits_b = loaded.model(x)

    prob_a = torch.softmax(logits_a, dim=1)[0].cpu().numpy()
    prob_b = torch.softmax(logits_b, dim=1)[0].cpu().numpy()

    pred_a = int(prob_a.argmax())
    pred_b = int(prob_b.argmax())
    return pred_a, prob_a, pred_b, prob_b


def robustness_versions(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    versions = [("original", img)]

    for q in [95, 75, 50]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        im.load()
        versions.append((f"jpeg_q{q}", im))

    for s in [0.75, 0.50]:
        w, h = img.size
        down = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        up = down.resize((w, h), Image.BICUBIC)
        versions.append((f"resize_{int(s * 100)}%", up))

    return versions


def compute_robustness(p_ai_list: List[float], threshold: float) -> Tuple[float, float, float]:
    p = np.array(p_ai_list, dtype=np.float64)
    var = float(np.var(p))

    verdicts = (p >= threshold).astype(int)
    orig = verdicts[0]
    flips = int(np.sum(verdicts[1:] != orig))
    flip_rate = flips / max(1, (len(verdicts) - 1))

    score = 100.0
    score -= min(60.0, var * 200.0)
    score -= min(40.0, flip_rate * 100.0)
    score = float(np.clip(score, 0.0, 100.0))
    return score, var, flip_rate

def _occlusion_core(
    loaded: Loaded,
    pil_img: Image.Image,
    target: str,
    occ_size: int,
    stride: int,
    baseline: str,
    time_guard_s: float,
) -> Tuple[np.ndarray, float, bool]:
    """
    Returns: (heatmap01, base_score, finished)
    finished=False means time guard stopped early and heatmap is partial but still useful.
    """
    img = pil_img.convert("RGB")
    S = loaded.image_size

    pred_a, prob_a, pred_b, prob_b = predict_pil(loaded, img)
    base_p_ai = float(prob_a[1])

    if target == "A":
        base_score = float(prob_a[pred_a])
    elif target == "B":
        base_score = float(prob_b[pred_b])
    else:
        base_score = base_p_ai

    work = img.resize((S, S), Image.BICUBIC)

    if baseline == "gray":
        base_patch = Image.new("RGB", (occ_size, occ_size), (127, 127, 127))
        blurred = None
    else:
        blurred = work.filter(ImageFilter.GaussianBlur(radius=6))
        base_patch = None

    heat = np.zeros((S, S), dtype=np.float32)
    count = np.zeros((S, S), dtype=np.float32)

    t_start = time.time()
    finished = True

    for y in range(0, S - occ_size + 1, stride):
        for x in range(0, S - occ_size + 1, stride):
            if time_guard_s and (time.time() - t_start) > time_guard_s:
                finished = False
                break

            occluded = work.copy()
            if baseline == "gray":
                occluded.paste(base_patch, (x, y))
            else:
                patch = blurred.crop((x, y, x + occ_size, y + occ_size))
                occluded.paste(patch, (x, y))

            pa, p_a, pb, p_b = predict_pil(loaded, occluded)
            p_ai = float(p_a[1])

            if target == "A":
                score = float(p_a[pred_a])
            elif target == "B":
                score = float(p_b[pred_b])
            else:
                score = p_ai

            drop = base_score - score
            if drop < 0:
                drop = 0.0

            heat[y:y + occ_size, x:x + occ_size] += drop
            count[y:y + occ_size, x:x + occ_size] += 1.0

        if not finished:
            break

    heat = heat / (count + 1e-6)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    return heat, base_score, finished


@st.cache_data(show_spinner=False)
def cached_heatmap(
    ckpt_path: str,
    img_hash: str,
    image_size: int,
    explain_target: str,
    occ_size: int,
    stride: int,
    baseline: str,
    time_guard_s: float,
) -> Dict:
    """
    Cache container. We can't cache 'Loaded' directly in cache_data, so we cache params + output.
    The caller computes the heatmap and then returns a serializable dict.
    """
    return {"_placeholder": True}


def compute_or_get_heatmap(
    loaded: Loaded,
    ckpt_path: str,
    img_bytes: bytes,
    pil: Image.Image,
    explain_target: str,
    occ_size: int,
    stride: int,
    baseline: str,
    time_guard_s: float,
) -> Tuple[np.ndarray, float, bool]:
    img_hash = sha256_bytes(img_bytes)

    cached = cached_heatmap(
        ckpt_path=ckpt_path,
        img_hash=img_hash,
        image_size=loaded.image_size,
        explain_target=explain_target,
        occ_size=occ_size,
        stride=stride,
        baseline=baseline,
        time_guard_s=time_guard_s,
    )

    if isinstance(cached, dict) and "heat" in cached:
        heat = np.array(cached["heat"], dtype=np.float32)
        return heat, float(cached["base_score"]), bool(cached.get("finished", True))

    heat, base_score, finished = _occlusion_core(
        loaded=loaded,
        pil_img=pil,
        target=explain_target,
        occ_size=occ_size,
        stride=stride,
        baseline=baseline,
        time_guard_s=time_guard_s,
    )

    payload = _store_heatmap_payload(
        ckpt_path=ckpt_path,
        img_hash=img_hash,
        image_size=loaded.image_size,
        explain_target=explain_target,
        occ_size=occ_size,
        stride=stride,
        baseline=baseline,
        time_guard_s=time_guard_s,
        heat=heat.astype(np.float32),
        base_score=float(base_score),
        finished=bool(finished),
    )
    heat2 = np.array(payload["heat"], dtype=np.float32)
    return heat2, float(payload["base_score"]), bool(payload.get("finished", True))


@st.cache_data(show_spinner=False)
def _store_heatmap_payload(
    ckpt_path: str,
    img_hash: str,
    image_size: int,
    explain_target: str,
    occ_size: int,
    stride: int,
    baseline: str,
    time_guard_s: float,
    heat: np.ndarray,
    base_score: float,
    finished: bool,
) -> Dict:
    return {
        "heat": heat.tolist(),
        "base_score": base_score,
        "finished": finished,
    }


def overlay_heatmap(base_rgb: Image.Image, heatmap01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.array(base_rgb.convert("RGB")).astype(np.float32) / 255.0
    hm = heatmap01.astype(np.float32)

    overlay = np.zeros_like(base)
    overlay[..., 0] = hm
    overlay[..., 1] = hm * 0.8
    overlay[..., 2] = 0.0

    blended = (1 - alpha) * base + alpha * overlay
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def main():
    st.set_page_config(
    page_title="AI Image Authenticity Analyzer",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI Image Authenticity Analyzer\nMulti-Task Vision Transformer\nCPU Optimized Deployment"
    }
)

    st.title("AI Image Authenticity Analyzer")
    st.caption(
        "Single-image detector using your multi-task ViT (Task A: Real vs AI, Task B: generator). "
        "Includes robustness checks and a CPU-safe explanation heatmap."
    )

    st.sidebar.header("Settings")

    ckpts = list_checkpoints("checkpoints")

    if not ckpts:
        st.sidebar.error("No checkpoints found in /checkpoints folder.")
        st.stop()

    default_index = 0
    for i, p in enumerate(ckpts):
        if "best" in os.path.basename(p).lower():
            default_index = i
            break

    ckpt_path = st.sidebar.selectbox(
        "Select model checkpoint",
        ckpts,
        index=default_index,
    )


    cpu_threads = st.sidebar.slider("CPU threads", 1, 4, 2, 1)

    threshold = st.sidebar.slider("AI suspicion threshold P(AI) ≥", 0.50, 0.99, 0.80, 0.01)

    st.sidebar.divider()
    enable_why = st.sidebar.checkbox("Show explanation heatmap", value=True)

    explain_mode = st.sidebar.selectbox(
        "Heatmap explains",
        [
            "AI probability (P(AI))",
            "Decision confidence (Real vs AI)",
            "Generator confidence (source model)",
        ],
        help=(
            "The heatmap marks regions that most affect the selected score.\n\n"
            "• AI probability → P(AI)\n"
            "• Decision confidence → confidence of predicted Real/AI label\n"
            "• Generator confidence → confidence of predicted generator label"
        ),
    )


    why_quality = st.sidebar.selectbox(
        "Explanation speed",
        ["Fast", "Medium", "Strong"],
        help="Stronger = finer heatmap but slower on CPU.",
    )

    time_guard_s = st.sidebar.slider(
        "Heatmap time limit (seconds)",
        3, 25, 12, 1,
        help="Stops early if heatmap takes too long (still shows partial heatmap).",
    )

    st.sidebar.divider()
    enable_robust = st.sidebar.checkbox("Enable Robustness Check", value=True)

    try:
        loaded = load_model_cpu(ckpt_path, cpu_threads=cpu_threads)
        meta_bits = []
        if loaded.meta.get("epoch") is not None:
            meta_bits.append(f"epoch={loaded.meta['epoch']}")
        if loaded.meta.get("best_val_A") is not None:
            meta_bits.append(f"val_A={loaded.meta['best_val_A']}")
        if loaded.meta.get("best_val_B") is not None:
            meta_bits.append(f"val_B={loaded.meta['best_val_B']}")
        meta_str = (" | " + " | ".join(meta_bits)) if meta_bits else ""
        st.info(
            f"Loaded: `{ckpt_path}` | Backbone: `{loaded.backbone}` | Input: `{loaded.image_size}×{loaded.image_size}`{meta_str}"
        )
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    if explain_mode == "Why it looks AI":
        tgt = "AI_PROB"
    elif explain_mode == "Why the final decision was made":
        tgt = "A"
    else:
        tgt = "B"

    st.subheader("Upload one image")
    uploaded = st.file_uploader("Upload image (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

    if not uploaded:
        st.caption("Tip: Try both a real camera photo and an AI-generated image to compare output.")
        return

    img_bytes = uploaded.getvalue()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.write(f"Original size: **{pil.size[0]}×{pil.size[1]}** (internally resized to **{loaded.image_size}×{loaded.image_size}**)")
    st_image_safe(pil, caption="Uploaded image")

    if st.button("Analyze", type="primary"):
        t0 = time.time()
        pred_a, prob_a, pred_b, prob_b = predict_pil(loaded, pil)
        dt = time.time() - t0

        p_ai = float(prob_a[1])
        band, emoji = risk_band(p_ai)

        verdict = "AI-Generated" if p_ai >= threshold else "Likely Real (below threshold)"
        verdict_emoji = ">" if p_ai >= threshold else ""

        st.subheader("Result")
        st.write(f"{verdict_emoji} **{verdict}**")
        st.write(f"{emoji} **{band}**")
        st.write(f"P(AI) = **{p_ai:.6f}** | Threshold = **{threshold:.2f}** | Inference: **{dt*1000:.0f} ms**")

        st.progress(clamp01(p_ai))
        st.caption("Gauge shows P(AI). Threshold only controls the final label.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Task A: Real vs AI")
            st.write(f"Prediction: **{LABEL_A[pred_a]}**")
            st.json({LABEL_A[i]: float(f"{p:.8f}") for i, p in enumerate(prob_a)})

        with col2:
            st.markdown("### Task B: Source model")
            tk = topk(prob_b, LABEL_B, k=3)
            for name, p in tk:
                st.write(f"- **{name}**: {p:.6f}")

        if enable_why:
            st.subheader("Explanation Heatmap (Occlusion)")
            st.caption(
                "We hide parts of the image and measure how much the selected confidence drops. "
                "This is a helpful hint, not proof."
            )

            if why_quality == "Fast":
                occ, stride = 56, 28
            elif why_quality == "Medium":
                occ, stride = 48, 24
            else:
                occ, stride = 40, 20

            with st.spinner("Generating heatmap (cached)…"):
                heat, _base_score, finished = compute_or_get_heatmap(
                    loaded=loaded,
                    ckpt_path=ckpt_path,
                    img_bytes=img_bytes,
                    pil=pil,
                    explain_target=tgt,
                    occ_size=occ,
                    stride=stride,
                    baseline="gray",
                    time_guard_s=float(time_guard_s),
                )

            base = pil.resize((loaded.image_size, loaded.image_size), Image.BICUBIC)
            overlay = overlay_heatmap(base, heat, alpha=0.45)
            st_image_safe(overlay, caption=f"Heatmap overlay — {explain_mode}")

            if not finished:
                st.warning("Heatmap time limit reached — shown heatmap is partial but still informative. Increase time limit for finer detail.")

        if enable_robust:
            st.subheader("Robustness Check (JPEG + resize)")
            if st.button("Run robustness check"):
                rows = []
                p_ai_list = []
                for name, vimg in robustness_versions(pil):
                    pa, proba, pb, probb = predict_pil(loaded, vimg)
                    p_ai2 = float(proba[1])
                    p_ai_list.append(p_ai2)
                    rows.append({
                        "variant": name,
                        "TaskA_pred": LABEL_A[pa],
                        "P(AI)": p_ai2,
                        "TaskB_top1": LABEL_B[int(probb.argmax())],
                        "P(top1_B)": float(probb.max()),
                    })

                score, var, flip_rate = compute_robustness(p_ai_list, threshold)

                st.markdown("### Robustness Score")
                st.write(
                    f"**Score:** {score:.1f}/100  |  "
                    f"**Var(P(AI))**: {var:.6f}  |  "
                    f"**Flip rate**: {flip_rate*100:.1f}%"
                )
                st.dataframe(rows, use_container_width=True)

        st.caption("Disclaimer: This is a probabilistic detector. Use it as guidance, not a legal/forensic proof.")


if __name__ == "__main__":
    main()