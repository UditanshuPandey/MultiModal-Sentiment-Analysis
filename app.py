"""
MVSA Multimodal Sentiment Analysis — Streamlit Application
BERT (text) + ResNet-50 (image) → Gated Cross-Attention Fusion → 3-class Sentiment
Model loaded from Hugging Face Hub
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageFile
import re, os, io, base64, tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Hugging Face Model Config ────────────────────────────────────────────────
# Replace with your Hugging Face repo details
HF_REPO_ID = "TechyCode/multimodal-sentiment-model"  # Your HF repo
HF_MODEL_FILENAME = "best_model.pt"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MVSA Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: rgba(255,255,255,0.85) !important;
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 28px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .glass-card-accent {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 16px;
        padding: 28px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.15);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Sentiment badge */
    .badge-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-neutral {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-negative {
        background: linear-gradient(135deg, #f093fb, #f5515f);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Probability bar */
    .prob-bar-container {
        margin: 8px 0;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        color: rgba(255,255,255,0.75);
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill-pos { background: linear-gradient(90deg,#00b09b,#96c93d); border-radius:8px; height:100%; transition: width 0.8s ease; }
    .prob-bar-fill-neu { background: linear-gradient(90deg,#4facfe,#00f2fe); border-radius:8px; height:100%; transition: width 0.8s ease; }
    .prob-bar-fill-neg { background: linear-gradient(90deg,#f093fb,#f5515f); border-radius:8px; height:100%; transition: width 0.8s ease; }

    /* Architecture node */
    .arch-node {
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.35);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        color: rgba(255,255,255,0.88);
        font-size: 0.88rem;
    }
    .arch-node strong { color: #a78bfa; }

    /* Section heading */
    .section-heading {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    /* Page title */
    .page-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .page-subtitle {
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
        margin-bottom: 28px;
    }

    /* Nav pill in sidebar */
    .nav-pill {
        background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(96,165,250,0.2));
        border: 1px solid rgba(167,139,250,0.4);
        border-radius: 10px;
        padding: 10px 16px;
        color: white;
        font-weight: 500;
        margin: 4px 0;
        cursor: pointer;
    }

    /* Streamlit overrides */
    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.9) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
    }
    .stFileUploader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px dashed rgba(167,139,250,0.4) !important;
        border-radius: 12px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,0.88);
    }

    .info-chip {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        color: #a78bfa;
        margin: 2px 3px;
    }
    hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CONFIG = {
    "BERT_MODEL": "bert-base-uncased",
    "MAX_TEXT_LEN": 128,
    "IMAGE_SIZE": 224,
    "HIDDEN_DIM": 512,
    "DROPOUT": 0.3,
    "NUM_CLASSES": 3,
}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_EMOJIS = ["😠", "😐", "😊"]
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
ASSETS_DIR = Path(__file__).parent / "assets"

TRAINING_HISTORY = {
    "epoch":     [1,2,3,4,5,6,7,8,9,10,11],
    "train_loss":[1.1167,1.0041,0.8439,0.7600,0.7028,0.6327,0.5773,0.5168,0.4708,0.4423,0.4243],
    "val_loss":  [1.0680,0.9055,0.8433,0.8092,0.8046,0.8533,0.8591,0.8893,0.9291,0.9432,0.9575],
    "train_acc": [0.3422,0.5333,0.6625,0.7156,0.7593,0.8086,0.8474,0.8870,0.9210,0.9331,0.9486],
    "val_acc":   [0.4487,0.6211,0.6457,0.6867,0.6895,0.6881,0.7004,0.6826,0.6840,0.6867,0.6881],
    "train_f1":  [0.3459,0.5362,0.6638,0.7164,0.7601,0.8089,0.8476,0.8870,0.9211,0.9331,0.9486],
    "val_f1":    [0.4517,0.6214,0.6441,0.6851,0.6897,0.6858,0.7006,0.6839,0.6864,0.6888,0.6888],
}

# ─── Model Definition ─────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, bert_model_name, hidden_dim):
        super().__init__()
        from transformers import BertModel as BM
        self.bert = BM.from_pretrained(bert_model_name)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0, :])

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for name, param in resnet.named_parameters():
            if not any(x in name for x in ['layer3','layer4','fc']):
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.proj(self.backbone(x).flatten(1))

class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.gate  = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, txt, img):
        t = txt.unsqueeze(1); i = img.unsqueeze(1)
        cross, _ = self.attn(t, i, i)
        cross = self.norm1(cross.squeeze(1) + txt)
        g = self.gate(torch.cat([cross, img], dim=-1))
        return self.norm2(g * cross + (1-g) * img)

class MultimodalSentimentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, D = cfg["HIDDEN_DIM"], cfg["DROPOUT"]
        self.text_enc  = TextEncoder(cfg["BERT_MODEL"], H)
        self.image_enc = ImageEncoder(H)
        self.fusion    = MultimodalFusion(H, D)
        self.classifier = nn.Sequential(
            nn.Dropout(D),
            nn.Linear(H, H//2),
            nn.GELU(),
            nn.Dropout(D/2),
            nn.Linear(H//2, cfg["NUM_CLASSES"]),
        )
    def forward(self, input_ids, attention_mask, images):
        return self.classifier(self.fusion(
            self.text_enc(input_ids, attention_mask),
            self.image_enc(images)
        ))

# ─── Hugging Face Model Loader ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_hf():
    """Load model checkpoint directly from Hugging Face Hub."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download model file from Hugging Face
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )
    
    # Load model
    model = MultimodalSentimentModel(CONFIG).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")

def get_img_transform():
    return transforms.Compose([
        transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

def predict(model, tokenizer, device, text: str, pil_img: Image.Image):
    text_clean = re.sub(r'http\S+|@\w+', '', text).strip() or "no text"
    enc = tokenizer(
        text_clean, max_length=CONFIG["MAX_TEXT_LEN"],
        padding="max_length", truncation=True, return_tensors="pt"
    )
    img_tensor = get_img_transform()(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            img_tensor.to(device)
        )
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def asset_img(name):
    p = ASSETS_DIR / name
    if p.exists():
        return str(p)
    return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px;'>
        <div style='font-size:2.8rem;'>🎭</div>
        <div style='font-size:1.1rem; font-weight:700; color:white;'>MVSA Sentiment</div>
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:4px;'>Multimodal Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔮  Predict", "📊  Training Details"],
        label_visibility="collapsed"
    )

    st.divider()

    # Model source info
    st.markdown("<p style='color:rgba(255,255,255,0.6); font-size:0.82rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>Model Source</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.35);
                border-radius:10px; padding:10px 14px; font-size:0.8rem;'>
        <span style='color:#34d399; font-weight:600;'>🤗 Hugging Face Hub</span><br>
        <span style='color:rgba(255,255,255,0.45); font-size:0.74rem; word-break:break-all;'>
            {HF_REPO_ID}<br>
            File: {HF_MODEL_FILENAME}
        </span>
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='color:rgba(255,255,255,0.35); font-size:0.72rem; line-height:1.6;'>
        <b style='color:rgba(255,255,255,0.5);'>Model Info</b><br>
        BERT-base-uncased<br>
        ResNet-50 (ImageNet V2)<br>
        Gated Cross-Attention Fusion<br>
        3-class · 136M params
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "🔮" in page:

    st.markdown('<div class="page-title">🔮 Multimodal Sentiment Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Combine an image with text for richer, context-aware sentiment analysis.</div>', unsafe_allow_html=True)

    # ── Input columns
    col_img, col_txt = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-heading">📷 Upload Image</div>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Drop an image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if uploaded_img:
            pil_img = Image.open(uploaded_img).convert("RGB")
            st.image(pil_img, use_container_width=True, caption="Input image")
        else:
            st.markdown("""
            <div style='border:2px dashed rgba(167,139,250,0.3); border-radius:12px;
                        height:220px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        color:rgba(255,255,255,0.3);'>
                <div style='font-size:3rem;'>🖼️</div>
                <div style='font-size:0.85rem; margin-top:8px;'>JPG · PNG · WEBP</div>
            </div>""", unsafe_allow_html=True)

    with col_txt:
        st.markdown('<div class="section-heading">💬 Enter Text</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            "Tweet / caption / post accompanying the image",
            placeholder="e.g.  Amazing sunset at the beach! Best day ever 🌅 #happy",
            height=140,
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style='color:rgba(255,255,255,0.35); font-size:0.78rem; margin-top:6px;'>
            URLs and @mentions are automatically stripped before inference.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("✨  Analyse Sentiment", use_container_width=True)

    # ── Prediction
    if run_btn:
        if not uploaded_img:
            st.warning("Please upload an image first.")
        elif not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading model from Hugging Face Hub and running inference..."):
                try:
                    # Load model from Hugging Face
                    model, device = load_model_from_hf()
                    tokenizer = load_tokenizer()
                    pred, probs = predict(model, tokenizer, device, user_text, pil_img)

                    label = LABEL_NAMES[pred]
                    emoji = LABEL_EMOJIS[pred]
                    badge_cls = f"badge-{label.lower()}"

                    st.markdown("---")

                    conf = float(probs[pred]) * 100
                    colors = ["neg","neu","pos"]
                    bars_html = "".join(
                        "<div class=\"prob-bar-container\">"
                        "<div class=\"prob-bar-label\">"
                        f"<span>{LABEL_EMOJIS[i]} {lbl_}</span>"
                        f"<span>{float(prob)*100:.1f}%</span>"
                        "</div>"
                        "<div class=\"prob-bar-bg\">"
                        f"<div class=\"prob-bar-fill-{col_}\" style=\"width:{float(prob)*100:.1f}%\"></div>"
                        "</div></div>"
                        for i, (lbl_, prob, col_) in enumerate(zip(LABEL_NAMES, probs, colors))
                    )
                    result_html = (
                        "<div class=\"glass-card-accent\">"
                        "<div style=\"display:grid;grid-template-columns:1.2fr 1fr 1.2fr;gap:16px;\">"
                        "<div style=\"text-align:center;padding:10px 0;\">"
                        f"<div style=\"font-size:4rem;margin-bottom:6px;\">{emoji}</div>"
                        f"<div class=\"{badge_cls}\">{label}</div>"
                        "<div style=\"color:rgba(255,255,255,0.45);font-size:0.8rem;margin-top:10px;\">Predicted Sentiment</div>"
                        "</div>"
                        "<div style=\"text-align:center;padding:10px 0;\">"
                        "<div style=\"font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#60a5fa);"
                        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;\">"
                        f"{conf:.1f}%</div>"
                        "<div style=\"color:rgba(255,255,255,0.45);font-size:0.8rem;margin-top:6px;\">Confidence</div>"
                        "</div>"
                        "<div style=\"padding:10px 0;\">"
                        "<div style=\"color:rgba(255,255,255,0.55);font-size:0.78rem;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;\">Class Probabilities</div>"
                        + bars_html +
                        "</div></div></div>"
                    )
                    st.markdown(result_html, unsafe_allow_html=True)

                    # — input summary
                    with st.expander("📋 Input summary", expanded=False):
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.image(pil_img, width=200)
                        with ec2:
                            cleaned = re.sub(r'http\S+|@\w+', '', user_text).strip()
                            st.markdown(f"**Raw text:** {user_text}")
                            st.markdown(f"**Cleaned:** {cleaned}")
                            st.markdown(f"**Device:** `{device}`")

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.exception(e)

    else:
        # placeholder hint
        st.info("⬅️  Upload an image and enter text above to run prediction. Model will be downloaded from Hugging Face Hub automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TRAINING DETAILS (KEPT EXACTLY THE SAME AS ORIGINAL)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-title">📊 Training Details & Model Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Architecture overview, training metrics, evaluation curves, and confusion matrix.</div>', unsafe_allow_html=True)

    # ── Top metric cards
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    metrics = [
        ("69.63%", "Test Accuracy"),
        ("0.7006", "Best Val F1"),
        ("69.68%", "Test Wt. F1"),
        ("Epoch 7", "Best Epoch"),
        ("136M", "Total Params"),
    ]
    for col, (val, lbl) in zip([mc1,mc2,mc3,mc4,mc5], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <span class='metric-value'>{val}</span>
            <span class='metric-label'>{lbl}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture + Config
    arch_col, cfg_col = st.columns([1.15, 1], gap="large")

    with arch_col:
        st.markdown("""
        <div class="glass-card">
        <div class="section-heading">🏗️ Model Architecture</div>
        <div class='arch-node'>
            <strong>① Text Branch — BERT-base-uncased</strong><br>
            768-d CLS embedding → Linear(768→512) → LayerNorm → GELU<br>
            <span style='color:rgba(255,255,255,0.45); font-size:0.8rem;'>
                Embeddings + first 10 transformer layers frozen · Last 2 layers fine-tuned
            </span>
        </div>
        <div style='text-align:center; color:rgba(167,139,250,0.5); font-size:0.9rem; margin:2px 0;'>↓ txt_feat [B, 512]</div>

        <div class='arch-node'>
            <strong>② Image Branch — ResNet-50 (ImageNet V2)</strong><br>
            2048-d global pool → Linear(2048→512) → LayerNorm → GELU<br>
            <span style='color:rgba(255,255,255,0.45); font-size:0.8rem;'>
                Stem + layer1/layer2 frozen · layer3/layer4 fine-tuned
            </span>
        </div>
        <div style='text-align:center; color:rgba(167,139,250,0.5); font-size:0.9rem; margin:2px 0;'>↓ img_feat [B, 512]</div>

        <div class='arch-node' style='border-color:rgba(251,191,36,0.4); background:rgba(251,191,36,0.08);'>
            <strong style='color:#fbbf24;'>③ Gated Cross-Attention Fusion</strong><br>
            MultiheadAttention(512, 8 heads) — text queries image<br>
            Sigmoid gate blends cross-attended text with image features<br>
            Dual LayerNorm stabilisation
        </div>
        <div style='text-align:center; color:rgba(167,139,250,0.5); font-size:0.9rem; margin:2px 0;'>↓ fused [B, 512]</div>

        <div class='arch-node'>
            <strong>④ Classifier Head</strong><br>
            Dropout(0.3) → Linear(512→256) → GELU → Dropout(0.15) → Linear(256→3)<br>
            <span style='color:rgba(255,255,255,0.45); font-size:0.8rem;'>
                3-class softmax: Negative · Neutral · Positive
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with cfg_col:
        cfg_items = [
            ("📦 Dataset",        "MVSA-Single · 4,869 samples"),
            ("🔀 Split",          "70% train · 15% val · 15% test"),
            ("🔤 Tokenizer",      "bert-base-uncased · max_len=128"),
            ("🖼️ Image size",     "224 × 224 px (ResNet standard)"),
            ("📐 Hidden dim",     "512"),
            ("💧 Dropout",        "0.30 (head) · 0.15 (inner)"),
            ("⚡ Optimizer",       "AdamW · differential LRs"),
            ("📉 LR (BERT tail)", "1e-5  (0.5 × base LR)"),
            ("📉 LR (others)",    "2e-5"),
            ("⚖️ Weight decay",   "1e-4"),
            ("📊 Loss",           "CrossEntropy + label smoothing 0.1"),
            ("⚖️ Class weights",  "[1.285 · 0.813 · 0.903]"),
            ("🔁 Batch size",     "16"),
            ("🏋️ Max epochs",     "20  (early stopping patience=4)"),
            ("✂️ Grad clip",      "1.0"),
            ("🎯 Metric",         "Weighted F1 (primary)"),
            ("🔢 Seed",           "42"),
        ]
        card_html = (
            '<div class="glass-card">'
            '<div class="section-heading">&#9881;&#65039; Training Configuration</div>'
            + "".join(
                '<div style="display:flex;justify-content:space-between;padding:6px 0;'
                'border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="color:rgba(255,255,255,0.55);font-size:0.83rem;">{k}</span>'
                f'<span style="color:rgba(255,255,255,0.88);font-size:0.83rem;font-weight:500;">{v}</span>'
                '</div>'
                for k, v in cfg_items
            )
            + '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training epoch table
    st.markdown('<div class="section-heading" style="margin-top:8px;">📈 Epoch-by-Epoch Training History</div>', unsafe_allow_html=True)

    import pandas as pd
    hist_df = pd.DataFrame(TRAINING_HISTORY)
    hist_df.columns = ["Epoch","Train Loss","Val Loss","Train Acc","Val Acc","Train F1","Val F1"]
    hist_df["Best?"] = ["" if i != 6 else "⭐ Best" for i in range(len(hist_df))]

    st.dataframe(
        hist_df.style
            .format({c: "{:.4f}" for c in hist_df.columns if c not in ["Epoch","Best?"]})
            .highlight_max(subset=["Val F1"], color="#1e3a2f")
            .highlight_min(subset=["Val Loss"], color="#1e2a3a"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Saved training curves image
    curves_path = asset_img("training_curves.png")
    if curves_path:
        st.markdown('<div class="section-heading" style="margin-top:24px;">📉 Training Curves</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:rgba(255,255,255,0.45); font-size:0.82rem; margin-bottom:12px;'>
            Loss · Accuracy · Weighted F1 over 11 epochs (early stopping at patience=4 after epoch 7 best).
        </div>""", unsafe_allow_html=True)
        st.image(curves_path, use_container_width=True)

    # ── Confusion matrix image
    cm_path = asset_img("confusion_matrix.png")
    if cm_path:
        st.markdown('<div class="section-heading" style="margin-top:24px;">🔢 Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:rgba(255,255,255,0.45); font-size:0.82rem; margin-bottom:12px;'>
            Left: raw counts · Right: row-normalised. Best epoch checkpoint loaded (epoch 7, val F1 = 0.7006).
        </div>""", unsafe_allow_html=True)
        st.image(cm_path, use_container_width=True)

        # Per-class results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="color:rgba(255,255,255,0.6); font-size:0.85rem; font-weight:600; margin-bottom:10px;">Per-class Test Results</div>', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        class_data = [
            ("😠 Negative", "#f5515f", "0.61", "0.66", "0.64", "183"),
            ("😐 Neutral",  "#00f2fe", "0.72", "0.72", "0.72", "288"),
            ("😊 Positive", "#96c93d", "0.72", "0.69", "0.70", "260"),
        ]
        for col, (lbl, color, prec, rec, f1, sup) in zip([pc1,pc2,pc3], class_data):
            col.markdown(f"""
            <div class='glass-card' style='border-color:{color}44; text-align:center; padding:18px;'>
                <div style='font-size:1.5rem; margin-bottom:6px;'>{lbl}</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;'>
                    <div>
                        <div style='font-size:1.2rem; font-weight:700; color:{color};'>{prec}</div>
                        <div style='font-size:0.72rem; color:rgba(255,255,255,0.4);'>Precision</div>
                    </div>
                    <div>
                        <div style='font-size:1.2rem; font-weight:700; color:{color};'>{rec}</div>
                        <div style='font-size:0.72rem; color:rgba(255,255,255,0.4);'>Recall</div>
                    </div>
                    <div>
                        <div style='font-size:1.2rem; font-weight:700; color:{color};'>{f1}</div>
                        <div style='font-size:0.72rem; color:rgba(255,255,255,0.4);'>F1-Score</div>
                    </div>
                    <div>
                        <div style='font-size:1.2rem; font-weight:700; color:rgba(255,255,255,0.7);'>{sup}</div>
                        <div style='font-size:0.72rem; color:rgba(255,255,255,0.4);'>Support</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Sample images
    dist_path = asset_img("class_distribution.png")
    samples_path = asset_img("sample_from_each_class.png")

    if dist_path or samples_path:
        st.markdown("<br>", unsafe_allow_html=True)
        da_col, sa_col = st.columns(2, gap="large")
        if dist_path:
            with da_col:
                st.markdown('<div class="section-heading">📊 Class Distribution</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style='color:rgba(255,255,255,0.45); font-size:0.82rem; margin-bottom:10px;'>
                    Negative: 1217 · Neutral: 1921 · Positive: 1731
                </div>""", unsafe_allow_html=True)
                st.image(dist_path, use_container_width=True)

        if samples_path:
            with sa_col:
                st.markdown('<div class="section-heading">🖼️ Sample Images per Class</div>', unsafe_allow_html=True)
                st.image(samples_path, use_container_width=True)

    # ── Inference demo
    demo_path = asset_img("inference_demo.png")
    if demo_path:
        st.markdown('<div class="section-heading" style="margin-top:24px;">🔬 Inference Demo — Random Test Samples</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:rgba(255,255,255,0.45); font-size:0.82rem; margin-bottom:12px;'>
            6 random test samples with ground-truth vs predicted label and probability distribution.
        </div>""", unsafe_allow_html=True)
        st.image(demo_path, use_container_width=True)

    # ── Dataset info chips
    st.markdown('<div class="section-heading" style="margin-top:24px;">📂 Dataset & Deployment</div>', unsafe_allow_html=True)
    chips = [
        "MVSA-Single", "4,869 paired samples", "Text + Image pairs",
        "3 sentiment classes", "bert-base-uncased", "ResNet-50 (V2 weights)",
        "Trained on CUDA", "PyTorch 2.x", "HuggingFace Transformers",
        "AdamW + CosineAnnealingLR", "Mixed-precision AMP", "Early stopping",
        "Label smoothing 0.1", "Class-weighted CE loss", "best_model.pt saved"
    ]
    st.markdown(" ".join([f"<span class='info-chip'>{c}</span>" for c in chips]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    dep1, dep2 = st.columns(2)
    with dep1:
        st.markdown("""
        <div class='arch-node'>
            <strong>Deployment Package</strong><br>
            <code style='color:#a78bfa;'>outputs/deployment/</code><br>
            ├── best_model.pt &nbsp;(full checkpoint)<br>
            ├── model_meta.json &nbsp;(config + metrics)<br>
            └── inference.py &nbsp;(ready-to-use helper)
        </div>""", unsafe_allow_html=True)
    with dep2:
        st.markdown("""
        <div class='arch-node'>
            <strong>Key Results</strong><br>
            Test Accuracy &nbsp;&nbsp;→ &nbsp;<strong style='color:#34d399;'>69.63%</strong><br>
            Test Weighted F1 → &nbsp;<strong style='color:#34d399;'>0.6968</strong><br>
            Test Macro F1 &nbsp;&nbsp;→ &nbsp;<strong style='color:#60a5fa;'>0.6910</strong><br>
            Best epoch &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;<strong style='color:#fbbf24;'>7 / 11</strong>
        </div>""", unsafe_allow_html=True)
