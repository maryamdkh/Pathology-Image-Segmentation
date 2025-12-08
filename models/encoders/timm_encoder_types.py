# ==================================================================================================
#                               ENCODER MODEL REFERENCE GUIDE
# ==================================================================================================
# This guide helps you choose an encoder backbone for UNet++ when training pathology segmentation
# models using TIMM architectures. Each model family below is widely used in vision tasks but has
# different strengths, computational demands, and feature extraction behaviors. Pathology images
# often contain fine textures (nuclei), multi-scale structures (glands), and large contextual areas
# (tissue architecture). The notes below summarize which architectures capture these patterns most
# effectively.
#
# ==================================================================================================
#                                        1. SWIN TRANSFORMER
# --------------------------------------------------------------------------------------------------
# • Hierarchical Vision Transformer with shifted-window self-attention.
# • Captures both local texture (small windows) and larger context (shifted windows).
# • More efficient than ViT because attention is limited to windows.
# • Excellent for histopathology because tissue patterns are highly local, but require context.
#
# Naming:
#   swin_<size>_patch4_window7_224
#       size: tiny / small / base / large     -> capacity & depth
#       patch4: patch size used to tokenize input
#       window7: attention window size
#       224: pretraining image resolution
#
# S3 variants (swin_s3_*):
#   • “Stage 3” improvements: stronger normalization, deeper blocks.
#
# Best use cases:
#   • Medium/large pathology datasets
#   • Tasks requiring balanced local and global modeling
#
# ==================================================================================================
#                                      2. SWIN TRANSFORMER V2
# --------------------------------------------------------------------------------------------------
# • Improved Swin with:
#     – cosine attention scaling (more stable for high resolution)
#     – post-normalization
#     – continuous positional bias
# • Much better at *very high-resolution feature learning*, which is important for pathology images.
#
# Naming:
#   swinv2_<size>_window8_256:
#       window8 / window16: self-attention window size
#       256 / 192 / 384: pretraining resolution
#
# CR variants (swinv2_cr_*):
#   • Models pretrained on enormous datasets (IN22k + custom corpora).
#   • MUCH stronger generalization but heavier.
#
# ns variants:
#   • "No-Scale" versions; training stability improvement.
#
# Best use cases:
#   • Large WSI patches (512–1024px)
#   • Datasets with large variations in color and morphology
#
# ==================================================================================================
#                                           3. MAXViT
# --------------------------------------------------------------------------------------------------
# • Multi-axis attention: combines:
#       – Local window attention (captures nuclei/gland texture)
#       – Grid attention (captures tissue-level arrangement)
# • Very strong multi-scale modeling → ideal for pathology segmentation.
#
# Naming:
#   maxvit_<size>_<training_source>_<resolution>
#       size: tiny / small / base / large / xlarge / nano / pico
#       training_source: tf = TensorFlow weights, rw = timm real-world weights, pm = PyTorch meta
#       resolution: 224 / 256 / 384 / 512
#
# Larger resolutions imply:
#   • stronger long-range modeling
#   • higher training GPU cost
#
# Best use cases:
#   • When you need *strong global + local contextual modeling* (e.g., gland segmentation).
#
# ==================================================================================================
#                                       4. MAXViT-RMLP
# --------------------------------------------------------------------------------------------------
# • Replaces convolutional feedforward layers with RMLP (Residual MLP).
# • Lighter, faster, still retains MaxViT’s multi-axis attention.
# • Good accuracy/performance tradeoff.
#
# Naming:
#   maxvit_rmlp_<size>_rw_256
#       rmlp: indicates RMLP feedforward layers
#       rw: timm pretrained weights
#
# Best use cases:
#   • Limited GPU memory
#   • Small/medium datasets with high image resolution
#
# ==================================================================================================
#                                         5. MAXXViT (Hybrid)
# --------------------------------------------------------------------------------------------------
# • Hybrid architecture with CNN + Vision Transformer blocks.
# • Better early texture modeling than pure transformer (good for nuclei).
# • Better global reasoning than CNN alone.
#
# Naming:
#   maxxvit_rmlp_<size>_rw_256
#       Similar size naming conventions to MaxViT.
#
# Best use cases:
#   • Noisy or heterogeneous pathology images
#   • When features need sharper local texture encoding
#
# ==================================================================================================
#                                     6. MAXXViT V2 (Improved)
# --------------------------------------------------------------------------------------------------
# • Efficiency upgrades + improved RMLP + refined attention patterns.
# • Very strong accuracy/compute ratio.
#
# Naming:
#   maxxvitv2_<size>_rw_256
#
# Best use cases:
#   • Small to medium hardware
#   • High accuracy requirement with limited memory
#
# ==================================================================================================
#                                         7. ConvNeXt V2
# --------------------------------------------------------------------------------------------------
# • Modernized CNN that mimics transformer training stability.
# • Pure convolutional architecture: extremely good at capturing local texture.
# • Works very well in pathology, which relies heavily on local patterns.
# • Much lighter than transformers and easy to train.
#
# Naming:
#   convnextv2_<size>   (atto, femto, pico, nano, tiny, small, base, large, huge)
#       size: model depth + width
#       smaller sizes are extremely efficient for segmentation.
#
# Best use cases:
#   • Small datasets
#   • Low GPU memory
#   • Tasks dominated by local morphology (nuclei, mitosis detection)
#
# ==================================================================================================
#                       PRACTICAL ENC0DER SELECTION FOR PATHOLOGY SEGMENTATION
# --------------------------------------------------------------------------------------------------
#   • For **strongest accuracy** → MaxViT Large / SwinV2 Large / MaxXViT V2
#   • For **best balance (recommended)** → Swin Base / SwinV2 Base / MaxViT Small or Base
#   • For **low compute** → ConvNeXtV2 Tiny/Nano, MaxViT Tiny/Pico
#   • For **fine-grained nuclei tasks** → ConvNeXtV2 / MaxXViT
#   • For **large gland/tissue regions** → SwinV2 / MaxViT (larger window/global attention)
#
# ==================================================================================================
# END OF MODEL FAMILY GUIDE
# ==================================================================================================


TIMM_ENCODERS = {
    # ----------------------------- SWIN -----------------------------
    "timm_swin_tiny": "swin_tiny_patch4_window7_224",
    "timm_swin_small": "swin_small_patch4_window7_224",
    "timm_swin_base": "swin_base_patch4_window7_224",

    "timm_swin_base_patch4_window7_224": "swin_base_patch4_window7_224",
    "timm_swin_base_patch4_window12_384": "swin_base_patch4_window12_384",
    "timm_swin_large_patch4_window7_224": "swin_large_patch4_window7_224",
    "timm_swin_large_patch4_window12_384": "swin_large_patch4_window12_384",

    "timm_swin_s3_base_224": "swin_s3_base_224",
    "timm_swin_s3_small_224": "swin_s3_small_224",
    "timm_swin_s3_tiny_224": "swin_s3_tiny_224",

    "timm_swin_small_patch4_window7_224": "swin_small_patch4_window7_224",
    "timm_swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",

    # ----------------------------- SWIN V2 -----------------------------
    "timm_swinv2_base_window8_256": "swinv2_base_window8_256",
    "timm_swinv2_base_window12_192": "swinv2_base_window12_192",
    "timm_swinv2_base_window12to16_192to256": "swinv2_base_window12to16_192to256",
    "timm_swinv2_base_window12to24_192to384": "swinv2_base_window12to24_192to384",
    "timm_swinv2_base_window16_256": "swinv2_base_window16_256",

    "timm_swinv2_cr_base_224": "swinv2_cr_base_224",
    "timm_swinv2_cr_base_384": "swinv2_cr_base_384",
    "timm_swinv2_cr_base_ns_224": "swinv2_cr_base_ns_224",

    "timm_swinv2_cr_giant_224": "swinv2_cr_giant_224",
    "timm_swinv2_cr_giant_384": "swinv2_cr_giant_384",

    "timm_swinv2_cr_huge_224": "swinv2_cr_huge_224",
    "timm_swinv2_cr_huge_384": "swinv2_cr_huge_384",

    "timm_swinv2_cr_large_224": "swinv2_cr_large_224",
    "timm_swinv2_cr_large_384": "swinv2_cr_large_384",

    "timm_swinv2_cr_small_224": "swinv2_cr_small_224",
    "timm_swinv2_cr_small_384": "swinv2_cr_small_384",
    "timm_swinv2_cr_small_ns_224": "swinv2_cr_small_ns_224",
    "timm_swinv2_cr_small_ns_256": "swinv2_cr_small_ns_256",

    "timm_swinv2_cr_tiny_224": "swinv2_cr_tiny_224",
    "timm_swinv2_cr_tiny_384": "swinv2_cr_tiny_384",
    "timm_swinv2_cr_tiny_ns_224": "swinv2_cr_tiny_ns_224",

    "timm_swinv2_large_window12_192": "swinv2_large_window12_192",
    "timm_swinv2_large_window12to16_192to256": "swinv2_large_window12to16_192to256",
    "timm_swinv2_large_window12to24_192to384": "swinv2_large_window12to24_192to384",

    "timm_swinv2_small_window8_256": "swinv2_small_window8_256",
    "timm_swinv2_small_window16_256": "swinv2_small_window16_256",

    "timm_swinv2_tiny_window8_256": "swinv2_tiny_window8_256",
    "timm_swinv2_tiny_window16_256": "swinv2_tiny_window16_256",
    
    # ----------------------------- MAXViT -----------------------------
    "timm_maxvit_base_tf_224": "maxvit_base_tf_224",
    "timm_maxvit_base_tf_384": "maxvit_base_tf_384",
    "timm_maxvit_base_tf_512": "maxvit_base_tf_512",

    "timm_maxvit_large_tf_224": "maxvit_large_tf_224",
    "timm_maxvit_large_tf_384": "maxvit_large_tf_384",
    "timm_maxvit_large_tf_512": "maxvit_large_tf_512",

    "timm_maxvit_small_tf_224": "maxvit_small_tf_224",
    "timm_maxvit_small_tf_384": "maxvit_small_tf_384",
    "timm_maxvit_small_tf_512": "maxvit_small_tf_512",

    "timm_maxvit_tiny_pm_256": "maxvit_tiny_pm_256",
    "timm_maxvit_tiny_rw_224": "maxvit_tiny_rw_224",
    "timm_maxvit_tiny_rw_256": "maxvit_tiny_rw_256",
    "timm_maxvit_tiny_tf_224": "maxvit_tiny_tf_224",
    "timm_maxvit_tiny_tf_384": "maxvit_tiny_tf_384",
    "timm_maxvit_tiny_tf_512": "maxvit_tiny_tf_512",

    "timm_maxvit_xlarge_tf_224": "maxvit_xlarge_tf_224",
    "timm_maxvit_xlarge_tf_384": "maxvit_xlarge_tf_384",
    "timm_maxvit_xlarge_tf_512": "maxvit_xlarge_tf_512",

    "timm_maxvit_nano_rw_256": "maxvit_nano_rw_256",
    "timm_maxvit_pico_rw_256": "maxvit_pico_rw_256",

    # -------------------------- MAXViT RMLP --------------------------
    "timm_maxvit_rmlp_base_rw_224": "maxvit_rmlp_base_rw_224",
    "timm_maxvit_rmlp_base_rw_384": "maxvit_rmlp_base_rw_384",
    "timm_maxvit_rmlp_nano_rw_256": "maxvit_rmlp_nano_rw_256",
    "timm_maxvit_rmlp_pico_rw_256": "maxvit_rmlp_pico_rw_256",
    "timm_maxvit_rmlp_small_rw_224": "maxvit_rmlp_small_rw_224",
    "timm_maxvit_rmlp_small_rw_256": "maxvit_rmlp_small_rw_256",
    "timm_maxvit_rmlp_tiny_rw_256": "maxvit_rmlp_tiny_rw_256",

    # ---------------------------- MAXXViT -----------------------------
    "timm_maxxvit_rmlp_nano_rw_256": "maxxvit_rmlp_nano_rw_256",
    "timm_maxxvit_rmlp_small_rw_256": "maxxvit_rmlp_small_rw_256",
    "timm_maxxvit_rmlp_tiny_rw_256": "maxxvit_rmlp_tiny_rw_256",

    # --------------------------- MAXXViT V2 ---------------------------
    "timm_maxxvitv2_nano_rw_256": "maxxvitv2_nano_rw_256",
    "timm_maxxvitv2_rmlp_base_rw_224": "maxxvitv2_rmlp_base_rw_224",
    "timm_maxxvitv2_rmlp_base_rw_384": "maxxvitv2_rmlp_base_rw_384",
    "timm_maxxvitv2_rmlp_large_rw_224": "maxxvitv2_rmlp_large_rw_224",

    # --------------------------- CONVNEXT V2 ---------------------------
    "timm_convnextv2_tiny": "convnextv2_tiny",
    "timm_convnextv2_atto": "convnextv2_atto",
    "timm_convnextv2_base":"convnextv2_base",
    "timm_convnextv2_femto":"convnextv2_femto",
    "timm_convnextv2_huge": "convnextv2_huge",
    "timm_convnextv2_large": "convnextv2_large",
    "timm_convnextv2_nano": "convnextv2_nano",
    "timm_convnextv2_pico": "convnextv2_pico",
    "timm_convnextv2_small": "convnextv2_small",
    "timm_convnextv2_tiny": "convnextv2_tiny",
}