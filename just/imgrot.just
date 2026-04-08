# ── imgrot ─────────────────────────────────────────────────────────────────────
imgrot-raw extras:
  {{py}} ./scripts/datagen/rotation_image.py --multi sharpness=1.0 grid_dim=64,128 speed_schedule=const,acc {{extras}}

imgrot-processed extras:
  {{process}} {{process_defaults}} data.name=imgrot-64-acc,imgrot-128-acc,imgrot-64-const,imgrot-128-const {{extras}}

imgrot-cfm-small extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=imgrot-64-acc,imgrot-64-const unet.base_ch=32,64 {{extras}}

imgrot-cfm-reg extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=imgrot-64-acc unet.base_ch=32 \
    conditioning_reg=1e-6,1e-4,1e-2,1.0 opt.clip_grad_norm=1.0 {{extras}}
