
# ── 1. config ────────────────────────────────────────────────────────
from core.conf import get, active_categories, enabled
print('CONFIG SOURCE:', end=' ')
import core.conf as cc; print(cc._source)
print('categories:', list(active_categories().keys()))
print('exam_meta:', get('exam_meta.student_id'), get('exam_meta.student_name'))
print('hud layout:', get('vision.hud.layout'))
print('gesture.skin_ycrcb:', get('vision.gesture.skin_ycrcb.lower'))
print('color_track.presets.red:', get('vision.color_track.presets.red'))
print('intents keys:', list((get('intents') or {}).keys()))
print()

# ── 2. full replay ───────────────────────────────────────────────────
from pipeline.replay import run_replay_from_config
state = run_replay_from_config(use_llm_response=False)
print()
print('STATE:', state.summary())
