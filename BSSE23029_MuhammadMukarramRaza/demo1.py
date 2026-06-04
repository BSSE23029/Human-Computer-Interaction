
import warnings; warnings.filterwarnings('ignore')

# ── 1. imports ────────────────────────────────────────────────────────
from core.provider import LIVE
from core.executor import validate_tabs, run_tab_pipeline
from core.state import SessionState
from pipeline.turn import run_turn, process_turn
from pipeline.stages import STAGE_REGISTRY

print('=== 1. STAGE REGISTRY ===')
print('Registered stages:', list(STAGE_REGISTRY.keys()))

# ── 2. LIVE provider init ────────────────────────────────────────────
s = SessionState()
LIVE.init_session(s)
print()
print('=== 2. LIVE PROVIDER ===')
print('session:', LIVE.session)
print('vision running:', LIVE.vision.running)
print('vision stale:', LIVE.vision.is_stale())

# ── 3. tab validation ─────────────────────────────────────────────────
print()
print('=== 3. TAB VALIDATION ===')
warnings_list = validate_tabs()
if warnings_list:
    for w in warnings_list: print('  WARN:', w)
else:
    print('  All enabled tabs have at least one input stage. OK')

# ── 4. run_turn chat ──────────────────────────────────────────────────
print()
print('=== 4. run_turn (chat tab) ===')
ctx = run_turn('chat', text='I have a major assignment due tomorrow')
print('  text:     ', ctx.get('text'))
print('  source:   ', ctx.get('source'))
print('  tier:     ', ctx.get('tier', '?'), ctx.get('scale',{}).get('emoji',''))
print('  category: ', ctx.get('category', '?'))
print('  response: ', ctx.get('response','')[:60])
print('  errors:   ', ctx.get('errors', []))
print('  session turns:', LIVE.session.n_turns)

# ── 5. interdependency: sentiment disabled ────────────────────────────
print()
print('=== 5. INTERDEPENDENCY: sentiment._skipped → scale uses LLM ===')
from core.conf import CFG
CFG['sentiment']['enabled'] = False
ctx2 = run_turn('chat', text='I feel completely hopeless and alone')
sent = ctx2.get('sentiment', {})
scl  = ctx2.get('scale', {})
print('  sentiment._skipped:', sent.get('_skipped'))
print('  scale tier:', scl.get('tier'), '(LLM scored text directly)')
print('  scale._skipped:', scl.get('_skipped'))
CFG['sentiment']['enabled'] = True

# ── 6. interdependency: classify disabled → trajectory skips transitions
print()
print('=== 6. INTERDEPENDENCY: classify._skipped → trajectory skips transitions ===')
CFG['categories']['enabled'] = False
ctx3 = run_turn('chat', text='I need help with my fees and my mental health')
cls  = ctx3.get('classify', {})
traj = ctx3.get('trajectory', {})
print('  classify._skipped:', cls.get('_skipped'))
print('  trajectory.trend:', traj.get('trend'))
print('  trajectory.transition:', traj.get('transition'), '(should be None)')
CFG['categories']['enabled'] = True

# ── 7. all disabled → pure LLM ──────────────────────────────────────
print()
print('=== 7. ALL STAGES OFF → pure LLM ===')
for k in ['sentiment','scale','categories','responses']:
    if k in CFG: CFG[k]['enabled'] = False
ctx4 = run_turn('chat', text='I am struggling with everything')
print('  response[:60]:', ctx4.get('response','')[:60])
print('  errors:', ctx4.get('errors',[]))
for k in ['sentiment','scale','categories','responses']:
    if k in CFG: CFG[k]['enabled'] = True

# ── 8. legacy process_turn still works ───────────────────────────────
print()
print('=== 8. LEGACY process_turn ===')
s2 = SessionState()
r  = process_turn(s2, text='I cannot pay my rent')
print('  legacy result keys:', list(r.keys()))
print('  support:', r.get('support',{}).get('primary'))
print('  response[:50]:', r.get('response','')[:50])

# ── 9. replay still works ─────────────────────────────────────────────
print()
print('=== 9. REPLAY (config log, first 3 turns) ===')
from core.conf import CFG as _CFG
orig_log = _CFG.get('replay',{}).get('log',[])
_CFG['replay']['log'] = orig_log[:3]
from pipeline.replay import run_replay_from_config
state = run_replay_from_config(use_llm_response=False)
print('  replay turns:', state.n_turns)
print('  lowest tier:', state.lowest_tier)
_CFG['replay']['log'] = orig_log