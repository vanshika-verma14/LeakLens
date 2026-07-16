# smoke_test.py — does GTR + ielabgroup vec2text round-trip on your CPU?
import sys, types
_res = types.ModuleType("resource")
_res.getrlimit = lambda *a, **k: (0, 0)
_res.setrlimit = lambda *a, **k: None
_res.RLIMIT_AS = _res.RLIMIT_DATA = _res.RLIM_INFINITY = 0
sys.modules["resource"] = _res

import time, torch, vec2text, transformers
from sentence_transformers import SentenceTransformer

print("Loading inverter checkpoints (first run downloads ~2GB)...")
inv = vec2text.models.InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")
cor = vec2text.models.CorrectorEncoderModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")

inv_trainer = vec2text.trainers.InversionTrainer(
    model=inv, train_dataset=None, eval_dataset=None,
    data_collator=transformers.DataCollatorForSeq2Seq(inv.tokenizer, label_pad_token_id=-100),
)
cor.config.dispatch_batches = None
corrector = vec2text.trainers.Corrector(
    model=cor, inversion_trainer=inv_trainer, args=None,
    data_collator=vec2text.collator.DataCollatorForCorrection(tokenizer=inv.tokenizer),
)

print("Loading GTR encoder...")
enc = SentenceTransformer("sentence-transformers/gtr-t5-base")

# --- everything above this line loads ONCE. Everything below runs PER sentence. ---

test_sentences = [
    "The launch code for project atlas is 7719 and the meeting is at midnight.",
    "Please reset the admin password to Th1sIsSecret! before Friday's audit.",
    "Contact Priya Sharma at priya.sharma@acme.com or call 98765 43210 for the invoice.",
    "The quarterly report shows a 12 percent increase in revenue across all regions.",
]

for secret in test_sentences:
    print(f"\nORIGINAL : {secret}")

    emb = enc.encode([secret], convert_to_tensor=True).to("cpu")
    print("Inverting (slow part on CPU — give it a minute or two)...")
    t = time.time()
    recovered = vec2text.invert_embeddings(embeddings=emb, corrector=corrector, num_steps=20)
    print(f"RECOVERED: {recovered[0]}")
    print(f"Took {time.time()-t:.0f}s.")
 
print("\nAll sentences done. Compare ORIGINAL vs RECOVERED above for each.")
