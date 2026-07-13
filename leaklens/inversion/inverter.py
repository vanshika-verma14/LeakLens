"""GTR encoder + vec2text inverter, wrapped so it loads once and stays on CPU.

This is the reusable core the smoke test proved out. The heavy checkpoints (~2GB) load
lazily on first use, so importing this module is cheap and never forces a download. The
target laptop is CPU-only (docs/DECISIONS.md), so everything is pinned to CPU; do not
route to GPU/NPU.

Recovery is deliberately *partial* on high-entropy input — that is the finding, not a
bug (DECISIONS D5). Callers score it with metrics.py, not exact-match.
"""
import leaklens._compat  # noqa: F401  — MUST be first: resource stub + CPU pin

import torch
import transformers
import vec2text
from sentence_transformers import SentenceTransformer

# Model ids locked in DECISIONS D2/D3 — a public inverter exists only for this encoder.
ENCODER = "sentence-transformers/gtr-t5-base"
INVERSION = "ielabgroup/vec2text_gtr-base-st_inversion"
CORRECTOR = "ielabgroup/vec2text_gtr-base-st_corrector"

DEFAULT_NUM_STEPS = 20  # vec2text correction steps; fewer = faster/rougher (DEVELOPMENT.md)


class Inverter:
    """Encode text to GTR embeddings and invert embeddings back to text, on CPU.

    Construction is cheap; the encoder and vec2text corrector build on first use and are
    reused thereafter, so a process pays the load cost at most once.
    """

    def __init__(self, *, encoder: str = ENCODER, inversion: str = INVERSION,
                 corrector: str = CORRECTOR, num_steps: int = DEFAULT_NUM_STEPS,
                 device: str = "cpu"):
        self.encoder_name = encoder
        self.inversion_name = inversion
        self.corrector_name = corrector
        self.num_steps = num_steps
        self.device = device
        self._encoder = None
        self._corrector = None

    def _ensure_loaded(self) -> None:
        """Build the encoder + vec2text corrector once (mirrors the smoke test)."""
        if self._corrector is not None:
            return
        inv = vec2text.models.InversionModel.from_pretrained(self.inversion_name)
        cor = vec2text.models.CorrectorEncoderModel.from_pretrained(self.corrector_name)
        inv_trainer = vec2text.trainers.InversionTrainer(
            model=inv, train_dataset=None, eval_dataset=None,
            data_collator=transformers.DataCollatorForSeq2Seq(
                inv.tokenizer, label_pad_token_id=-100),
        )
        cor.config.dispatch_batches = None
        self._corrector = vec2text.trainers.Corrector(
            model=cor, inversion_trainer=inv_trainer, args=None,
            data_collator=vec2text.collator.DataCollatorForCorrection(tokenizer=inv.tokenizer),
        )
        self._encoder = SentenceTransformer(self.encoder_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return GTR embeddings for `texts`, on CPU (the inverter's expected input)."""
        self._ensure_loaded()
        return self._encoder.encode(texts, convert_to_tensor=True).to(self.device)

    def invert(self, embeddings: torch.Tensor, num_steps: int | None = None) -> list[str]:
        """Reconstruct text from GTR embeddings via vec2text's correction loop."""
        self._ensure_loaded()
        steps = self.num_steps if num_steps is None else num_steps
        return vec2text.invert_embeddings(
            embeddings=embeddings.to(self.device), corrector=self._corrector, num_steps=steps)

    def roundtrip(self, texts: list[str], num_steps: int | None = None) -> list[str]:
        """Convenience: encode then invert, for demos and the golden test."""
        return self.invert(self.encode(texts), num_steps)


_DEFAULT: Inverter | None = None


def get_inverter(**kwargs) -> Inverter:
    """Return a process-wide cached Inverter so callers don't reload the ~2GB models.

    A new configuration (any kwargs) replaces the cached instance.
    """
    global _DEFAULT
    if _DEFAULT is None or kwargs:
        _DEFAULT = Inverter(**kwargs)
    return _DEFAULT
