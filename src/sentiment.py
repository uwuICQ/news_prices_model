"""Тональность: P(positive) − P(negative), модель seara/rubert-tiny2-russian-sentiment."""

from __future__ import annotations

SENTIMENT_MODEL = "seara/rubert-tiny2-russian-sentiment"


class SentimentScorer:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        self.model.to("cpu")
        self.model.eval()
        labels = {int(idx): str(name).lower() for idx, name in self.model.config.id2label.items()}
        try:
            self.pos_idx = next(idx for idx, name in labels.items() if "pos" in name)
            self.neg_idx = next(idx for idx, name in labels.items() if "neg" in name)
        except StopIteration as exc:
            raise RuntimeError(f"У модели нет меток positive/negative: {labels}") from exc

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        if not texts:
            return []
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            chunk = [text[:2000] for text in texts[start: start + batch_size]]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend(float(row[self.pos_idx] - row[self.neg_idx]) for row in probs)
        return scores


def score_pending(conn, scorer: SentimentScorer | None = None, batch_size: int = 32) -> int:
    """Считает тональность релевантных постов, у которых её ещё нет."""
    rows = conn.execute(
        """
        SELECT channel, msg_id, text
        FROM news
        WHERE relevant = 1 AND sentiment IS NULL
        ORDER BY published_at
        """
    ).fetchall()
    if not rows:
        return 0
    scorer = scorer or SentimentScorer()
    done = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start: start + batch_size]
        values = scorer.score([row["text"] for row in chunk])
        conn.executemany(
            "UPDATE news SET sentiment = ? WHERE channel = ? AND msg_id = ?",
            [(score, row["channel"], row["msg_id"]) for score, row in zip(values, chunk)],
        )
        conn.commit()
        done += len(chunk)
        print(f"  тональность: {done}/{len(rows)}", flush=True)
    return done
