from pathlib import Path

from loralens.training.loop import Train

if __name__ == "__main__":
    train = Train(
        model_name="gpt2",
        data_source="pile",  # set to "pile" if your env supports zstd
        # text_paths=[Path("data.txt")],

        max_seq_len=1024,
        stride=1024,
        drop_remainder=True,

        lens_type="tuned",
        loss="kl",
        token_shift=None,          # uses tuned-lens default (KL->0, CE->1)

        per_gpu_batch_size=20,      # per process/GPU
        num_steps=1024,
        lr=1e-3,

        amp=True,
        amp_dtype="bf16",          # A100: use bf16

        log_every=1,
        log_mem_every=1,
        save_every=512,

        output=Path("lens.pt"),
        ddp=True,
    )

    train.execute()