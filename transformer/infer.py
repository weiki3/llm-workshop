from train import Transformer, TransformerConfig, DataLoaderLite, BLOCK_SIZE, enc, BOS_IDX, EOS_IDX
from pathlib import Path
import torch
from torch.nn import functional as F

BATCH_SIZE = 1
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / '20260610_1446.model'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

infer_loader = DataLoaderLite(BATCH_SIZE, BLOCK_SIZE)
model = Transformer(TransformerConfig())
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()
model.to(device)

max_length = BLOCK_SIZE

src_idx, _, _ = infer_loader.next_batch()
idx = torch.tensor([BOS_IDX], dtype=torch.long)
idx = idx.unsqueeze(0).repeat(BATCH_SIZE, 1)

src_idx = src_idx.to(device)
idx = idx.to(device)

while idx.size(1) < max_length and not (idx[:, -1] == EOS_IDX).all():
    with torch.no_grad():
        logits, _ = model(src_idx, idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, -1)
        topk_probs, topk_indices = torch.topk(probs, 20, -1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        idx = torch.cat((idx, xcol), dim=1)

for seq in idx.tolist():
    print(enc.decode(seq))
