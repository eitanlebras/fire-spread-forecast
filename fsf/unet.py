import torch, torch.nn as nn


def block(i, o, p):
    # GroupNorm (no running stats): trainable under torch.func.vmap across seeds, and batch-size independent.
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(), nn.Dropout2d(p))


class UNet(nn.Module):
    """depth down/up blocks with skips. 64x64 in, 64x64 out logits."""
    def __init__(self, in_ch, width=32, depth=3, dropout=0.1, out_ch=1):
        super().__init__()
        ws = [width * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList(); c = in_ch
        for w in ws[:-1]: self.enc.append(block(c, w, dropout)); c = w
        self.bott = block(c, ws[-1], dropout)
        self.up = nn.ModuleList(); self.dec = nn.ModuleList(); c = ws[-1]
        for w in reversed(ws[:-1]):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2)); self.dec.append(block(2 * w, w, dropout)); c = w
        self.head = nn.Conv2d(c, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for e in self.enc: x = e(x); skips.append(x); x = self.pool(x)
        x = self.bott(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)): x = d(torch.cat([u(x), s], 1))
        return self.head(x)
