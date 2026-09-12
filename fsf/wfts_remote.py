"""Pull individual fire directories out of the 48 GB Zenodo zip via HTTP range requests (no full download).
  python -m fsf.wfts_remote list                      -> fires per year
  python -m fsf.wfts_remote fetch OUT_DIR YEAR [YEAR] [--shard i/k] -> all fires of those years not already in OUT_DIR
"""
import sys, os, io, zipfile, requests

URL = "https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content"


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges with a chunk cache. zipfile reads member-by-member."""
    def __init__(self, url, chunk=32 << 20):
        self.url, self.chunk, self.pos, self.s = url, chunk, 0, requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60); self.url = r.url
        self.size = int(r.headers["Content-Length"]); self.cache = {}
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def tell(self): return self.pos
    def readable(self): return True
    def seekable(self): return True
    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 8: self.cache.pop(next(iter(self.cache)))
            a, b = i * self.chunk, min((i + 1) * self.chunk, self.size) - 1
            for attempt in range(6):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120); r.raise_for_status(); self.cache[i] = r.content; break
                except Exception as e:
                    if attempt == 5: raise
        return self.cache[i]
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.chunk); blk = self._block(i); take = blk[o:o + n]
            out += take; self.pos += len(take); n -= len(take)
        return bytes(out)


def open_zip():
    return zipfile.ZipFile(RangeFile(URL))


def fires_by_year(z):
    fires = {}
    for n in z.namelist():
        p = n.split("/")
        if len(p) >= 3 and p[-1].endswith(".tif") and p[-3].isdigit(): fires.setdefault(p[-3], {}).setdefault(p[-2], []).append(n)
    return fires


if __name__ == "__main__":
    z = open_zip(); fires = fires_by_year(z)
    if sys.argv[1] == "list":
        for y in sorted(fires): print(y, len(fires[y]), "fires,", sum(len(v) for v in fires[y].values()), "days")
    elif sys.argv[1] == "fetch":
        out = sys.argv[2]; n_done = 0; args = sys.argv[3:]; shard = (0, 1)
        if "--shard" in args: i = args.index("--shard"); shard = tuple(int(v) for v in args[i + 1].split("/")); args = args[:i]
        todo = [(y, f, ms) for y in args for f, ms in sorted(fires[y].items()) if not os.path.isdir(f"{out}/{y}/{f}") and len(ms) >= 3]
        todo = todo[shard[0]::shard[1]]
        print(f"fetching {len(todo)} fires ({sum(len(ms) for _,_,ms in todo)} days)", flush=True)
        for y, f, members in todo:
            tmp = f"{out}/{y}/{f}.partial"; os.makedirs(tmp, exist_ok=True)
            for m in sorted(members):
                with z.open(m) as src, open(f"{tmp}/{os.path.basename(m)}", "wb") as dst: dst.write(src.read())
            os.rename(tmp, f"{out}/{y}/{f}"); n_done += 1
            if n_done % 10 == 0: print(f"  {n_done}/{len(todo)} fires done", flush=True)
        print("FETCH_DONE", n_done, flush=True)
