const { VectorIndex } = require("./index.js");
const DIM = 384, N = 100_000;

const all = new Float32Array(N * DIM);

for (let i = 0; i < N; i++) {
  let s = 0;

  for (let j = 0; j < DIM; j++) {
    const v = Math.random() * 2 - 1;
    all[i * DIM + j] = v;
    s += v * v;
  }

  s = Math.sqrt(s);
  for (let j = 0; j < DIM; j++) {
    all[i * DIM + j] /= s;
  }
}

const q = all.slice(42*DIM, 43*DIM);

const idx = new VectorIndex(DIM);
idx.addBatch(all);

function jsSearch(data, q, k) {
  const sc = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    let d = 0; const o = i * DIM;
    
    for (let j = 0; j < DIM; j++) {
      d += data[o+j] * q[j];
    }

    sc[i] = d;
  }

  return Array.from({
    length: N
  }, (_, i) => i)
    .sort((a, b) => sc[b] - sc[a])
    .slice(0, k);
}

const run = (name, f) => {
  for (let i = 0; i < 5; i++) f();
  const t = process.hrtime.bigint();
  const r = f();

  console.log(
    name.padEnd(12),
    (Number(process.hrtime.bigint() - t) / 1e6).toFixed(2) + " ms"
  );

  return r;
};

const rust = run("rust", () => idx.search(q, 5));
const js = run("javascript", () => jsSearch(all, q, 5));

console.log("same:", JSON.stringify(rust.map(h => h.index)) === JSON.stringify(js));
console.log(rust);