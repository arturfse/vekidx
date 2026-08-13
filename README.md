# vekidx

Vector similarity search for Node, written in Rust

* Exact search
* No runtime dependencies
* One file on disk

## Install

```bash
npm install vekidx
```

## Use

```js
const { VectorIndex } = require("vekidx");

const index = new VectorIndex(384);
// Float32Array, all vectors end to end
index.addBatch(vectors);

// [ { index: 42, score: 0.91 }, ... ]
const hits = index.search(query, 5);
// same, off the main thread
await index.searchAsync(query, 5);

index.save("./my.vidx");
const back = VectorIndex.load("./my.vidx");
```

**Vectors must be normalized to length 1.**

Most embedding models give normalized vectors already.

## Speed

100k vectors × 384 dims, Apple M-series:

| Action | Time |
|---|---|
| search | 12 ms |
| the same algorithm in JavaScript | 57 ms |
| save (146 MB) | 74 ms |
| load | 27 ms |

Measured with `bench.js` in this repository.

## Limits

- **Brute force** Reads every vector on each search (Good to ~1 million)
- **One process** Not shared between Node worker threads
- **`search` blocks the JavaScript thread** Use `searchAsync` in a server

## Build from source

```bash
npm install
npm run build
npm run check
```

