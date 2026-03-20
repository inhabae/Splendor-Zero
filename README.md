# Splendor AI Engine

An AlphaZero-style AI for [Splendor](https://boardgamegeek.com/boardgame/148228/splendor) — built with PyTorch and trained entirely from self-play, with a custom C++ game engine, a React analysis UI, and a browser-automation bridge for playing live on [Spendee](https://spendee.mattle.online).

<!-- INSERT: Hero screenshot of the web UI in Analysis mode -->

---

## What's in here

**C++ game engine** — Full Splendor rules (69-action space, 252-dim state vector) with pybind11 bindings for Python.

**Neural network** — A masked policy-value net (PyTorch) that takes the encoded board state and outputs move probabilities + a win estimate. Trained end-to-end from self-play with no human data.

**MCTS / IS-MCTS** — Two search modes implemented natively in C++. Standard MCTS re-determinizes hidden information per simulation. Information Set MCTS maintains a shared tree indexed by observable state, used for live play where the opponent's hand is unknown.

**Self-play training loop** — AlphaZero-style cycle: generate games → train → evaluate → promote champion. Supports parallel workers, rolling replay buffers, and automatic champion promotion.

**Web UI** — React + FastAPI app for playing against the engine, setting up positions manually, and running continuous IS-MCTS analysis on any position.

<img width="1297" height="745" alt="Screenshot 2026-03-20 at 2 20 01 PM" src="https://github.com/user-attachments/assets/6510249e-7d30-434b-ae10-ad610c792654" />


**Spendee bridge** — A Playwright automation layer that reads Spendee's Meteor reactive state, tracks hidden information across turns, and submits moves via `clientAction()`.

<!-- INSERT: Screenshot or GIF of the bridge playing a live game -->

---

## Tech stack

| | |
|---|---|
| Game engine | C++17, pybind11, CMake |
| Neural network | PyTorch |
| Search | Native C++ MCTS / IS-MCTS |
| Web backend | FastAPI |
| Web frontend | React 18, TypeScript, Vite |
| Live play | Playwright |

---

## Getting started

**Install dependencies:**
```bash
pip install -r requirements-nn.txt          # training + web UI
pip install -r requirements-webui.txt       # FastAPI server
pip install -r requirements-spendee.txt     # live bridge (optional)
```

**Build the native extension:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target splendor_native -j$(nproc)
```

**Smoke test:**
```bash
python -m nn.train --mode smoke --episodes 5 --mcts-sims 32
```

**Run a training loop:**
```bash
python -m nn.train --mode cycles \
  --cycles 500 --episodes-per-cycle 5000 \
  --mcts-sims 200 --model-res-blocks 5 \
  --save-checkpoint-every-cycles 5 --auto-promote \
  --collector-workers 4
```

**Launch the web UI:**
```bash
cd webui && npm install && npm run build && cd ..
python -m uvicorn nn.webapp:app --port 8000
```
Drop `.pt` checkpoint files in `nn_artifacts/checkpoints/` and they appear in the UI automatically.

**Run the Spendee bridge:**
```bash
python -m spendee.cli \
  --checkpoint nn_artifacts/checkpoints/champion.pt \
  --user-data-dir ~/.config/splendor-chromium \
  --search-type ismcts --num-simulations 5000 \
  --live
```

---

## Results
![win rate graph](https://github.com/user-attachments/assets/16a287cc-9865-4fc6-85da-9a03589a8de2)
| Opponent | Win rate |
|----------|----------|
| Random | 100% |
| Greedy heuristic | 100% |
| Spendee leaderboard | **2068 rating — attaining the highest title of Grandmaster** |
