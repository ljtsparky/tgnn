# Stage B — Qualitative gallery (20 AG test graphs)

Each card shows the **prompt** (same for both columns), the **ground-truth objects** the AG annotation lists in the target frame, and the two SDXL generations: **graph-conditioned** (left) uses our temporal-GNN adapter, **baseline** (right) is text-only with the same prompt and seed. Detection is GroundingDINO-tiny @ 0.30 confidence.

Sorted: graph-cond wins first, ties next, losses last.

---

## J3LY1 &nbsp; — &nbsp; 🟢 WIN &nbsp; (Δ=+0.500, graph=1.000, baseline=0.500)

- **Prompt:** `a person, with doorway, light, in an indoor scene`
- **GT objects (2):** `doorway, light`
- **Graph node count:** 40 &nbsp;&nbsp; **edges:** 55 &nbsp;&nbsp; **target_frame:** 533
- Graph-cond detected: `doorway, light` &nbsp; → missing `(none)`
- Baseline   detected: `doorway` &nbsp; → missing `light`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/J3LY1_graph_cond.png) | ![base](reportlatex/figures/stageb/J3LY1_baseline.png) |

---

## LLTBQ &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, carrying towel, with doorway, towel, in an indoor scene`
- **GT objects (4):** `clothes, door, doorway, towel`
- **Graph node count:** 48 &nbsp;&nbsp; **edges:** 71 &nbsp;&nbsp; **target_frame:** 145
- Graph-cond detected: `clothes, door, doorway, towel` &nbsp; → missing `(none)`
- Baseline   detected: `clothes, door, doorway, towel` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/LLTBQ_graph_cond.png) | ![base](reportlatex/figures/stageb/LLTBQ_baseline.png) |

---

## VP4OG &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, touching book, with book, in an indoor scene`
- **GT objects (1):** `book`
- **Graph node count:** 14 &nbsp;&nbsp; **edges:** 19 &nbsp;&nbsp; **target_frame:** 485
- Graph-cond detected: `book` &nbsp; → missing `(none)`
- Baseline   detected: `book` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/VP4OG_graph_cond.png) | ![base](reportlatex/figures/stageb/VP4OG_baseline.png) |

---

## SM8Y0 &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, sitting on chair, with chair, table, in an indoor scene`
- **GT objects (2):** `chair, table`
- **Graph node count:** 22 &nbsp;&nbsp; **edges:** 31 &nbsp;&nbsp; **target_frame:** 208
- Graph-cond detected: `chair, table` &nbsp; → missing `(none)`
- Baseline   detected: `chair, table` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/SM8Y0_graph_cond.png) | ![base](reportlatex/figures/stageb/SM8Y0_baseline.png) |

---

## WK9HE &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, holding towel, standing on floor, touching door, holding broom, with broom, door, floor, towel, window, in an indoor scene`
- **GT objects (6):** `broom, door, floor, table, towel, window`
- **Graph node count:** 150 &nbsp;&nbsp; **edges:** 244 &nbsp;&nbsp; **target_frame:** 99
- Graph-cond detected: `broom, door, floor, table, towel, window` &nbsp; → missing `(none)`
- Baseline   detected: `broom, door, floor, table, towel, window` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/WK9HE_graph_cond.png) | ![base](reportlatex/figures/stageb/WK9HE_baseline.png) |

---

## 8YD0O &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, holding sandwich, eating sandwich, holding food, eating food, with box, chair, food, laptop, sandwich, table, in an indoor scene`
- **GT objects (6):** `box, chair, food, laptop, sandwich, table`
- **Graph node count:** 242 &nbsp;&nbsp; **edges:** 409 &nbsp;&nbsp; **target_frame:** 567
- Graph-cond detected: `box, chair, food, laptop, sandwich, table` &nbsp; → missing `(none)`
- Baseline   detected: `box, chair, food, laptop, sandwich, table` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/8YD0O_graph_cond.png) | ![base](reportlatex/figures/stageb/8YD0O_baseline.png) |

---

## TRVEA &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, touching pillow, sitting on bed, with bed, pillow, in an indoor scene`
- **GT objects (2):** `bed, pillow`
- **Graph node count:** 21 &nbsp;&nbsp; **edges:** 32 &nbsp;&nbsp; **target_frame:** 12
- Graph-cond detected: `bed, pillow` &nbsp; → missing `(none)`
- Baseline   detected: `bed, pillow` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/TRVEA_graph_cond.png) | ![base](reportlatex/figures/stageb/TRVEA_baseline.png) |

---

## GH19N &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, holding book, holding clothes, with book, clothes, table, in an indoor scene`
- **GT objects (5):** `book, clothes, door, doorway, table`
- **Graph node count:** 114 &nbsp;&nbsp; **edges:** 168 &nbsp;&nbsp; **target_frame:** 268
- Graph-cond detected: `book, clothes, door, doorway, table` &nbsp; → missing `(none)`
- Baseline   detected: `book, clothes, door, doorway, table` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/GH19N_graph_cond.png) | ![base](reportlatex/figures/stageb/GH19N_baseline.png) |

---

## 1XBU2 &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, touching table, holding sandwich, eating sandwich, touching laptop, with chair, food, laptop, sandwich, table, in an indoor scene`
- **GT objects (5):** `chair, food, laptop, sandwich, table`
- **Graph node count:** 180 &nbsp;&nbsp; **edges:** 285 &nbsp;&nbsp; **target_frame:** 123
- Graph-cond detected: `chair, food, laptop, sandwich, table` &nbsp; → missing `(none)`
- Baseline   detected: `chair, food, laptop, sandwich, table` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/1XBU2_graph_cond.png) | ![base](reportlatex/figures/stageb/1XBU2_baseline.png) |

---

## X95D0 &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=1.000, baseline=1.000)

- **Prompt:** `a person, with table, in an indoor scene`
- **GT objects (1):** `table`
- **Graph node count:** 14 &nbsp;&nbsp; **edges:** 19 &nbsp;&nbsp; **target_frame:** 490
- Graph-cond detected: `table` &nbsp; → missing `(none)`
- Baseline   detected: `table` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/X95D0_graph_cond.png) | ![base](reportlatex/figures/stageb/X95D0_baseline.png) |

---

## U5T4M &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.857, baseline=0.857)

- **Prompt:** `a person, holding sandwich, holding food, with chair, food, refrigerator, sandwich, table, in an indoor scene`
- **GT objects (7):** `chair, food, groceries, refrigerator, sandwich, shelf, table`
- **Graph node count:** 271 &nbsp;&nbsp; **edges:** 432 &nbsp;&nbsp; **target_frame:** 394
- Graph-cond detected: `chair, food, refrigerator, sandwich, shelf, table` &nbsp; → missing `groceries`
- Baseline   detected: `chair, groceries, refrigerator, sandwich, shelf, table` &nbsp; → missing `food`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/U5T4M_graph_cond.png) | ![base](reportlatex/figures/stageb/U5T4M_baseline.png) |

---

## 17P5V &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.750, baseline=0.750)

- **Prompt:** `a person, holding laptop, with doorway, laptop, in an indoor scene`
- **GT objects (4):** `dish, doorway, laptop, table`
- **Graph node count:** 52 &nbsp;&nbsp; **edges:** 71 &nbsp;&nbsp; **target_frame:** 276
- Graph-cond detected: `doorway, laptop, table` &nbsp; → missing `dish`
- Baseline   detected: `doorway, laptop, table` &nbsp; → missing `dish`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/17P5V_graph_cond.png) | ![base](reportlatex/figures/stageb/17P5V_baseline.png) |

---

## J2XFQ &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.500, baseline=0.500)

- **Prompt:** `a person, with door, refrigerator, in an indoor scene`
- **GT objects (4):** `door, food, refrigerator, sandwich`
- **Graph node count:** 136 &nbsp;&nbsp; **edges:** 200 &nbsp;&nbsp; **target_frame:** 642
- Graph-cond detected: `door, refrigerator` &nbsp; → missing `food, sandwich`
- Baseline   detected: `door, refrigerator` &nbsp; → missing `food, sandwich`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/J2XFQ_graph_cond.png) | ![base](reportlatex/figures/stageb/J2XFQ_baseline.png) |

---

## OMFVL &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.500, baseline=0.500)

- **Prompt:** `a person, with doorknob, in an indoor scene`
- **GT objects (2):** `box, doorknob`
- **Graph node count:** 50 &nbsp;&nbsp; **edges:** 68 &nbsp;&nbsp; **target_frame:** 80
- Graph-cond detected: `doorknob` &nbsp; → missing `box`
- Baseline   detected: `doorknob` &nbsp; → missing `box`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/OMFVL_graph_cond.png) | ![base](reportlatex/figures/stageb/OMFVL_baseline.png) |

---

## 41A89 &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.429, baseline=0.429)

- **Prompt:** `a person, with food, shelf, in an indoor scene`
- **GT objects (7):** `door, floor, food, groceries, pillow, refrigerator, shelf`
- **Graph node count:** 262 &nbsp;&nbsp; **edges:** 382 &nbsp;&nbsp; **target_frame:** 291
- Graph-cond detected: `door, refrigerator, shelf` &nbsp; → missing `floor, food, groceries, pillow`
- Baseline   detected: `door, floor, shelf` &nbsp; → missing `food, groceries, pillow, refrigerator`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/41A89_graph_cond.png) | ![base](reportlatex/figures/stageb/41A89_baseline.png) |

---

## ZMY8M &nbsp; — &nbsp; ⚪ TIE &nbsp; (Δ=+0.000, graph=0.400, baseline=0.400)

- **Prompt:** `a person, holding food, with dish, food, table, in an indoor scene`
- **GT objects (5):** `bag, dish, food, sandwich, table`
- **Graph node count:** 209 &nbsp;&nbsp; **edges:** 342 &nbsp;&nbsp; **target_frame:** 59
- Graph-cond detected: `food, table` &nbsp; → missing `bag, dish, sandwich`
- Baseline   detected: `food, table` &nbsp; → missing `bag, dish, sandwich`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/ZMY8M_graph_cond.png) | ![base](reportlatex/figures/stageb/ZMY8M_baseline.png) |

---

## TE4PT &nbsp; — &nbsp; 🔴 LOSE &nbsp; (Δ=-0.200, graph=0.800, baseline=1.000)

- **Prompt:** `a person, holding towel, holding blanket, holding broom, with blanket, broom, doorway, towel, in an indoor scene`
- **GT objects (5):** `blanket, broom, doorway, floor, towel`
- **Graph node count:** 105 &nbsp;&nbsp; **edges:** 170 &nbsp;&nbsp; **target_frame:** 134
- Graph-cond detected: `blanket, doorway, floor, towel` &nbsp; → missing `broom`
- Baseline   detected: `blanket, broom, doorway, floor, towel` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/TE4PT_graph_cond.png) | ![base](reportlatex/figures/stageb/TE4PT_baseline.png) |

---

## Y8L60 &nbsp; — &nbsp; 🔴 LOSE &nbsp; (Δ=-0.333, graph=0.667, baseline=1.000)

- **Prompt:** `a person, with door, floor, in an indoor scene`
- **GT objects (3):** `door, doorknob, floor`
- **Graph node count:** 111 &nbsp;&nbsp; **edges:** 175 &nbsp;&nbsp; **target_frame:** 129
- Graph-cond detected: `door, floor` &nbsp; → missing `doorknob`
- Baseline   detected: `door, doorknob, floor` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/Y8L60_graph_cond.png) | ![base](reportlatex/figures/stageb/Y8L60_baseline.png) |

---

## 2Y8XQ &nbsp; — &nbsp; 🔴 LOSE &nbsp; (Δ=-0.333, graph=0.667, baseline=1.000)

- **Prompt:** `a person, holding clothes, holding towel, holding blanket, with blanket, clothes, towel, in an indoor scene`
- **GT objects (3):** `blanket, clothes, towel`
- **Graph node count:** 127 &nbsp;&nbsp; **edges:** 207 &nbsp;&nbsp; **target_frame:** 68
- Graph-cond detected: `clothes, towel` &nbsp; → missing `blanket`
- Baseline   detected: `blanket, clothes, towel` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/2Y8XQ_graph_cond.png) | ![base](reportlatex/figures/stageb/2Y8XQ_baseline.png) |

---

## 2S9LB &nbsp; — &nbsp; 🔴 LOSE &nbsp; (Δ=-0.333, graph=0.667, baseline=1.000)

- **Prompt:** `a person, sitting on chair, leaning on chair, sitting on bed, holding medicine bottle, with bed, chair, medicine bottle, in an indoor scene`
- **GT objects (3):** `bed, chair, medicine`
- **Graph node count:** 44 &nbsp;&nbsp; **edges:** 61 &nbsp;&nbsp; **target_frame:** 223
- Graph-cond detected: `bed, medicine` &nbsp; → missing `chair`
- Baseline   detected: `bed, chair, medicine` &nbsp; → missing `(none)`

| Graph-conditioned | Text-only baseline |
|---|---|
| ![graph](reportlatex/figures/stageb/2S9LB_graph_cond.png) | ![base](reportlatex/figures/stageb/2S9LB_baseline.png) |

---

