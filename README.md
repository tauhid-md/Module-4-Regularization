# Module 4: Regularized Cost and Gradient

Linear aur logistic regression ke cost function aur gradient me **L2 regularization** add kiya gaya hai.

## Files

| File | Description |
|------|-------------|
| `Module_4_Regularization_SOLVED.ipynb` | Solved notebook, code aur outputs ek hi jagah |
| `regularization.py` | Same solution ka standalone Python script |

## Setup

```bash
pip install numpy jupyter
```

```bash
jupyter notebook Module_4_Regularization_SOLVED.ipynb
# ya
python regularization.py
```

## What was completed

Regularization term: $\frac{\lambda}{2m}\sum_{j=0}^{n-1} w_j^2$. Parameter `b` regularize nahi hota (standard practice).

| Function | Regularization part |
|----------|---------------------|
| `compute_cost_linear_reg` | `reg_cost = (lambda_ / (2*m)) * sum(w[j]**2)` |
| `compute_cost_logistic_reg` | `hx = sigmoid(w·x + b)`, cost `-y·log(hx) - (1-y)·log(1-hx)`, phir wahi `reg_cost` |
| `compute_gradient_linear_reg` | `dj_dw[j] += (lambda_ / m) * w[j]` |
| `compute_gradient_logistic_reg` | `dj_dw[j] += (lambda_ / m) * w[j]` |

Linear aur logistic ka fark sirf `h(x)` me hai: linear me `w·x + b`, logistic me `sigmoid(w·x + b)`. Regularization ka tarika dono me same hai.

## Results (notebook ke Expected Output se match kiya)

| Check | Output |
|-------|--------|
| Linear regularized cost | 0.07917239320214275 |
| Logistic regularized cost | 0.6850849138741673 |
| Linear `dj_db` | 0.6648774569425726 |
| Linear `dj_dw` | [0.29653214748822276, 0.4911679625918033, 0.21645877535865857] |
| Logistic `dj_db` | 0.341798994972791 |
| Logistic `dj_dw` | [0.17380012933994293, 0.32007507881566943, 0.10776313396851499] |

## Notes
- `lambda_` badhane par weights chhote hote hain aur overfitting kam hoti hai. `lambda_ = 0` par regularization band ho jata hai.
- `compute_gradient_*` functions `(dj_db, dj_dw)` is order me return karte hain.
