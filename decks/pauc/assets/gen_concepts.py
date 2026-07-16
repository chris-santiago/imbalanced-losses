"""Conceptual diagrams for the pAUC teaching deck (no existing figure covers these).

Emits two SVGs:
  concept_roc_band.svg   — ROC curve with the shaded FPR band [alpha, beta] and the
                           normalized partial-AUC region the loss targets.
  concept_hnm_axis.svg   — score axis showing the band as a label-free hard-negative
                           miner: decoys inside [t_beta, t_alpha] get gradient (pushed
                           down), decoys that escape above t_alpha do not.

These are schematic (illustrate mechanism), not data plots, so they carry no
provenance caption. Run: uv run --python 3.12 python gen_concepts.py
"""

BLUE = "#1a4a7a"
ORANGE = "#d97706"
GREEN = "#15803d"
RED = "#b91c1c"
GRAY = "#6b7280"
LGRAY = "#d1d5db"
FILL_BLUE = "#eaf1f8"
FILL_GREEN = "#e4f0e6"
BAND = "#f4d35e"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="Helvetica,Arial,sans-serif">'
        ]

    def text(self, x, y, s, size=17, fill="#222", anchor="middle",
             weight="normal", style="", mono=False):
        fam = ' font-family="SFMono-Regular,Menlo,monospace"' if mono else ""
        st = f' font-style="{style}"' if style else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{st}{fam}>{esc(s)}</text>'
        )

    def rect(self, x, y, w, h, fill="#fff", stroke=GRAY, sw=1.5, rx=5, opacity=1.0):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'fill-opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=GRAY, w=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>'
        )

    def path(self, d, stroke=BLUE, w=3, fill="none"):
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>'
        )

    def circle(self, x, y, r, fill=GRAY, stroke="none", sw=1.5):
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def arrow(self, x1, y1, x2, y2, stroke=GRAY, w=2.5, head=9):
        self.line(x1, y1, x2, y2, stroke=stroke, w=w)
        dx, dy = x2 - x1, y2 - y1
        n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        self.parts.append(
            f'<polygon points="{x2:.1f},{y2:.1f} '
            f'{x2 - head * ux + head * 0.55 * px:.1f},{y2 - head * uy + head * 0.55 * py:.1f} '
            f'{x2 - head * ux - head * 0.55 * px:.1f},{y2 - head * uy - head * 0.55 * py:.1f}" '
            f'fill="{stroke}"/>'
        )

    def save(self, path):
        self.parts.append("</svg>")
        with open(path, "w") as f:
            f.write("\n".join(self.parts))
        print("wrote", path)


# ---------------------------------------------------------------- ROC band
def roc_band():
    W, H = 1000, 560
    s = SVG(W, H)
    ax_l, ax_r, ax_t, ax_b = 130, 900, 70, 470
    pw, ph = ax_r - ax_l, ax_b - ax_t

    s.text(W / 2, 38, "The loss targets a BAND of the ROC, not the whole curve",
           25, BLUE, weight="bold")

    # FPR band [alpha, beta] as fractions of the FPR axis
    alpha, beta = 0.0, 0.30  # schematic; alpha=0 recommended default
    bx0 = ax_l + alpha * pw
    bx1 = ax_l + beta * pw

    # a concave ROC curve y = x^k mapped into the axes (k<1 -> concave, good ranker)
    def rocx(fpr):
        return ax_l + fpr * pw

    def rocy(tpr):
        return ax_b - tpr * ph

    def tpr_of(fpr):
        return fpr ** 0.32  # concave

    # shaded partial-AUC region: under the curve, within [alpha, beta]
    steps = 60
    region = [f"M {rocx(alpha):.1f} {ax_b:.1f}"]
    for i in range(steps + 1):
        f = alpha + (beta - alpha) * i / steps
        region.append(f"L {rocx(f):.1f} {rocy(tpr_of(f)):.1f}")
    region.append(f"L {rocx(beta):.1f} {ax_b:.1f} Z")
    s.parts.append(
        f'<path d="{" ".join(region)}" fill="{GREEN}" fill-opacity="0.28" stroke="none"/>'
    )

    # band strip (full height, light)
    s.rect(bx0, ax_t, bx1 - bx0, ph, fill=BAND, stroke="none", rx=0, opacity=0.20)

    # axes
    s.line(ax_l, ax_t, ax_l, ax_b, stroke="#333", w=2)
    s.line(ax_l, ax_b, ax_r, ax_b, stroke="#333", w=2)
    # diagonal chance line
    s.line(ax_l, ax_b, ax_r, ax_t, stroke=LGRAY, w=1.5, dash="5,5")

    # ROC curve
    d = [f"M {rocx(0):.1f} {rocy(0):.1f}"]
    for i in range(1, 101):
        f = i / 100
        d.append(f"L {rocx(f):.1f} {rocy(tpr_of(f)):.1f}")
    s.path(" ".join(d), stroke=BLUE, w=3.5)

    # band edge lines + labels
    for x, lab in [(bx0, "α = 0"), (bx1, "β = budget")]:
        s.line(x, ax_t, x, ax_b, stroke=RED, w=1.8, dash="4,3")
        s.text(x, ax_t - 10, lab, 18, RED, weight="bold")

    # axis labels
    s.text(ax_l + pw / 2, ax_b + 42, "False-positive rate (FPR)", 20, "#333")
    s.parts.append(
        f'<text x="{ax_l - 52}" y="{ax_t + ph / 2}" font-size="20" fill="#333" '
        f'text-anchor="middle" transform="rotate(-90 {ax_l - 52} {ax_t + ph / 2})">'
        f'True-positive rate (recall)</text>'
    )

    # callouts
    s.text(rocx((alpha + beta) / 2), rocy(tpr_of(beta / 2)) + 70,
           "partial AUC", 19, GREEN, weight="bold")
    s.text(rocx((alpha + beta) / 2), rocy(tpr_of(beta / 2)) + 92,
           "over the band", 19, GREEN, weight="bold")
    s.text(ax_r - 150, ax_t + 40, "whole-curve AUC", 17, GRAY, style="italic")
    s.text(ax_r - 150, ax_t + 62, "(what CE / AP chase)", 15, GRAY, style="italic")

    s.text(W / 2, H - 18,
           "normalized pAUC over [α, β]  =  P( s_pos > s_neg | negative in the band )",
           18, "#333", mono=True)
    s.save("assets/concept_roc_band.svg")


# ---------------------------------------------------- score axis / HNM miner
def hnm_axis():
    W, H = 1000, 470
    s = SVG(W, H)
    ax_l, ax_r = 90, 910
    axis_y = 250

    s.text(W / 2, 40, "The band is a label-free hard-negative miner",
           25, BLUE, weight="bold")
    s.text(W / 2, 70, "pairwise surrogate pushes the in-band negatives (decoys) below the positives",
           17, GRAY)

    # band region on the score axis [t_beta, t_alpha]
    tb = ax_l + 0.58 * (ax_r - ax_l)
    ta = ax_l + 0.86 * (ax_r - ax_l)
    s.rect(tb, axis_y - 95, ta - tb, 190, fill=BAND, stroke="none", rx=0, opacity=0.35)
    for x, lab in [(tb, "t_β"), (ta, "t_α = max(neg)")]:
        s.line(x, axis_y - 100, x, axis_y + 95, stroke=RED, w=1.8, dash="4,3")
        s.text(x, axis_y - 108, lab, 17, RED, weight="bold", mono=True)

    # the axis
    s.arrow(ax_l, axis_y, ax_r + 8, axis_y, stroke="#333", w=2.5)
    s.text(ax_r + 4, axis_y + 26, "higher score →", 16, "#333", anchor="end")

    # positives (orange, above axis) clustered high
    import random
    rnd = random.Random(7)
    for _ in range(9):
        x = ax_l + (0.70 + 0.26 * rnd.random()) * (ax_r - ax_l)
        y = axis_y - 40 - 22 * rnd.random()
        s.circle(x, y, 6, fill=ORANGE)
    s.text(ax_l + 0.84 * (ax_r - ax_l), axis_y - 92 + 190, "", 1)  # spacer noop

    # bulk negatives (blue) low, inert
    for _ in range(16):
        x = ax_l + (0.02 + 0.5 * rnd.random()) * (ax_r - ax_l)
        y = axis_y + 34 + 30 * rnd.random()
        s.circle(x, y, 5.5, fill=BLUE, stroke="none")

    # in-band decoys (red, ringed) — get gradient, pushed to LOWER score (leftward)
    for _ in range(6):
        x = tb + (ta - tb) * rnd.random()
        y = axis_y + 40 + 24 * rnd.random()
        s.circle(x, y, 6, fill=RED, stroke="#7a0f1c", sw=2)
        s.arrow(x - 6, y, x - 46, y, stroke=RED, w=2, head=7)

    # escaped decoys above t_alpha (red, no ring) — no gradient
    for _ in range(2):
        x = ta + (0.02 + 0.09 * rnd.random()) * (ax_r - ax_l)
        y = axis_y + 34 + 20 * rnd.random()
        s.circle(x, y, 6, fill=RED, stroke="none")

    # legends / callouts
    s.text((ax_l + tb) / 2, axis_y + 116, "bulk negatives", 16, BLUE)
    s.text((ax_l + tb) / 2, axis_y + 136, "no gradient (inert)", 14, GRAY, style="italic")
    s.text((tb + ta) / 2, axis_y + 116, "in-band decoys", 16, RED, weight="bold")
    s.text((tb + ta) / 2, axis_y + 136, "← pushed down-rank (mined)", 14, RED)
    s.text(ta + 55, axis_y + 116, "escapees", 15, RED)
    s.text(ta + 55, axis_y + 136, "(α=0 fixes this)", 13, GRAY, style="italic")
    s.text(ax_l + 0.82 * (ax_r - ax_l), axis_y - 70, "positives", 16, ORANGE, weight="bold")

    s.save("assets/concept_hnm_axis.svg")


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + 2.718281828459045 ** (-z))


def tau_scale():
    """Two panels: at late training (inflated score scale), a fixed raw tau
    turns the sigmoid into a near-step (thin gradient zone, saturated), while
    tau_eff proportional to scale keeps the transition zone tracking the spread."""
    W, H = 1040, 430
    s = SVG(W, H)
    s.text(W / 2, 34, "Late in training the score scale inflates — the sigmoid's width must track it",
           23, BLUE, weight="bold")

    def panel(px, title, sub, tau_frac, ok):
        pw = 430
        x0, x1 = px + 40, px + pw - 20
        ybot, ytop = 330, 120
        t = (x0 + x1) / 2          # threshold (band edge), centered
        D = x1 - x0
        tau = tau_frac * D

        # panel heading
        s.text(px + pw / 2, 66, title, 19, "#222", weight="bold")
        s.text(px + pw / 2, 88, sub, 15, GREEN if ok else RED)

        # inflated negative-score spread (same in both panels): shaded bell near axis
        sd = 0.20 * D
        pts = [f"M {x0:.1f} {ybot:.1f}"]
        n = 60
        for k in range(n + 1):
            x = x0 + D * k / n
            g = 2.718281828459045 ** (-0.5 * ((x - t) / sd) ** 2)
            pts.append(f"L {x:.1f} {ybot - 46 * g:.1f}")
        pts.append(f"L {x1:.1f} {ybot:.1f} Z")
        s.parts.append(
            f'<path d="{" ".join(pts)}" fill="{BLUE}" fill-opacity="0.14" stroke="none"/>'
        )

        # gradient zone: where |s-t|/tau <= 2  (sigmoid slope non-negligible)
        gx0, gx1 = t - 2 * tau, t + 2 * tau
        s.rect(gx0, ytop, gx1 - gx0, ybot - ytop, fill=(FILL_GREEN if ok else "#fdecef"),
               stroke="none", rx=0)

        # axes
        s.line(x0, ybot, x1, ybot, stroke="#333", w=2)
        s.line(t, ytop - 6, t, ybot + 6, stroke=RED, w=1.6, dash="4,3")
        s.text(t, ybot + 20, "t (band edge)", 13, RED)
        s.text(t, ybot + 40, "negative scores (inflated spread)", 12, GRAY)

        # sigmoid curve sigma((x - t)/tau)
        d = []
        for k in range(0, 121):
            x = x0 + D * k / 120
            y = _sigmoid((x - t) / tau)
            yy = ybot - y * (ybot - ytop)
            d.append(("M" if k == 0 else "L") + f" {x:.1f} {yy:.1f}")
        s.path(" ".join(d), stroke=(GREEN if ok else RED), w=3)
        s.text(x0 - 6, ytop + 4, "σ=1", 12, GRAY, anchor="end")
        s.text(x0 - 6, ybot, "σ=0", 12, GRAY, anchor="end")

        # gradient-zone label
        lab = "gradient zone" + (" — tracks the spread" if ok else " — razor-thin")
        s.text(t, ytop - 12, lab, 14, (GREEN if ok else RED), weight="bold")
        if not ok:
            s.text(gx1 + 8, (ytop + ybot) / 2 - 10, "saturated", 13, RED, anchor="start")
            s.text(gx1 + 8, (ytop + ybot) / 2 + 8, "σ'≈0", 13, RED, anchor="start", mono=True)

    panel(10, "Fixed raw τ", "sigmoid → step: band gradient vanishes", 0.045, ok=False)
    panel(540, "Scale-aware  τ_eff = temperature × scale",
          "constant sharpness in FPR units", 0.16, ok=True)

    s.save("assets/concept_tau_scale.svg")


if __name__ == "__main__":
    roc_band()
    hnm_axis()
    tau_scale()
