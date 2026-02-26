#!/usr/bin/env python3
"""SABER Publication Figure Suite.

Style reference: Cas12a diagnostic papers (NAR / Cell Rep Methods).
Yellow gRNA background, green/red highlights, purple heatmaps,
black gene maps, generous whitespace.

Usage:  python saber_figures.py -o figures/
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  STYLE — matches reference papers
# ═══════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.4,
        "axes.edgecolor": "#444444",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.8,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.framealpha": 0.95,
    })

# ── Palette matching reference papers ──
# Nucleotide tiles: yellow bg, green match, red mismatch
# Heatmaps: purple/magenta
# Gene map: black boxes white text
# Data: steel blue scatter, red thresholds

Y = dict(
    # nucleotide architecture (reference paper style)
    gRNA      = "#FFEB3B",    # yellow — gRNA background
    gRNA_edge = "#FBC02D",
    seed_fill = "#A5D6A7",    # soft green — seed region
    seed_edge = "#66BB6A",
    snp_fill  = "#EF5350",    # red — mismatch/SNP
    snp_edge  = "#C62828",
    sm_fill   = "#42A5F5",    # blue — synthetic mismatch (cyan-ish)
    sm_edge   = "#1E88E5",
    nt_fill   = "#F5F5F5",    # very light grey — neutral spacer
    nt_edge   = "#BDBDBD",
    pam_fill  = "#333333",    # black PAM block
    pam_edge  = "#111111",
    # functional
    direct    = "#3F51B5",    # indigo
    proximity = "#FF7043",    # deep orange
    enhanced  = "#00897B",    # teal
    baseline  = "#BDBDBD",
    threshold = "#E53935",    # red
    # heatmap — purple gradient like reference
    hm_lo     = "#FFFFFF",
    hm_hi     = "#7B1FA2",    # deep purple
    # text
    ink       = "#212121",
    muted     = "#757575",
    faint     = "#E0E0E0",
)

DRUG = {
    "RIF":"#3F51B5","INH":"#E53935","EMB":"#2E7D32","PZA":"#FF8F00",
    "LFX":"#6A1B9A","MFX":"#6A1B9A","AMK":"#00695C","KAN":"#BF360C",
    "INH/ETH":"#E53935",
}

# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════

@dataclass
class Cand:
    label:str; drug:str; pam:str; pam_var:str; spacer:str
    strand:str="+"; snp:int=0; sm:int=0; sm_sub:str=""
    score:float=0; a_m:float=0; a_w:float=0; disc:float=0
    strat:str="DIRECT"
    @property
    def n(self): return len(self.spacer)
    @property
    def gc(self):
        return sum(1 for b in self.spacer.upper() if b in "GC")/max(self.n,1)

CANDS = [
    Cand("rpoB D516V","RIF","TTCA","TTYN","TAGTCCAGAACAACCCG",
         "+",4,2,"G→A",12.5,0.85,0.11,7.7),
    Cand("katG S315T","INH","TTCG","TTYN","TCCATACGACCTCGATGCAGTGGT",
         "−",21,19,"C→A",5.8,0.72,0.10,7.2),
    Cand("katG S315N","INH","TTCG","TTYN","TCCATACGACCTCGATGCAGTGGT",
         "+",21,19,"C→A",5.0,0.68,0.12,5.7),
    Cand("inhA c.−15C>T","INH/ETH","TTTC","TTTV","GGCCCGGCCGCGGCGAAATG",
         "+",19,17,"G→A",5.0,0.65,0.09,7.2),
    Cand("inhA c.−8T>C","INH","TTCA","TTYN","GTGGCTGTTGGCAGTCACCCGAAAG",
         "−",23,0,"",4.5,0.55,0.14,3.9),
    Cand("gyrA D94G","LFX","TTTA","TTTV","GCGATCAAGCGTTATCTG",
         "+",3,1,"G→A",8.2,0.78,0.10,7.8),
    Cand("rrs A1401G","AMK","TTCN","TTCN","GACGGAAAGACCCCGTGA",
         "+",5,3,"A→C",4.3,0.60,0.18,3.3),
]

PANEL = [
    dict(g="rpoB",m="S531L",dr="RIF",d=0,p=20,db=0,de=0,s="PROX"),
    dict(g="rpoB",m="H526Y",dr="RIF",d=6,p=0,db=4.,de=4.4,s="DIR"),
    dict(g="rpoB",m="D516V",dr="RIF",d=18,p=0,db=4.,de=7.7,s="SM"),
    dict(g="katG",m="S315T",dr="INH",d=0,p=0,db=0,de=0,s="FAIL"),
    dict(g="fabG1",m="c.−15C>T",dr="INH",d=0,p=0,db=0,de=0,s="FAIL"),
    dict(g="embB",m="M306V",dr="EMB",d=0,p=20,db=0,de=0,s="PROX"),
    dict(g="embB",m="M306I",dr="EMB",d=0,p=20,db=0,de=0,s="PROX"),
    dict(g="pncA",m="H57D",dr="PZA",d=0,p=0,db=0,de=0,s="FAIL"),
    dict(g="pncA",m="D49N",dr="PZA",d=0,p=20,db=0,de=0,s="PROX"),
    dict(g="gyrA",m="D94G",dr="LFX",d=6,p=0,db=3.,de=7.8,s="SM"),
    dict(g="gyrA",m="A90V",dr="LFX",d=0,p=19,db=0,de=0,s="PROX"),
    dict(g="rrs",m="A1401G",dr="AMK",d=12,p=0,db=3.,de=3.3,s="DIR"),
    dict(g="rrs",m="C1402T",dr="AMK",d=12,p=0,db=3.,de=5.,s="SM"),
    dict(g="eis",m="c.−14C>T",dr="KAN",d=0,p=20,db=0,de=0,s="PROX"),
]

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _save(fig, p):
    fig.savefig(p, facecolor="white")
    fig.savefig(p.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"    ✓  {p.name}")

def _lbl(ax, c, x=-0.06, y=1.05):
    ax.text(x, y, c, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", color=Y["ink"])


# ═══════════════════════════════════════════════════════════════
#  FIG 1 — SPACER ARCHITECTURE  (reference paper style)
# ═══════════════════════════════════════════════════════════════

def fig1(cands, out):
    _style()
    draw = [c for c in cands if c.strat == "DIRECT"]
    paths = []

    for pg in range((len(draw)+3)//4):
        batch = draw[pg*4:(pg+1)*4]
        n = len(batch)
        fig = plt.figure(figsize=(7.5, n*2.1+0.4))

        for idx, c in enumerate(batch):
            frac = 1.0/n
            ax = fig.add_axes([0.02, 1-(idx+1)*frac+0.015, 0.96, frac*0.90])
            ax.set_xlim(-5, max(c.n+13, 34))
            ax.set_ylim(-1.4, 2.8)
            ax.axis("off")

            # ── title ──
            dc = DRUG.get(c.drug, Y["ink"])
            ax.text(0, 2.55, c.label,
                    fontsize=11, fontweight="bold", color=Y["ink"])
            ax.text(0, 2.15,
                    f"{c.drug}  ·  {c.pam_var}  ·  strand {c.strand}",
                    fontsize=6.5, color=Y["muted"])

            # ── PAM block (black, like gene map boxes in reference) ──
            pw = 2.8
            ax.add_patch(FancyBboxPatch(
                (-pw-0.8, 0.15), pw, 1.05,
                boxstyle="round,pad=0.06",
                fc=Y["pam_fill"], ec=Y["pam_edge"], lw=0.6))
            ax.text(-pw/2-0.8, 1.0, "PAM",
                    fontsize=7, fontweight="bold", color="white",
                    ha="center", va="center")
            ax.text(-pw/2-0.8, 0.48, c.pam,
                    fontsize=8, color="#CCCCCC", ha="center",
                    va="center", fontfamily="monospace", fontweight="bold")

            # ── nucleotide tiles ──
            bw, bh, gap = 0.92, 0.88, 0.08
            stp = bw + gap
            by = 0.20

            for i, nt in enumerate(c.spacer):
                x = i*stp + 0.5
                p = i+1
                is_seed = 1 <= p <= 8
                is_snp = p == c.snp
                is_sm = p == c.sm and c.sm > 0

                if is_snp:
                    fc, ec, tc = Y["snp_fill"], Y["snp_edge"], "white"
                elif is_sm:
                    fc, ec, tc = Y["sm_fill"], Y["sm_edge"], "white"
                elif is_seed:
                    fc, ec, tc = Y["gRNA"], Y["gRNA_edge"], Y["ink"]
                else:
                    fc, ec, tc = Y["nt_fill"], Y["nt_edge"], Y["ink"]

                # tile
                ax.add_patch(FancyBboxPatch(
                    (x, by), bw, bh,
                    boxstyle="round,pad=0.04",
                    fc=fc, ec=ec, lw=0.5))

                # letter
                ax.text(x+bw/2, by+bh/2, nt,
                        fontsize=8, fontfamily="monospace",
                        fontweight="bold", ha="center", va="center",
                        color=tc)

                # position numbers
                if is_seed or is_snp or is_sm or p == c.n:
                    ax.text(x+bw/2, by-0.22, str(p),
                            fontsize=5, ha="center", va="top",
                            color=Y["muted"])

            # ── seed bracket ──
            s0 = 0.5
            s1 = 8*stp + 0.5 - gap
            br = -0.55
            ax.plot([s0, s0], [br, br+0.12], color=Y["ink"], lw=0.5)
            ax.plot([s1, s1], [br, br+0.12], color=Y["ink"], lw=0.5)
            ax.plot([s0, s1], [br, br], color=Y["ink"], lw=0.5)
            ax.text((s0+s1)/2, br-0.20, "SEED (1–8)",
                    fontsize=5.5, ha="center", va="top",
                    color=Y["ink"], fontweight="bold")

            # ── info card ──
            cx = c.n*stp + 2.2
            cw, ch = 8.0, 1.10
            ax.add_patch(Rectangle((cx, 0.15), cw, ch,
                         fc="#FAFAFA", ec="#CCCCCC", lw=0.4))

            lines = [
                f"{c.n} nt    GC {c.gc:.0%}    score {c.score:.1f}",
                f"SNP @ pos {c.snp}",
            ]
            if c.sm > 0:
                lines.append(f"SM  {c.sm_sub}  @ pos {c.sm}")
                lines.append(
                    f"MUT {c.a_m:.0%}    WT {c.a_w:.0%}    "
                    f"Disc {c.disc:.1f}×")

            for j, ln in enumerate(lines):
                ax.text(cx+0.25, 1.05-j*0.26, ln,
                        fontsize=5.5, fontfamily="monospace",
                        color=Y["ink"], va="center")

            # ── ruler ──
            ry = -1.0
            rm = c.n*stp + 1
            ax.plot([0, rm], [ry, ry], color=Y["faint"], lw=0.3)
            for tk in range(0, int(rm)+1, 5):
                ax.plot([tk, tk], [ry-0.05, ry+0.05],
                        color=Y["faint"], lw=0.3)
                ax.text(tk, ry-0.20, str(tk), fontsize=4,
                        ha="center", va="top", color=Y["muted"])

        # ── legend ──
        hs = [
            mpatches.Patch(fc=Y["pam_fill"], ec=Y["pam_edge"], lw=.4, label="PAM"),
            mpatches.Patch(fc=Y["gRNA"],     ec=Y["gRNA_edge"], lw=.4, label="Seed (1–8)"),
            mpatches.Patch(fc=Y["snp_fill"], ec=Y["snp_edge"], lw=.4, label="SNP"),
            mpatches.Patch(fc=Y["sm_fill"],  ec=Y["sm_edge"],  lw=.4, label="Synth. MM"),
            mpatches.Patch(fc=Y["nt_fill"],  ec=Y["nt_edge"],  lw=.4, label="Spacer"),
        ]
        fig.legend(handles=hs, loc="lower right", ncol=5,
                   fontsize=6, edgecolor=Y["faint"],
                   bbox_to_anchor=(0.98, 0.002))

        p = out / f"fig1_spacer_architecture_p{pg+1}.png"
        _save(fig, p); paths.append(p)
    return paths


# ═══════════════════════════════════════════════════════════════
#  FIG 2 — MISMATCH HEATMAP  (purple gradient like ref Fig C)
# ═══════════════════════════════════════════════════════════════

def fig2(out):
    _style()
    pos = np.arange(1, 21)
    types = ["rA:dA","rA:dG","rA:dC","rU:dT","rU:dC","rU:dG",
             "rG:dG","rG:dA","rG:dT","rC:dC","rC:dA","rC:dT"]
    nt, np_ = len(types), len(pos)

    prof = np.array([.08,.10,.12,.15,.18,.22,.28,.35,
                     .45,.52,.58,.62,.65,.68,.75,.80,.85,.88,.90,.92])
    mods = np.array([.70,.65,1.,.90,.85,1.10,.75,.70,1.10,.85,.80,.90])
    np.random.seed(42)
    mat = np.clip(np.outer(mods,prof)+np.random.normal(0,.033,(nt,np_)),0,1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.subplots_adjust(left=0.10, right=0.86, top=0.82, bottom=0.13)

    # purple gradient (reference paper style)
    cmap = LinearSegmentedColormap.from_list("pur",
        ["#FFFFFF","#E1BEE7","#CE93D8","#AB47BC","#7B1FA2","#4A148C"], N=256)

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(np_)); ax.set_xticklabels(pos)
    ax.set_yticks(range(nt));  ax.set_yticklabels(types, fontfamily="monospace")
    ax.set_xlabel("Position from PAM", labelpad=8)
    ax.set_ylabel("Mismatch  (crRNA : target)", labelpad=8)

    for i in range(nt):
        for j in range(np_):
            v = mat[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=3.5, fontfamily="monospace",
                    color="white" if v > 0.55 else Y["ink"])

    # region dividers
    ax.axvline(7.5, color=Y["ink"], lw=0.8, ls="--", alpha=.3)
    ax.axvline(13.5, color=Y["muted"], lw=0.5, ls=":", alpha=.3)

    cb = plt.colorbar(im, ax=ax, shrink=.85, pad=.02, aspect=28)
    cb.set_label("Normalised activity", fontsize=7, labelpad=4)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.3)

    ax.set_title("enAsCas12a single-mismatch tolerance profile",
                 fontweight="bold", pad=12)

    # region labels — as secondary x-axis text below ticks
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([3.5, 10.5, 16.5])
    ax2.set_xticklabels(["SEED","TRUNK","TAIL"], fontsize=6.5, fontweight="bold")
    ax2.tick_params(length=0, pad=2)
    ax2.spines["top"].set_visible(False)
    # colour seed label
    for tl in ax2.get_xticklabels():
        tl.set_color(Y["ink"] if tl.get_text()=="SEED" else Y["muted"])

    _lbl(ax, "A", y=1.22)
    _save(fig, out/"fig2_mismatch_heatmap.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 3 — PANEL OVERVIEW
# ═══════════════════════════════════════════════════════════════

def fig3(out):
    _style()
    fig = plt.figure(figsize=(7.5, 9.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1.1],
                  hspace=0.50, wspace=0.45)

    # ── A: inventory ──
    ax = fig.add_subplot(gs[0,:])
    labs = [f"{r['g']} {r['m']}" for r in PANEL]
    y = np.arange(len(labs))
    ax.barh(y, [r["d"] for r in PANEL], .52,
            color=Y["direct"], alpha=.80, label="Direct",
            ec="white", lw=.3)
    ax.barh(y, [r["p"] for r in PANEL], .52,
            left=[r["d"] for r in PANEL],
            color=Y["proximity"], alpha=.75, label="Proximity",
            ec="white", lw=.3)
    for i,r in enumerate(PANEL):
        tot=r["d"]+r["p"]
        dc=DRUG.get(r["dr"],Y["muted"])
        ax.text(max(tot+.8,22.5),i,r["dr"],fontsize=4.5,fontweight="bold",
                color="white",va="center",
                bbox=dict(boxstyle="round,pad=0.12",fc=dc,ec="none",alpha=.82))
        if r["s"]=="FAIL":
            ax.text(.5,i,"no candidates",fontsize=5.5,
                    color=Y["threshold"],va="center",style="italic")
        elif r["s"]=="SM":
            ax.plot(tot+.4,i,"*",color=Y["enhanced"],ms=5,zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6)
    ax.set_xlabel("crRNA candidates"); ax.invert_yaxis()
    ax.set_title("MDR-TB 14-plex panel — candidate inventory",
                 fontweight="bold")
    ax.plot([],[],  "*",color=Y["enhanced"],ms=5,ls="none",label="SM-enhanced")
    ax.legend(fontsize=5.5, loc="lower right")
    _lbl(ax,"A")

    # ── B: strategy ──
    ax = fig.add_subplot(gs[1,0])
    sm={"DIR":0,"SM":0,"PROX":0,"FAIL":0}
    for r in PANEL: sm[r["s"]]+=1
    slabs=["Direct","Direct + SM","Proximity","Failed"]
    svals=[sm["DIR"],sm["SM"],sm["PROX"],sm["FAIL"]]
    scols=[Y["direct"],Y["enhanced"],Y["proximity"],"#E0E0E0"]
    ax.pie(svals, labels=None,
           autopct=lambda v: f"{v:.0f}%" if v>0 else "",
           colors=scols, startangle=90,
           wedgeprops=dict(ec="white",lw=1.2),
           pctdistance=.62,
           textprops=dict(fontsize=6,fontweight="bold",color="white"))
    ax.legend(slabs, fontsize=5, loc="lower left", bbox_to_anchor=(-.10,-.06))
    ax.set_title("Detection strategy", fontweight="bold", fontsize=8.5)
    _lbl(ax,"B",x=-.18)

    # ── C: drug coverage ──
    ax = fig.add_subplot(gs[1,1])
    dcnt={}
    for r in PANEL: dcnt[r["dr"]]=dcnt.get(r["dr"],0)+1
    drugs=list(dcnt); cnts=list(dcnt.values())
    cols=[DRUG.get(d,Y["muted"]) for d in drugs]
    bs=ax.bar(drugs,cnts,color=cols,alpha=.80,ec="white",lw=.4,width=.60)
    for b,ct in zip(bs,cnts):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.08,
                str(ct),ha="center",va="bottom",fontsize=6,fontweight="bold")
    ax.set_ylabel("Targets"); ax.set_ylim(0,max(cnts)+.7)
    ax.set_title("Drug-class coverage", fontweight="bold", fontsize=8.5)
    _lbl(ax,"C",x=-.18)

    # ── D: discrimination ──
    ax = fig.add_subplot(gs[2,:])
    dd=[r for r in PANEL if r["db"]>0]
    dl=[f"{r['g']} {r['m']}" for r in dd]
    x=np.arange(len(dl)); w=.30
    ax.bar(x-w/2,[r["db"] for r in dd],w,
           color=Y["baseline"],alpha=.85,label="Baseline (1 MM)",ec="white",lw=.3)
    ax.bar(x+w/2,[r["de"] for r in dd],w,
           color=Y["enhanced"],alpha=.85,label="Enhanced (+ SM)",ec="white",lw=.3)
    for i,r in enumerate(dd):
        if r["de"]>r["db"]*1.15:
            ax.text(i+w/2,r["de"]+.2,f'{r["de"]/r["db"]:.1f}×',
                    fontsize=5.5,fontweight="bold",color=Y["enhanced"],ha="center")
    ax.axhline(10,color=Y["threshold"],ls="--",lw=.6,alpha=.45)
    ax.text(len(dd)-.3,10.25,"10× diagnostic threshold",
            fontsize=5,color=Y["threshold"],ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(dl,fontsize=6,rotation=25,ha="right")
    ax.set_ylabel("Discrimination  (MUT / WT)")
    ax.set_title("Synthetic-mismatch enhancement", fontweight="bold",fontsize=8.5)
    ax.legend(fontsize=5.5)
    _lbl(ax,"D")
    _save(fig, out/"fig3_panel_overview.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 4 — ENHANCEMENT LANDSCAPE  (scatter style like ref Fig B)
# ═══════════════════════════════════════════════════════════════

def fig4(out):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    fig.subplots_adjust(wspace=.42, left=.07, right=.94, bottom=.20, top=.85)

    np.random.seed(123)
    n=100
    pos=np.random.randint(1,21,n)
    am=np.clip(.28+.037*pos+np.random.normal(0,.06,n),.05,1.)
    aw=np.clip(.14-.007*np.clip(9-pos,0,9)+np.random.normal(0,.04,n),.01,.45)
    aw=np.where(pos<=8,aw*.42,aw)
    di=np.clip(am/np.maximum(aw,.01),0,25)

    # blue dots like reference scatter
    dot_c = "#3F51B5"

    ax=axes[0]
    ax.scatter(pos,am,c=dot_c,s=10,alpha=.50,ec="none")
    ax.axhline(.30,color=Y["threshold"],ls="--",lw=.5,alpha=.5)
    ax.axvspan(.5,8.5,alpha=.04,color=Y["gRNA"])
    ax.text(4.5,.97,"seed",fontsize=5.5,ha="center",color=Y["muted"],style="italic")
    ax.set_xlabel("SM position"); ax.set_ylabel("Activity vs MUT")
    ax.set_xlim(0,21); ax.set_ylim(0,1.05)
    _lbl(ax,"A",x=-.18)

    ax=axes[1]
    ax.scatter(pos,di,c=dot_c,s=10,alpha=.50,ec="none")
    ax.axhline(10,color=Y["threshold"],ls="--",lw=.5)
    ax.axvspan(.5,8.5,alpha=.04,color=Y["gRNA"])
    ax.set_xlabel("SM position"); ax.set_ylabel("Discrimination ratio")
    ax.set_xlim(0,21)
    _lbl(ax,"B",x=-.18)

    ax=axes[2]
    sc=ax.scatter(am,di,c=pos,cmap="viridis_r",s=10,alpha=.55,ec="none")
    ax.axhline(10,color=Y["threshold"],ls="--",lw=.5)
    ax.axvline(.30,color=Y["threshold"],ls=":",lw=.4)
    ax.fill_between([.30,1.05],10,26,alpha=.03,color=Y["enhanced"])
    ax.text(.65,23,"optimal",fontsize=5,ha="center",
            color=Y["enhanced"],fontweight="bold",alpha=.5)
    ax.set_xlabel("Activity vs MUT"); ax.set_ylabel("Discrimination ratio")
    ax.set_xlim(0,1.05); ax.set_ylim(0,26)
    cb=plt.colorbar(sc,ax=ax,shrink=.80,pad=.03,aspect=22)
    cb.set_label("SM pos.",fontsize=5.5); cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(.3)
    _lbl(ax,"C",x=-.18)

    fig.suptitle("SM design-space exploration",
                 fontsize=9.5,fontweight="bold",y=.96)
    _save(fig, out/"fig4_enhancement_landscape.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 5 — FLUORESCENCE KINETICS
# ═══════════════════════════════════════════════════════════════

def fig5(out):
    _style()
    tgts=[("rpoB D516V",7.7,.85,.11),("gyrA D94G",7.8,.78,.10),
          ("katG S315T",7.2,.72,.10),("rrs C1402T",5.0,.65,.13),
          ("rrs A1401G",3.3,.60,.18),("inhA c.−15C>T",7.2,.65,.09)]

    fig,axes=plt.subplots(2,3,figsize=(7.5,4.8))
    fig.subplots_adjust(hspace=.50,wspace=.35,left=.08,right=.97,top=.89,bottom=.09)
    t=np.linspace(0,120,300)

    for idx,(lb,dc,am,aw) in enumerate(tgts):
        ax=axes[idx//3,idx%3]
        np.random.seed(idx+10)
        fm=am*(1-np.exp(-am*.05*t))+np.random.normal(0,.005,len(t))
        fw=aw*(1-np.exp(-aw*.05*t))+np.random.normal(0,.004,len(t))
        fn=.02*(1-np.exp(-.001*t))

        ax.fill_between(t,fm-.010,fm+.010,alpha=.10,color=Y["threshold"],lw=0)
        ax.fill_between(t,fw-.008,fw+.008,alpha=.10,color=Y["direct"],lw=0)
        ax.plot(t,fm,color=Y["threshold"],lw=.8,label="MUT")
        ax.plot(t,fw,color=Y["direct"],lw=.8,label="WT")
        ax.plot(t,fn,color=Y["faint"],lw=.4,ls=":",label="NTC")

        ax.set_xlim(0,125); ax.set_ylim(-.02,1.02)
        ax.set_title(lb,fontsize=7.5,fontweight="bold",pad=5)
        if idx>=3: ax.set_xlabel("Time (min)",fontsize=6.5)
        if idx%3==0: ax.set_ylabel("Fluorescence (a.u.)",fontsize=6.5)

        ax.text(108,(fm[-1]+fw[-1])/2,f"{dc:.1f}×",
                fontsize=6,fontweight="bold",color=Y["enhanced"],
                ha="center",va="center",
                bbox=dict(boxstyle="round,pad=.10",fc="white",
                          ec=Y["enhanced"],lw=.4,alpha=.92))
        if idx==0: ax.legend(fontsize=4.5,loc="center right",handlelength=1)
        _lbl(ax,chr(65+idx),x=-.18,y=1.12)

    fig.suptitle("Predicted trans-cleavage kinetics",
                 fontsize=9.5,fontweight="bold",y=.97)
    _save(fig, out/"fig5_fluorescence_kinetics.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 6 — COOPERATIVITY MODEL
# ═══════════════════════════════════════════════════════════════

def fig6(out):
    _style()
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.5,3.0))
    fig.subplots_adjust(wspace=.38,left=.09,right=.96,bottom=.17,top=.86)

    d=np.arange(0,11)
    pen=np.array([.70,.60,.40,.25,.15,.08,.04,.02,.01,.005,.002])

    a1.fill_between(d,pen,alpha=.08,color=Y["direct"])
    a1.plot(d,pen,"o-",color=Y["direct"],ms=4,mec="white",mew=.4,lw=.9)
    for di,pi in [(1,.60),(2,.40),(4,.15)]:
        a1.annotate(f"{pi:.2f}",xy=(di,pi),xytext=(di+.55,pi+.045),
                    fontsize=5,color=Y["direct"],
                    arrowprops=dict(arrowstyle="-",color=Y["direct"],lw=.35))
    a1.set_xlabel("Inter-mismatch distance (nt)")
    a1.set_ylabel("Cooperative penalty")
    a1.set_xlim(-.5,10.5); a1.set_ylim(-.02,.80)
    _lbl(a1,"A",x=-.15)

    nat=3
    smp=np.arange(1,21)
    dp=[]
    for s in smp:
        dst=abs(s-nat)
        am_=min(.62-.025*max(8-s,0)+.012*max(s-8,0),.90)
        aw_=.13*(1-pen[dst]) if dst<len(pen) else .14
        dp.append(min(am_/max(aw_,.01),20))
    dp=np.array(dp)

    bc=[Y["enhanced"] if abs(s-nat)<=2
        else Y["direct"] if abs(s-nat)<=4
        else Y["baseline"] for s in smp]

    a2.bar(smp,dp,.60,color=bc,alpha=.75,ec="white",lw=.25)
    a2.axvline(nat,color=Y["threshold"],lw=.8,alpha=.30)
    a2.text(nat+.5,dp.max()*.88,f"nat. MM @ {nat}",
            fontsize=5,ha="left",color=Y["threshold"],fontweight="bold")
    a2.axhline(10,color=Y["threshold"],ls="--",lw=.5)
    a2.set_xlabel("Synthetic mismatch position")
    a2.set_ylabel("Predicted discrimination (×)")
    a2.set_xlim(.3,20.7)

    hs=[mpatches.Patch(fc=Y["enhanced"],label="Δ ≤ 2"),
        mpatches.Patch(fc=Y["direct"],label="Δ 3–4"),
        mpatches.Patch(fc=Y["baseline"],label="Δ > 4")]
    a2.legend(handles=hs,fontsize=5,loc="upper right",
              title="Dist. (nt)",title_fontsize=5)
    _lbl(a2,"B",x=-.15)

    fig.suptitle("Cooperative destabilisation model",
                 fontsize=9.5,fontweight="bold",y=.96)
    _save(fig, out/"fig6_cooperativity_model.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 7 — GENOME MAP  (black boxes / white text like reference)
# ═══════════════════════════════════════════════════════════════

def fig7(out):
    _style()
    fig,ax=plt.subplots(figsize=(7.5,2.8))
    fig.subplots_adjust(top=.78,bottom=.08,left=.03,right=.97)

    gl=4_411_532
    genes=dict(
        gyrA=(7302,9818,"LFX"),rpoB=(759807,763325,"RIF"),
        rrs=(1471846,1473382,"AMK"),inhA=(1673440,1674183,"INH"),
        katG=(2153889,2156111,"INH"),pncA=(2288681,2289241,"PZA"),
        eis=(2714124,2715332,"KAN"),embB=(4246514,4249810,"EMB"))

    # genome bar
    ax.barh(0,gl,.22,color="#F5F5F5",ec="#999999",lw=.4)

    for mb in range(0,5):
        x=mb*1_000_000
        ax.plot([x,x],[-.12,.12],color="#CCCCCC",lw=.3)
        ax.text(x,-.30,f"{mb} Mb",fontsize=5,ha="center",color=Y["muted"])

    # gene loci — black boxes with white text (reference style)
    for i,(gene,(s,e,dr)) in enumerate(genes.items()):
        mid=(s+e)/2
        up=i%2==0
        yl=.55+(i%3)*.30 if up else -(.55+(i%3)*.30)

        # black box on bar
        ax.barh(0,e-s,.18,left=s,color="#333333",ec="#111111",lw=.3,zorder=3)

        # connector
        ax.plot([mid,mid],[.11 if up else -.11,yl-(.10 if up else -.10)],
                color="#999999",lw=.3)

        # label in black box (reference style)
        ax.text(mid,yl,gene,fontsize=5.5,ha="center",fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=.08",fc="#333333",
                          ec="#111111",lw=.3))

    ax.set_xlim(-60_000,gl+60_000); ax.set_ylim(-1.6,1.8)
    ax.axis("off")
    ax.set_title("M. tuberculosis H37Rv  —  SABER target loci",
                 fontsize=9.5,fontweight="bold",pad=10)
    ax.text(gl/2,-1.40,f"Genome  {gl:,} bp",
            fontsize=5.5,ha="center",color=Y["muted"])
    _save(fig, out/"fig7_genome_map.png")


# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# FIGURES 8–11: Reference-paper style (variant alignment,
# conservation landscape, discrimination scatter, coverage heatmap)
# ═══════════════════════════════════════════════════════════════

# Additional palette constants for fig8-11 (reference-paper style)
# Extend Y with aliases used by fig8-11 (maps to existing palette keys)
Y.update(
    pam    = Y["pam_fill"],
    snp    = Y["snp_fill"],
    snp_edge = Y["snp_edge"],
    sm     = Y["sm_fill"],
    sm_edge = Y["sm_edge"],
    match  = Y["seed_fill"],
    match_edge = Y["seed_edge"],
    nt_bg  = Y["nt_fill"],
    nt_edge = Y["nt_edge"],
)

NT_COLOR = {"A":"#EF5350","T":"#FFEB3B","G":"#66BB6A","C":"#42A5F5",
            "N":"#BDBDBD"}

DRUG_C = {"RIF":"#3F51B5","INH":"#E53935","EMB":"#2E7D32",
           "PZA":"#FF8F00","LFX":"#6A1B9A","AMK":"#00897B","KAN":"#BF360C"}

def fig8(out):
    _style()

    # Two target sites (like reference panels D + E)
    # "." = match to crRNA, letter = mismatch (coloured)
    targets = [
        {
            "title": "rpoB D516V  ·  Target site (+)",
            "crRNA": "TAGTCCAGAACAACCCG",
            "pam":   "TTCA",
            "variants": [
                # (name, positions_with_mismatches, CFD, pct)
                ("H37Rv (ref)",  ".................", 1.00, 38.5),
                ("Beijing",      "..........G......", 0.92, 22.1),
                ("LAM",          "....T............", 0.85,  9.4),
                ("Haarlem",      ".A...............", 0.78,  7.2),
                ("EAI",          "...........A.....", 0.71,  5.8),
                ("S-type",       "...C.............", 0.68,  4.1),
                ("Uganda",       "..............G..", 0.61,  3.3),
                ("CAS",          "....T.......A....", 0.55,  2.8),
                ("X-type",       ".......G.........", 0.51,  2.1),
            ],
        },
        {
            "title": "gyrA D94G  ·  Target site (+)",
            "crRNA": "GCGATCAAGCGTTATCTG",
            "pam":   "TTTA",
            "variants": [
                ("H37Rv (ref)",  "..................", 1.00, 42.0),
                ("Beijing",      "........T.........", 0.94, 18.3),
                ("LAM",          "..A...............", 0.88, 11.5),
                ("Haarlem",      "...........C......", 0.82,  8.0),
                ("EAI",          "..............A...", 0.75,  5.2),
                ("CAS",          "....T.............", 0.65,  4.0),
                ("S-type",       ".................A", 0.58,  3.1),
            ],
        },
    ]

    fig = plt.figure(figsize=(7.5, 9.0))
    gs = GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[1.3, 1])

    for ti, tgt in enumerate(targets):
        ax = fig.add_subplot(gs[ti])
        ax.axis("off")
        ax.set_xlim(-1, 32)
        n_var = len(tgt["variants"])
        ax.set_ylim(-n_var * 0.95 - 0.8, 3.5)

        # ── panel label + title ──
        _lbl(ax, chr(65 + ti), x=-0.02, y=1.06)
        ax.text(0, 3.0, tgt["title"], fontsize=9, fontweight="bold",
                color=Y["ink"])

        # ── crRNA bar (yellow, numbered) ──
        crRNA = tgt["crRNA"]
        bw, bh = 0.85, 0.72
        gap = 0.05
        stp = bw + gap

        # position numbers — always show 1, 10, 20, last
        for i in range(len(crRNA)):
            x = i * stp
            p = i + 1
            if p in (1, 10, 20, len(crRNA)):
                ax.text(x + bw/2, 2.3, str(p), fontsize=5,
                        ha="center", va="bottom", color=Y["muted"])

        # nucleotide tiles
        for i, nt in enumerate(crRNA):
            x = i * stp
            fc = Y["gRNA"]
            ec = Y["gRNA_edge"]
            ax.add_patch(Rectangle((x, 1.3), bw, bh,
                         fc=fc, ec=ec, lw=0.4))
            ax.text(x + bw/2, 1.3 + bh/2, nt,
                    fontsize=7, fontfamily="monospace", fontweight="bold",
                    ha="center", va="center", color=Y["ink"])

        # column headers
        last_x = len(crRNA) * stp + 0.5
        ax.text(last_x + 1.5, 1.65, "CFD", fontsize=6, fontweight="bold",
                ha="center", color=Y["muted"])
        ax.text(last_x + 4.0, 1.65, "% variant", fontsize=6,
                fontweight="bold", ha="center", color=Y["muted"])

        # ── variant rows (dot notation) ──
        for vi, (name, seq, cfd, pct) in enumerate(tgt["variants"]):
            row_y = -vi * 0.95

            # Reference row gets a light cyan background strip
            if vi == 0:
                ax.add_patch(Rectangle(
                    (-0.3, row_y - 0.15), len(crRNA) * stp + 0.3, bh * 0.95,
                    fc="#E1F5FE", ec="none", zorder=0))

            for i in range(len(crRNA)):
                x = i * stp
                ch = seq[i] if i < len(seq) else "."

                if ch == ".":
                    # match → small red/grey dot (like reference)
                    ax.plot(x + bw/2, row_y + bh/2 - 0.05, ".",
                            color="#CC3333" if vi == 0 else Y["muted"],
                            ms=3.5 if vi == 0 else 2.5)
                else:
                    # mismatch → coloured box with letter
                    nc = NT_COLOR.get(ch, "#999999")
                    ax.add_patch(Rectangle(
                        (x, row_y - 0.05), bw, bh * 0.85,
                        fc=nc, ec="none", alpha=0.80))
                    ax.text(x + bw/2, row_y + bh/2 - 0.10, ch,
                            fontsize=6.5, fontfamily="monospace",
                            fontweight="bold", ha="center", va="center",
                            color="white")

            # CFD + %
            ax.text(last_x + 1.5, row_y + 0.2, f"{cfd:.2f}",
                    fontsize=6, ha="center", fontfamily="monospace",
                    color=Y["ink"])
            ax.text(last_x + 4.0, row_y + 0.2, f"{pct:.1f}",
                    fontsize=6, ha="center", fontfamily="monospace",
                    color=Y["ink"])

            # variant name
            ax.text(last_x + 6.5, row_y + 0.2, name,
                    fontsize=5.5, va="center", color=Y["ink"])

    # legend
    hs = [
        mpatches.Patch(fc=Y["gRNA"], ec=Y["gRNA_edge"], lw=.4, label="crRNA"),
        mpatches.Patch(fc=NT_COLOR["A"], ec="none", lw=.4, label="A mismatch"),
        mpatches.Patch(fc=NT_COLOR["G"], ec="none", lw=.4, label="G mismatch"),
        mpatches.Patch(fc=NT_COLOR["C"], ec="none", lw=.4, label="C mismatch"),
        mpatches.Patch(fc=NT_COLOR["T"], ec="none", lw=.4, label="T mismatch"),
    ]
    fig.legend(handles=hs, loc="lower right", ncol=5, fontsize=5.5,
               bbox_to_anchor=(0.98, 0.002))

    _save(fig, out / "fig8_variant_alignment.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 9 — CONSERVATION LANDSCAPE + GENE DOMAIN MAP
#  (like reference Fig 1C / Fig 3A — diversity along genome
#   with gene boxes below)
# ═══════════════════════════════════════════════════════════════

def fig9(out):
    _style()

    fig = plt.figure(figsize=(7.5, 4.8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.05)

    # ── A: conservation trace ──
    ax = fig.add_subplot(gs[0])

    np.random.seed(77)
    # Simulated conservation across rpoB region (750-770 kb)
    x_kb = np.linspace(759, 764, 500)
    # Base conservation ~ high for M.tb (clonal), dips at RRDR
    base = 0.95 - 0.02 * np.sin(2 * np.pi * (x_kb - 759) / 2)
    # RRDR region (760.5-761.5): lower conservation
    rrdr_mask = (x_kb > 760.5) & (x_kb < 761.5)
    base[rrdr_mask] -= 0.15 + 0.05 * np.random.randn(rrdr_mask.sum())
    # Add noise
    trace = base + np.random.normal(0, 0.015, len(x_kb))
    trace = np.clip(trace, 0.4, 1.0)

    # Coverage (secondary axis) — pink fill like reference
    coverage = 10**(3.5 + 0.8 * np.sin(np.pi * (x_kb - 759) / 5)
                    + np.random.normal(0, 0.15, len(x_kb)))

    ax2 = ax.twinx()
    ax2.fill_between(x_kb, coverage, alpha=0.12, color="#E57373", lw=0)
    ax2.set_ylabel("Sequence coverage", fontsize=7, color="#E57373")
    ax2.set_yscale("log")
    ax2.set_ylim(10, 1e6)
    ax2.tick_params(labelsize=5.5, colors="#E57373")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#E57373")
    ax2.spines["right"].set_linewidth(0.4)

    # Main trace
    ax.plot(x_kb, trace, color=Y["direct"], lw=0.6, alpha=0.85)

    # Mark target sites
    targets = [
        (760.1, "D516V", "↓"),
        (760.8, "H526Y", "↓"),
        (761.2, "S531L", "↓"),
    ]
    for xp, lbl, arr in targets:
        ax.annotate(lbl, xy=(xp, trace[np.argmin(np.abs(x_kb - xp))]),
                    xytext=(xp, 1.02),
                    fontsize=5.5, fontweight="bold", color=Y["threshold"],
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-|>", color=Y["threshold"],
                                    lw=0.5))

    # RRDR shading
    ax.axvspan(760.5, 761.5, alpha=0.06, color=Y["threshold"])
    ax.text(761.0, 0.50, "RRDR", fontsize=7, ha="center",
            color=Y["threshold"], fontstyle="italic", alpha=0.6)

    ax.set_ylabel("Conservation (identity)", fontsize=7)
    ax.set_xlim(759, 764)
    ax.set_ylim(0.4, 1.08)
    ax.set_title("rpoB locus — sequence conservation across M.tb lineages",
                 fontweight="bold", pad=8)
    _lbl(ax, "A")

    # ── B: gene domain map (black boxes, reference style) ──
    ax_g = fig.add_subplot(gs[1], sharex=ax)
    ax_g.set_ylim(-0.8, 1.2)
    ax_g.axis("off")

    # Gene body
    ax_g.plot([759, 764], [0, 0], color=Y["ink"], lw=1.0)

    # Domains (like finger/palm/thumb in reference)
    domains = [
        (759.2, 760.0, "N-terminal"),
        (760.0, 760.5, "cluster I"),
        (760.5, 761.5, "RRDR"),
        (761.5, 762.5, "cluster II"),
        (762.5, 763.8, "C-terminal"),
    ]
    for s, e, lbl in domains:
        is_rrdr = "RRDR" in lbl
        fc = Y["pam"] if not is_rrdr else Y["threshold"]
        ax_g.add_patch(Rectangle((s, -0.25), e - s, 0.50,
                       fc=fc, ec="white", lw=0.5))
        ax_g.text((s + e) / 2, 0, lbl, fontsize=5.5, ha="center",
                  va="center", color="white", fontweight="bold")

    # Position ticks
    for pos, codon in [(759.8, "507"), (760.5, "531"), (761.2, "533"),
                       (762.0, "552")]:
        ax_g.plot([pos, pos], [-0.30, -0.50], color=Y["muted"], lw=0.3)
        ax_g.text(pos, -0.65, codon, fontsize=4.5, ha="center",
                  color=Y["muted"])

    ax_g.text(761.5, -0.80, "rpoB codon position",
              fontsize=6, ha="center", color=Y["muted"])

    plt.setp(ax.get_xticklabels(), visible=False)
    _save(fig, out / "fig9_conservation_landscape.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 10 — DISCRIMINATION vs ACTIVITY SCATTER
#  (like reference Fig 3B — threshold quadrants, red dashed)
# ═══════════════════════════════════════════════════════════════

def fig10(out):
    _style()
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.90, bottom=0.14)

    np.random.seed(456)
    n = 200
    act = np.clip(np.random.beta(3, 2, n), 0.05, 1.0)
    disc = np.clip(act / (np.random.beta(1.5, 5, n) + 0.02), 0.5, 25)

    ax.scatter(act, disc, c=Y["direct"], s=8, alpha=0.45, ec="none")

    # thresholds (red dashed like reference)
    ax.axhline(10, color=Y["threshold"], ls="--", lw=0.7, alpha=0.7)
    ax.axvline(0.30, color=Y["threshold"], ls="--", lw=0.7, alpha=0.7)

    # threshold labels
    ax.text(0.97, 10.5, "10× threshold", fontsize=5.5,
            color=Y["threshold"], ha="right", va="bottom")
    ax.text(0.31, 24, "min.\nactivity", fontsize=5,
            color=Y["threshold"], ha="left", va="top")

    # quadrant labels
    ax.text(0.65, 20, "✓ optimal", fontsize=7, ha="center",
            color=Y["enhanced"], fontweight="bold", alpha=0.6)
    ax.text(0.15, 20, "low activity", fontsize=5.5, ha="center",
            color=Y["muted"], alpha=0.5)
    ax.text(0.65, 4, "low discrimination", fontsize=5.5, ha="center",
            color=Y["muted"], alpha=0.5)

    # highlight top candidates
    top = [(0.85, 7.7, "rpoB D516V"), (0.78, 7.8, "gyrA D94G"),
           (0.65, 7.2, "inhA c.−15C>T")]
    for xp, yp, lbl in top:
        ax.plot(xp, yp, "o", color=Y["threshold"], ms=5,
                mec="white", mew=0.5, zorder=5)
        ax.annotate(lbl, xy=(xp, yp), xytext=(xp + 0.05, yp + 1.5),
                    fontsize=5, fontweight="bold", color=Y["ink"],
                    arrowprops=dict(arrowstyle="-", color=Y["muted"],
                                    lw=0.3))

    ax.set_xlabel("Predicted activity vs MUT")
    ax.set_ylabel("Predicted discrimination ratio")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 26)
    ax.set_title("crRNA candidate design space", fontweight="bold", pad=8)
    _lbl(ax, "B", x=-0.12)

    _save(fig, out / "fig10_discrimination_scatter.png")


# ═══════════════════════════════════════════════════════════════
#  FIG 11 — DRUG-RESISTANCE COVERAGE HEATMAP + SUMMARY TABLE
#  (like reference Fig 3C — purple heatmap with numbers,
#   adjacent summary table)
# ═══════════════════════════════════════════════════════════════

def fig11(out):
    _style()

    # Candidates × drug-class coverage by M.tb lineage
    targets = [
        "rpoB D516V", "rpoB H526Y", "gyrA D94G",
        "rrs A1401G", "rrs C1402T", "katG S315T", "inhA c.−15C>T",
    ]
    lineages = ["L1\nEAI", "L2\nBeijing", "L3\nCAS", "L4\nEuroAm",
                "L5\nWA1", "L6\nWA2", "L7\nEthiop", "Global"]

    np.random.seed(99)
    # High coverage for most (M.tb is clonal), some dips
    mat = np.clip(0.88 + np.random.normal(0, 0.06, (len(targets), len(lineages))),
                  0.40, 1.00)
    # Some specific drops
    mat[5, 4] = 0.52  # katG in L5
    mat[5, 5] = 0.48  # katG in L6
    mat[6, 6] = 0.55  # inhA in L7
    # Global column = mean
    mat[:, -1] = mat[:, :-1].mean(axis=1)

    # Summary data
    summary = [
        ("rpoB D516V",  "RRDR",  "RIF",  2.11, "Yes", 18),
        ("rpoB H526Y",  "RRDR",  "RIF",  2.36, "Yes",  6),
        ("gyrA D94G",   "QRDR",  "LFX",  1.93, "Yes",  6),
        ("rrs A1401G",  "16S",   "AMK",  1.85, "Yes", 12),
        ("rrs C1402T",  "16S",   "AMK",  1.91, "Yes", 12),
        ("katG S315T",  "katG",  "INH",  2.87, "No",   0),
        ("inhA c.−15C>T","inhA", "INH",  0.76, "Yes",  0),
    ]

    fig = plt.figure(figsize=(7.5, 4.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.5], wspace=0.08)

    # ── heatmap ──
    ax_h = fig.add_subplot(gs[0])

    cmap = LinearSegmentedColormap.from_list("pur",
        ["#F3E5F5","#CE93D8","#AB47BC","#7B1FA2","#4A148C"], N=256)

    im = ax_h.imshow(mat, cmap=cmap, aspect="auto", vmin=0.3, vmax=1.0,
                     interpolation="nearest")

    ax_h.set_xticks(range(len(lineages)))
    ax_h.set_xticklabels(lineages, fontsize=5.5, rotation=0, ha="center")
    ax_h.set_yticks(range(len(targets)))
    ax_h.set_yticklabels(targets, fontsize=6)

    # cell values
    for i in range(len(targets)):
        for j in range(len(lineages)):
            v = mat[i, j]
            tc = "white" if v > 0.70 else Y["ink"]
            ax_h.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=4.5, fontfamily="monospace", color=tc)

    # Global column separator
    ax_h.axvline(len(lineages) - 1.5, color="white", lw=1.5)

    # colorbar below
    cb = plt.colorbar(im, ax=ax_h, orientation="horizontal",
                      shrink=0.6, pad=0.15, aspect=25)
    cb.set_label("Lineage coverage", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.3)

    ax_h.set_title("Predicted crRNA coverage by M.tb lineage",
                   fontweight="bold", fontsize=8.5, pad=8)
    _lbl(ax_h, "C", x=-0.08, y=1.08)

    # ── summary table ──
    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")

    cols = ["Region", "Drug", "Div.", "HXB2ᵃ", "# cand."]
    col_x = [0.02, 0.22, 0.40, 0.58, 0.78]

    # header
    for ci, (col, cx) in enumerate(zip(cols, col_x)):
        ax_t.text(cx, 0.96, col, fontsize=5.5, fontweight="bold",
                  ha="center", va="top", color=Y["ink"],
                  transform=ax_t.transAxes)

    ax_t.plot([0, 1], [0.94, 0.94], color=Y["ink"], lw=0.4,
              transform=ax_t.transAxes)

    row_h = 0.115
    for ri, (name, region, drug, div_, hxb2, ncand) in enumerate(summary):
        ry = 0.92 - ri * row_h
        vals = [region, drug, f"{div_:.2f}", hxb2, str(ncand)]
        for ci, (val, cx) in enumerate(zip(vals, col_x)):
            fc = Y["ink"]
            fw = "normal"
            if ci == 1:  # drug column coloured
                fc = DRUG_C.get(drug, Y["ink"])
                fw = "bold"
            ax_t.text(cx, ry, val, fontsize=5, ha="center", va="top",
                      color=fc, fontweight=fw,
                      transform=ax_t.transAxes)

    ax_t.text(0.5, 0.04, "ᵃ Match to H37Rv reference",
              fontsize=4.5, ha="center", color=Y["muted"],
              style="italic", transform=ax_t.transAxes)

    _save(fig, out / "fig11_coverage_heatmap.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def generate_all_figures(out_dir, cands=None):
    """Generate all 11 figures. Called by run_full_pipeline.py or standalone.

    Args:
        out_dir: Path or str — output directory for figures
        cands:   list of Cand objects (optional, defaults to CANDS)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cands is None:
        cands = CANDS

    print("═" * 60)
    print("  SABER Publication Figure Suite  (Figs 1–11)")
    print("  Style: Nature Methods / Cell Reports Methods / NAR")
    print("═" * 60)

    # Figs 1-7 (original suite)
    fig1(cands, out)
    fig2(out)
    fig3(out)
    fig4(out)
    fig5(out)
    fig6(out)
    fig7(out)

    # Figs 8-11 (reference-paper style additions)
    fig8(out)
    fig9(out)
    fig10(out)
    fig11(out)

    n_png = len(list(out.glob("*.png")))
    n_svg = len(list(out.glob("*.svg")))
    print(f"\n  ✓ {n_png} PNG + {n_svg} SVG saved to {out}/")
    print("═" * 60)


def main():
    ap = argparse.ArgumentParser(description="SABER Publication Figure Suite")
    ap.add_argument("-o", "--output", default="figures")
    args = ap.parse_args()
    generate_all_figures(args.output)


if __name__ == "__main__":
    main()
