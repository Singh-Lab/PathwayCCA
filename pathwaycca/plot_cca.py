import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import compute_canonical_loadings

# Similar to R's plotting but with coloring with respect to gene labels
def plt_var(load_X, load_Y,
            d1=1, d2=2,
            remove_last_xcols=0,
            Xnames=None, Ynames=None,
            gene_labels=None,
            radius=1.0,
            title="Variable Correlation Plot"):

    LX = pd.DataFrame(load_X)
    LX.columns = [f"Dim{i+1}" for i in range(LX.shape[1])] 
    LY = pd.DataFrame(load_Y)
    LY.columns = [f"Dim{i+1}" for i in range(LY.shape[1])]
    
    if remove_last_xcols > 0:
        LX = LX.iloc[:-remove_last_xcols, :]
        if Xnames is not None:
            Xnames = Xnames[:-remove_last_xcols]

    plt.figure(figsize=(7.5, 7))
    theta = np.linspace(0, 2*np.pi, 500)
    plt.plot(np.cos(theta)*radius, np.sin(theta)*radius, color="black", lw=1.3)
    plt.plot(0.5*np.cos(theta), 0.5*np.sin(theta),
             color="gray", lw=1.0, ls="--", alpha=0.8)

    plt.axvline(0, color="gray", ls="--", lw=0.8)
    plt.axhline(0, color="gray", ls="--", lw=0.8)

    if gene_labels is not None:
        df = LY.copy()
        df["label"] = gene_labels
        pretty = sns.color_palette("Set2", len(np.unique(gene_labels)))
        sns.scatterplot(
            data=df, 
            x=f"Dim{d1}", y=f"Dim{d2}",
            hue="label", palette=pretty,
            s=42, linewidth=0.3
        )
    else:
        plt.scatter(LY[f"Dim{d1}"], LY[f"Dim{d2}"],
                    s=40, c="blue", alpha=0.8)

    plt.scatter(LX[f"Dim{d1}"], LX[f"Dim{d2}"],
                s=65, c="#D62728", 
                linewidths=0.8, alpha=0.95)

    if Xnames is not None:
        offset = 0.04
        for name, (x, y) in zip(Xnames, LX[[f"Dim{d1}", f"Dim{d2}"]].values):

            dx = offset if x >= 0 else -offset
            dy = offset if y >= 0 else -offset

            plt.text(x + dx, y + dy, name,
                     color="red", fontsize=13,
                     fontweight="bold", ha="center", va="center")

    if Ynames is not None:
        for name, (x, y) in zip(Ynames, LY[[f"Dim{d1}", f"Dim{d2}"]].values):
            plt.text(x, y, name, color="blue", fontsize=8)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(f"Dimension {d1}", fontsize=13)
    plt.ylabel(f"Dimension {d2}", fontsize=13)
    plt.title(title, fontsize=15, weight="bold")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    # Place legend outside bottom
    if gene_labels is not None:
        plt.legend(
            title="Gene Type",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=11,
            title_fontsize=12,
            ncol=2
        )

    plt.tight_layout()
    return plt.gca()


def plt_indiv(U, V, d1=1, d2=2,
              plot="Z", labels=None,
              title="Individual Plot"):

    x = scores[:, d1-1]
    y = scores[:, d2-1]
    n = len(x)

    if labels is None:
        labels = np.arange(1, n + 1)

    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, c="gray", s=45, alpha=0.7)

    for i in range(n):
        plt.text(x[i], y[i], str(labels[i]), fontsize=7)

    plt.axvline(0, color="gray", ls="--")
    plt.axhline(0, color="gray", ls="--")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel(f"Dimension {d1}")
    plt.ylabel(f"Dimension {d2}")
    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    return plt.gca()

def plt_cca(U, V, X_input, Y_input,
            d1=1, d2=2,
            mode="v",          # "v", "i", "b"
            var_plot="Z",
            indiv_plot="Z",
            remove_last_xcols=0,
            Xnames=None, Ynames=None,
            gene_labels=None,
            labels=None,title_v=None,title_i=None):
    
    if var_plot == "Z":
        Z = U+V
    elif var_plot == "U":
        Z = U
    else:
        Z = V
    load_X = compute_canonical_loadings(X_input,Z)
    load_Y = compute_canonical_loadings(Y_input,Z)

    if mode == "v":
        return plt_var(load_X, load_Y,
                       d1, d2, 
                       remove_last_xcols=remove_last_xcols,
                       Xnames=Xnames, Ynames=Ynames,
                       gene_labels=gene_labels,title=title_v)

    elif mode == "i":
        return plt_indiv(U, V, d1, d2,
                         plot=indiv_plot,
                         labels=labels,title=title_i)

    elif mode == "b":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        plt.sca(axes[0])
        plt_var(load_X, load_Y,
                d1, d2, 
                remove_last_xcols=remove_last_xcols,
                Xnames=Xnames, Ynames=Ynames,
                gene_labels=gene_labels)

        plt.sca(axes[1])
        plt_indiv(U, V, d1, d2,
                  plot=indiv_plot,
                  labels=labels)

        plt.tight_layout()
        return fig

    else:
        raise ValueError("mode must be 'v', 'i', or 'b'")
