import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = ["XOR", "Circle", "Spiral"]


def make_data(name, n, noise, seed=42):
    rng = np.random.default_rng(seed)
    if name == "XOR":
        base = rng.integers(0, 2, size=(n, 2)).astype(float)
        X = base + rng.normal(0, noise * 0.35, size=base.shape)
        y = (base[:, 0] != base[:, 1]).astype(int)
        return X, y
    if name == "Circle":
        n1 = n // 2
        n2 = n - n1
        a1 = rng.uniform(0, 2*np.pi, n1)
        a2 = rng.uniform(0, 2*np.pi, n2)
        r1 = rng.normal(0.8, 0.08 + noise*0.25, n1)
        r2 = rng.normal(1.8, 0.10 + noise*0.30, n2)
        X = np.vstack([
            np.c_[r1*np.cos(a1), r1*np.sin(a1)],
            np.c_[r2*np.cos(a2), r2*np.sin(a2)]
        ])
        y = np.r_[np.zeros(n1, dtype=int), np.ones(n2, dtype=int)]
        return X, y
    if name == "Spiral":
        n1 = n // 2
        n2 = n - n1
        t1 = np.linspace(0, 3.5*np.pi, n1) + rng.normal(0, noise*0.25, n1)
        t2 = np.linspace(0, 3.5*np.pi, n2) + rng.normal(0, noise*0.25, n2)
        r1 = np.linspace(0.15, 1.8, n1)
        r2 = np.linspace(0.15, 1.8, n2)
        X1 = np.c_[r1*np.cos(t1), r1*np.sin(t1)] + rng.normal(0, noise*0.35, (n1,2))
        X2 = np.c_[r2*np.cos(t2+np.pi), r2*np.sin(t2+np.pi)] + rng.normal(0, noise*0.35, (n2,2))
        return np.vstack([X1,X2]), np.r_[np.zeros(n1,dtype=int), np.ones(n2,dtype=int)]
    raise ValueError(name)


def prepare(X, y, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def make_grid(X):
    pad = 0.7
    x0, x1 = X[:,0].min()-pad, X[:,0].max()+pad
    y0, y1 = X[:,1].min()-pad, X[:,1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x0,x1,180), np.linspace(y0,y1,180))
    return np.c_[xx.ravel(), yy.ravel()], xx, yy


def plot_data(Xtr,ytr,Xte,yte,title):
    fig, ax = plt.subplots(figsize=(6.2,4.8))
    ax.scatter(Xtr[ytr==0,0], Xtr[ytr==0,1], s=24, label='Train class 0')
    ax.scatter(Xtr[ytr==1,0], Xtr[ytr==1,1], s=24, label='Train class 1')
    ax.scatter(Xte[:,0], Xte[:,1], s=45, facecolors='none', edgecolors='black', label='Test')
    ax.set_title(title); ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    return fig


def plot_boundary(predict_fn, grid, Xtr,ytr,Xte,yte,title,point=None):
    G, xx, yy = grid
    z = predict_fn(G).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.2,4.8))
    ax.contourf(xx, yy, z, levels=30, alpha=0.35)
    ax.contour(xx, yy, z, levels=[0.5], linewidths=2)
    ax.scatter(Xtr[ytr==0,0], Xtr[ytr==0,1], s=22, label='Train 0')
    ax.scatter(Xtr[ytr==1,0], Xtr[ytr==1,1], s=22, label='Train 1')
    ax.scatter(Xte[:,0], Xte[:,1], s=45, facecolors='none', edgecolors='black', label='Test')
    if point is not None:
        ax.scatter([point[0]],[point[1]],s=140,marker='*',edgecolors='black',linewidths=1.5,label='Your point')
    ax.set_title(title); ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.legend(fontsize=8); ax.grid(alpha=.15)
    return fig


def plot_curves(hist):
    fig, ax = plt.subplots(figsize=(6.2,4.8))
    ax.plot(hist['loss'], label='Train loss')
    ax.plot(hist['val_loss'], label='Test loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Binary cross-entropy'); ax.set_title('Learning curve'); ax.legend(); ax.grid(alpha=.2)
    return fig


def plot_network(sizes, weights):
    fig, ax = plt.subplots(figsize=(8,4.8)); ax.axis('off')
    max_nodes = max(sizes)
    positions=[]
    for li,n in enumerate(sizes):
        ys=np.linspace(0.15,0.85,n) if n>1 else np.array([0.5])
        positions.append([(li/(len(sizes)-1), y) for y in ys])
    for li in range(len(sizes)-1):
        W=weights[li]
        scale=max(np.abs(W).max(),1e-8)
        for i,(x1,y1) in enumerate(positions[li]):
            for j,(x2,y2) in enumerate(positions[li+1]):
                ax.plot([x1,x2],[y1,y2],alpha=.45,lw=.5+2.2*abs(W[i,j])/scale)
    for li,layer in enumerate(positions):
        for x,y in layer:
            ax.scatter([x],[y],s=420,edgecolors='black',zorder=3)
    ax.set_xlim(-.08,1.08); ax.set_ylim(.05,.95)
    ax.set_title('Learned network weights'); return fig


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.8,4.2))
    im=ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); return fig
