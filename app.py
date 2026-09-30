"""ANN Activity 2 — Interactive Neural Network Learning Demo
Web-based demo built with TensorFlow/Keras + Streamlit.
Run: streamlit run app.py
"""
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from nn_utils import (
    DATASETS, make_data, prepare, make_grid, plot_data, plot_boundary,
    plot_curves, plot_network, plot_confusion
)

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")

class LiveView(tf.keras.callbacks.Callback):
    def __init__(self, total, every, bar, slot_b, slot_c, grid, data):
        super().__init__()
        self.total, self.every, self.bar = total, every, bar
        self.slot_b, self.slot_c, self.grid, self.data = slot_b, slot_c, grid, data
        self.hist = {k: [] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.hist:
            self.hist[k].append(float(logs.get(k, np.nan)))
        done = epoch + 1
        self.bar.progress(
            done / self.total,
            text=f"Epoch {done}/{self.total} · loss {logs.get('loss', 0):.3f} · "
                 f"test accuracy {logs.get('val_accuracy', 0):.1%}"
        )
        if done % self.every == 0 or done == self.total:
            Xtr, ytr, Xte, yte = self.data
            fig = plot_boundary(
                lambda g: self.model(g, training=False).numpy(), self.grid,
                Xtr, ytr, Xte, yte, f"Decision boundary · epoch {done}"
            )
            self.slot_b.pyplot(fig)
            plt.close(fig)
            fig = plot_curves(self.hist)
            self.slot_c.pyplot(fig)
            plt.close(fig)


def build_model(cfg):
    tf.keras.utils.set_random_seed(cfg["seed"])
    reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    model = tf.keras.Sequential([tf.keras.Input(shape=(2,))])
    for units in cfg["hidden"]:
        model.add(tf.keras.layers.Dense(units, activation=cfg["act"], kernel_regularizer=reg))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = {
        "Adam": tf.keras.optimizers.Adam(cfg["lr"]),
        "SGD": tf.keras.optimizers.SGD(cfg["lr"], momentum=0.9),
        "RMSprop": tf.keras.optimizers.RMSprop(cfg["lr"]),
    }[cfg["opt"]]
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ------------------------------ Sidebar controls
st.sidebar.title("⚙️ Controls")
st.sidebar.header("1 · Data")
ds = st.sidebar.selectbox("Dataset", DATASETS)
n_samples = st.sidebar.slider("Samples", 100, 1000, 400, 50)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.15, 0.05)
test_size = st.sidebar.slider("Test split", 0.1, 0.5, 0.25, 0.05)

st.sidebar.header("2 · Network")
n_layers = st.sidebar.slider("Hidden layers", 1, 4, 2)
hidden = [
    st.sidebar.slider(f"Neurons in hidden layer {i + 1}", 1, 16, 8, key=f"h{i}")
    for i in range(n_layers)
]
act = st.sidebar.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu"], index=1)
l2 = st.sidebar.select_slider(
    "L2 regularisation", [0.0, 0.0001, 0.001, 0.01, 0.05], value=0.0
)

st.sidebar.header("3 · Training")
opt = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
lr = st.sidebar.select_slider("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.03)
epochs = st.sidebar.slider("Epochs", 10, 300, 100, 10)
batch = st.sidebar.select_slider("Batch size", [8, 16, 32, 64, 128], value=32)
seed = int(st.sidebar.number_input("Random seed", 0, 9999, 42))

cfg = dict(ds=ds, n=n_samples, noise=noise, test=test_size, hidden=hidden,
           act=act, l2=l2, opt=opt, lr=lr, epochs=epochs, batch=batch, seed=seed)

X, y = make_data(ds, n_samples, noise, seed)
Xtr, Xte, ytr, yte = prepare(X, y, test_size, seed)
grid = make_grid(np.vstack([Xtr, Xte]))

st.title("🧠 Interactive Neural Network Playground")
st.caption("Build, train and watch a TensorFlow/Keras neural network learn — live in your browser.")
tab_play, tab_learn, tab_about = st.tabs(["🎮 Playground", "📘 Concepts", "ℹ️ About"])

with tab_play:
    run = st.button("🚀 Train network", type="primary", use_container_width=True)
    bar_slot = st.empty()
    c1, c2 = st.columns(2)
    slot_b, slot_c = c1.empty(), c2.empty()
    details = st.container()

    if run:
        model = build_model(cfg)
        bar = bar_slot.progress(0.0, text="Starting training…")
        live = LiveView(epochs, max(1, epochs // 10), bar, slot_b, slot_c, grid, (Xtr, ytr, Xte, yte))
        t0 = time.time()
        model.fit(
            Xtr, ytr, validation_data=(Xte, yte), epochs=epochs,
            batch_size=batch, verbose=0, callbacks=[live]
        )
        lines = []
        model.summary(print_fn=lambda s, **k: lines.append(s))
        st.session_state["res"] = dict(
            cfg=dict(cfg), model=model, hist=live.hist, secs=time.time() - t0,
            summary="\n".join(lines),
            weights=[layer.get_weights()[0] for layer in model.layers
                     if isinstance(layer, tf.keras.layers.Dense)],
            data=(Xtr, ytr, Xte, yte), grid=grid
        )
        bar_slot.success(f"Training finished in {st.session_state['res']['secs']:.1f} s")

    res = st.session_state.get("res")
    if res is None:
        fig = plot_data(Xtr, ytr, Xte, yte, f"{ds} dataset · circles = train, triangles = test")
        slot_b.pyplot(fig)
        plt.close(fig)
        c2.info("👈 Choose a dataset and network in the sidebar, then press **Train network**. "
                "The decision boundary and learning curves update live.")
    else:
        model, hist = res["model"], res["hist"]
        r_Xtr, r_ytr, r_Xte, r_yte = res["data"]
        if not run:
            if res["cfg"] != cfg:
                st.warning("Sidebar settings changed since the last run — press **Train network** "
                           "to apply them. Showing the previous result.")
            fig = plot_boundary(lambda g: model(g, training=False).numpy(), res["grid"],
                                r_Xtr, r_ytr, r_Xte, r_yte, "Decision boundary · final")
            slot_b.pyplot(fig)
            plt.close(fig)
            fig = plot_curves(hist)
            slot_c.pyplot(fig)
            plt.close(fig)

        with details:
            tr_loss, tr_acc = model.evaluate(r_Xtr, r_ytr, verbose=0)
            te_loss, te_acc = model.evaluate(r_Xte, r_yte, verbose=0)
            m = st.columns(4)
            m[0].metric("Train accuracy", f"{tr_acc:.1%}")
            m[1].metric("Test accuracy", f"{te_acc:.1%}")
            m[2].metric("Test loss", f"{te_loss:.3f}")
            m[3].metric("Trainable parameters", f"{model.count_params():,}")

            if tr_acc - te_acc > 0.08:
                st.warning("Train accuracy is much higher than test accuracy → **overfitting**. "
                           "Try fewer neurons, L2 regularisation, or more data.")
            elif tr_acc < 0.8:
                st.warning("Low accuracy → **underfitting**. Try more neurons/layers, a different "
                           "activation, a higher learning rate or more epochs.")
            else:
                st.success("Good fit: train and test accuracy are both high and close together.")

            t1, t2, t3, t4 = st.tabs([
                "🕸️ Network weights", "🧮 Confusion matrix", "📄 Model summary", "🎯 Try your own point"
            ])
            with t1:
                sizes = [2] + res["cfg"]["hidden"] + [1]
                fig = plot_network(sizes, res["weights"])
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Blue = positive weight, orange = negative; thicker = larger magnitude.")
            with t2:
                pred = (model.predict(r_Xte, verbose=0).ravel() > 0.5).astype(int)
                fig = plot_confusion(confusion_matrix(r_yte.astype(int), pred, labels=[0, 1]))
                st.pyplot(fig)
                plt.close(fig)
            with t3:
                st.code(res["summary"])
            with t4:
                a, b = st.columns(2)
                px = a.slider("Feature 1 (standardised)", -3.0, 3.0, 0.0, 0.1)
                py = b.slider("Feature 2 (standardised)", -3.0, 3.0, 0.0, 0.1)
                prob = float(model(np.array([[px, py]], dtype="float32"), training=False).numpy()[0, 0])
                st.metric("Predicted class", f"{int(prob > 0.5)}", f"P(class 1) = {prob:.1%}")
                fig = plot_boundary(lambda g: model(g, training=False).numpy(), res["grid"],
                                    r_Xtr, r_ytr, r_Xte, r_yte, "Where does your point fall?",
                                    point=(px, py))
                st.pyplot(fig)
                plt.close(fig)

with tab_learn:
    st.markdown("""
### How this demo works
1. **Data** – a 2-D binary classification problem. Features are standardised using the training set.
2. **Model** – `Input(2) → Dense(hidden, activation) × N → Dense(1, sigmoid)`.
3. **Training** – the optimizer minimises **binary cross-entropy** using mini-batch gradient descent.
4. **Evaluation** – accuracy and loss on unseen test data show how the network generalises.

### Key ideas
| Concept | What it does | What to try |
|---|---|---|
| Hidden layers / neurons | More capacity → more complex boundaries | Use Spiral with 1 vs 3 layers |
| Activation function | Adds non-linearity | Compare ReLU, tanh and sigmoid |
| Learning rate | Controls update step size | Compare 0.001 and 0.3 |
| Optimizer | Determines how weights are updated | Compare Adam, SGD and RMSprop |
| Epochs / batch size | Controls training duration and updates | Watch the curves flatten |
| L2 regularisation | Penalises large weights | Add L2 when overfitting appears |

### Suggested experiments
- **XOR:** compare 1 hidden neuron with 4+ neurons.
- **Circle:** compare 2 neurons with 8 neurons.
- **Spiral:** increase hidden layers and neurons.
""")

with tab_about:
    st.markdown("""
### About this project
**Course activity:** ANN – Activity 2 · *Deploy an Interactive Neural Network learning demo (web-based, TensorFlow)*

**Tech stack:** TensorFlow / Keras · Streamlit · scikit-learn · Matplotlib · NumPy

**Deployment:** GitHub repository + Streamlit Community Cloud
""")
