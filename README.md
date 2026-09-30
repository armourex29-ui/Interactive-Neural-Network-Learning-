# ANN Activity 2 — Interactive Neural Network Learning Demo

A web-based TensorFlow/Keras + Streamlit neural-network playground inspired by the supplied Activity 2 reference.

## Features
- XOR, Circle and Spiral datasets
- Adjustable samples, noise and test split
- 1–4 hidden layers
- Adjustable neurons and activation function
- Adam, SGD and RMSprop
- Learning-rate, epochs, batch-size and L2 controls
- Live decision-boundary updates during training
- Live loss curve
- Train/test accuracy and loss
- Learned network-weight visualization
- Confusion matrix
- Model summary
- Interactive point prediction

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit Cloud
Upload these files to one GitHub repository:
- `app.py`
- `nn_utils.py`
- `requirements.txt`
- `README.md`

Then deploy `app.py` using Streamlit Community Cloud and copy the public URL into your PPT.

## Suggested 5-minute demo
1. Open the Streamlit app.
2. Explain the neural-network concept.
3. Select XOR and show the initial dataset.
4. Set 2 hidden layers and press **Train network**.
5. Show the live loss curve and decision boundary.
6. Show train/test accuracy.
7. Open Network weights and Confusion matrix.
8. Try your own point.
9. Switch to Circle or Spiral and change the network width/depth.
10. Finish with the GitHub and Streamlit links.
