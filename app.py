from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

app = Flask(__name__)

# 正确的训练数据
W = np.array([0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1]).reshape(-1,1)
X_feature = np.array([19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,20.2,
                      27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8]).reshape(-1,1)
WX = np.hstack((W, X_feature))

y = np.array([137,118,124,124,120,129,122,142,128,114,
              132,130,130,112,132,117,134,132,121,128])

model = LinearRegression().fit(WX, y)

@app.route("/predict")
def predict():
    w = int(request.args.get("w", 0))  # 读取w
    x = float(request.args.get("x", 0))  # 读取x
    input_data = np.array([[w, x]])  # 必须是二维数组 [[w, x]]
    y_pred = model.predict(input_data)[0]
    
    return jsonify({"w": w, "x": x, "prediction": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
