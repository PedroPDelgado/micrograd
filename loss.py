def rmse(ys: list, y_preds: list):
    return (sum((y - y_pred)**2 for y, y_pred in zip(ys, y_preds)) / len(ys)) ** 0.5