from plotly import graph_objects as go
import sklearn.metrics as metrics


def draw_confusion_matrix(predicted, real, class_names):
    """
    Takes inputs of predicted and real labels with class names and draws
    confusion matrix
    """
    confusion_matrix = metrics.confusion_matrix(y_true=real, y_pred=predicted)
    fig_cm = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=class_names,
        y=class_names,
    ))
    fig_cm.update_xaxes(type='category')
    fig_cm.update_yaxes(type='category')
    fig_cm.show()