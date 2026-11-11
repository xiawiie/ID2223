from .training import evaluate

def compare(model_a,model_b,loaders,device):
    rows=[]
    for name,model in [("ResNet basic",model_a),("ResNet augmented",model_b)]:
        clean=evaluate(model,loaders["test"],device)[1]; noise=evaluate(model,loaders["noise"],device)[1]; rotation=evaluate(model,loaders["rotation"],device)[1]
        rows.append({"name":name,"clean_accuracy":clean,"noise_accuracy":noise,"rotation_accuracy":rotation,"noise_drop":clean-noise,"rotation_drop":clean-rotation})
    return rows
