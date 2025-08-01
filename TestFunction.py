import torch

def test_model(model, test_loader, device, *metrics_functions):

    model.eval()

    total_outputs = []
    total_labels = []
    total_fst = []
    total_ids = []
    metrics = {}

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            ids = batch['img_id']
            fst = batch['fst']
            labels = batch['diagnosis'].to(device) 
            
            outputs = model(images)   

            while outputs.dim() > 2:
                if outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                else:
                    break

            total_outputs.append(outputs)
            total_labels.append(labels)
            total_fst.append(fst)
            total_ids.extend(ids)

    total_outputs = torch.cat(total_outputs, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    total_fst = torch.cat(total_fst, dim=0)

    for metric_function in metrics_functions:
        metric_name, metric_value = metric_function(total_outputs, total_labels, total_fst, total_ids)
        metrics[metric_name] = metric_value
                
    return metrics