import numpy as np
from evaluate import load
import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
import copy
from torchvision import transforms
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def process_example(example, processor):
    image = example['image'].convert("RGB")
    inputs = processor(image, return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs

def build_transform(processor, train_augmentation = None, is_train=True):
    if is_train:
        if train_augmentation == None:
            augmentation = transforms.Compose([])
        else:
            augmentation = train_augmentation
    else:
        augmentation = transforms.Compose([])

    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        images = [augmentation(x.convert("RGB")) for x in example_batch['image']]
        inputs = processor(images, return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']
        return inputs
    return transform

def load_data(processor, link='timm/oxford-iiit-pet', augmentation=None):
    ds = load_dataset(link) 
    prepared_ds = ds.with_transform(build_transform(processor, augmentation))
    label = ds['train'].features['label'].names
    return prepared_ds, label

def load_model(model_name_or_path, label):
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label),
        id2label={str(i): c for i, c in enumerate(label)},
        label2id={c: str(i) for i, c in enumerate(label)},
        ignore_mismatched_sizes=True
    )
    for param in model.vit.parameters():
        param.requires_grad = False
    return model

def load_processor(model_name_or_path):
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return processor

# prepared_ds["test"]
def split_data(data, size=0.5):
    split_validation = data.train_test_split(
        test_size=size,
        shuffle=True,
        seed=42
    )

    prepared_val_ds = split_validation["train"]
    prepared_test_ds = split_validation["test"]

    return prepared_val_ds, prepared_test_ds

# Training and Evaluation
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p):
    metric = load("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

class GradualUnfreezeCallback(TrainerCallback):
    def __init__(self, start_epoch=5, unfreeze_interval=2):
        self.start_epoch = start_epoch
        self.unfreeze_interval = unfreeze_interval
        self.current_unfrozen = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch < self.start_epoch:
            return
        if (state.epoch - self.start_epoch) % self.unfreeze_interval == 0:
            model = kwargs['model']
            total_layers = len(model.vit.encoder.layer)
            layers_to_unfreeze = self.current_unfrozen + 1
            if layers_to_unfreeze > total_layers:
                return
            # Unfreeze from the last layer backward
            for layer in model.vit.encoder.layer[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {layers_to_unfreeze} layers at epoch {state.epoch}")
            self.current_unfrozen = layers_to_unfreeze

class CustomTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            # Split parameters into classifier and ViT groups
            classifier_params = []
            vit_params = []
            for name, param in opt_model.named_parameters():
                if "classifier" in name:
                    classifier_params.append(param)
                elif "vit" in name:
                    vit_params.append(param)
            
            optimizer_grouped_parameters = [
                {
                    "params": classifier_params,
                    "lr": self.args.learning_rate,  # Higher LR for classifier
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": vit_params,
                    "lr": self.args.learning_rate / 10,
                    "weight_decay": self.args.weight_decay / 10,
                },
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    
def train(original_model, processor, train_set, val_set, test_set, training_args):
    model = copy.deepcopy(original_model)
    model = model.to(device)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics,
    #     train_dataset=train_set,
    #     eval_dataset=val_set,
    #     processing_class=processor,
    # )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=processor,
        callbacks=[GradualUnfreezeCallback(start_epoch=5, unfreeze_interval=2), EarlyStoppingCallback(early_stopping_patience=5)],
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(test_set)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    logs = copy.deepcopy(trainer.state.log_history)
    torch.cuda.empty_cache()

    return logs

def plot_log(logs):
    print('plot_log start')
    print(logs)
    # Extract evaluation metrics
    eval_logs = [log for log in logs if 'eval_loss' in log]
    epochs = [log['epoch'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]
    eval_accuracies = [log['eval_accuracy'] for log in eval_logs]
    
    # Extract training loss (filter out evaluation entries)
    train_logs = [log for log in logs if 'loss' in log and 'eval_loss' not in log]
    train_epochs = [log['epoch'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]

    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, eval_losses, label='Validation Loss')
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig('training_metrics.png')
    plt.show()
    plt.close()
    print('plot_log end')
    
def main(training_args, model_name_or_path, augmentation, isPlot=True):
    # model_name_or_path = 'google/vit-base-patch16-224-in21k'
    # model_name_or_path = 'google/vit-base-patch16-384'
    processor = load_processor(model_name_or_path)
    prepared_ds, label = load_data(processor, 'timm/oxford-iiit-pet', augmentation)
    model = load_model(model_name_or_path, label)
    prepared_train = prepared_ds['train'].with_transform(build_transform(processor, is_train=True))
    prepared_test = prepared_ds['test'].with_transform(build_transform(processor, is_train=False))
    train_set, val_set = split_data(prepared_train, 0.2)
    test_set = prepared_test

    logs = train(model, processor, train_set, val_set, test_set, training_args)

    if isPlot:
        plot_log(logs)

