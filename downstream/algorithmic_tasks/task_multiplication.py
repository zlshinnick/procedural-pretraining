train_steps = 250
n_embd = 512
n_layer = 4
n_head = 8
n_positions = 40
weight_decay = 0.001
warmup_steps = 500
save_steps = 600
eval_steps = 40
logging_steps = 600
train_size = 40000
eval_size = 4000
vocab_size = 31
dropout = 0.0
per_device_train_batch_size = 64  # 1000
per_device_eval_batch_size = 64  # 1000

task_split = task.split("_")
assert len(task_split) in [2, 3]
min_len = int(task_split[1])
max_len = min_len if len(task_split) == 2 else int(task_split[2])


def drawnum():
    o = random.randint(min_len, max_len)
    return random.randint(10 ** (o - 1), 10**o - 1)


# Multiplication results can be much longer, so increase the max length
max_lenl = max_len * 5 + 4  # Increased buffer for longer multiplication results


def gen_single():
    a = drawnum()
    b = drawnum()
    # Modified to use multiplication instead of addition
    L = str(a) + "*" + str(b) + "="
    R = str(a * b)
    # Convert to token indices, updating the operator mapping
    l = (
        [int(x) if "0" <= x <= "9" else {"*": 10, "=": 11}[x] for x in L]
        + [int(x) + 20 for x in R]
        + [30]
    )  # 30=stop
    # print(len(l), max_lenl)
    assert len(l) <= max_lenl
    l += [12] * (max_lenl - len(l))
    return l


def taskA_gen():
    tasks = []
    for i in range(eval_size):
        tasks.append(gen_single())
    return tasks


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    wanted = labels[..., :-1] >= 20
    predictions = np.argmax(predictions, axis=-1)[..., :-1]
    corr = (predictions == labels[..., 1:]) * wanted
    return {"accuracy": np.sum(corr) / np.sum(wanted)}


class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        checkpoint_folder = os.path.join(
            self.args.output_dir, f"checkpoint-{self.state.global_step}"
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        return super()._save_checkpoint(model, trial)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        wanted = labels[:, :-1].detach().flatten() >= 20
        logits = logits[:, :-1]
        labels = labels[:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        ws = wanted.sum()
        loss = (loss * wanted).sum() / ws
        return (loss, outputs) if return_outputs else loss
