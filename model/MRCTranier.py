#-*- coding: utf-8 -*-
from transformers import Trainer

class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    value = metrics.pop(key)
                    metrics[f"{metric_key_prefix}_{key}"] =value

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    """
    def train(
        self,
        ##python main.py --resume 270851bert-base/checkpoint-500/
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
    
    train 함수에서 resume_from_checkpoint 인자로 할 수 있는 것 
    1. resume_from_checkpoint(str ckpt path)안에 파일 접근하기
    2. config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME)) #CONFIG_NAME = "config.json"
    3. state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu") #WEIGHTS_NAME = "pytorch_model.bin"
    4. self._load_optimizer_and_scheduler(resume_from_checkpoint)
    5. self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
    6. self._load_rng_state(resume_from_checkpoint) #rng_state.pth 
    """